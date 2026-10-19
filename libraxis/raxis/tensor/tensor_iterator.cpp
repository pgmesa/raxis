
#include <raxis/tensor/tensor_iterator.h>

#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <numeric>

#include <raxis/core/dtype.h>
#include <raxis/core/types.h>

#include <raxis/util/exception.h>
#include <raxis/util/range.h>
#include <raxis/util/function_ref.h>
#include <raxis/util/maybe_owned.h>
#include <raxis/util/parallel.h>
#include <raxis/util/small_buffer.h>
#include <raxis/util/array_ref.h>
#include <raxis/util/small_vector.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/type_promotion.h>
#include <raxis/tensor/memory_overlap.h>
#include <raxis/tensor/detail/dim_counter.h>


namespace raxis {

namespace {

void get_base_ptrs(char** ptrs, ArrayRef<Operand> operands);
void get_strides(int64_t* strides, ArrayRef<Operand> operands, int ndim);
void get_data_ptrs(
    char** ptrs,
    ArrayRef<char*> base_ptrs,
    IntArrayRef strides,
    IntArrayRef counter_values
);
    
} // namespace


using loop2d_t = TensorIterator::loop2d_t;
using PtrVector = TensorIterator::PtrVector;
using StrideVector = TensorIterator::StrideVector;

void Operand::set_tensor(MaybeOwned<Tensor>&& new_tensor) {
    RAXIS_INTERNAL_ASSERT(!original_tensor_, "Cannot set tensor on an exchanged tensor");
    tensor_ = std::move(new_tensor);
}

void Operand::exchange_tensor(MaybeOwned<Tensor>&& new_tensor) {
    RAXIS_INTERNAL_ASSERT(!original_tensor_, "Tensor already exchanged");
    original_tensor_ = std::move(tensor_);
    tensor_ = std::move(new_tensor);
}

void Operand::restore_original_tensor() {
    RAXIS_INTERNAL_ASSERT(!original_tensor_, "No original tensor to restore");
    tensor_ = std::move(*original_tensor_);
    original_tensor_.reset();
}

bool Operand::has_exchanged() const {
    return original_tensor_.has_value();
}


// ======================= Tensor Iterator Config =======================
TensorIteratorConfig& TensorIteratorConfig::add_borrowed_output(const Tensor& output) {
    RAXIS_INTERNAL_ASSERT(
      num_inputs_ == 0,
      "Keep in mind that you have to add all outputs first before adding any input");
    tensors_.push_back(MaybeOwned<Tensor>::borrowed(output));
    num_outputs_++;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_borrowed_input(const Tensor& input) {
    tensors_.push_back(MaybeOwned<Tensor>::borrowed(input));
    num_inputs_++;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_owned_output(const Tensor& output) {
    RAXIS_INTERNAL_ASSERT(
      num_inputs_ == 0,
      "Keep in mind that you have to add all outputs first before adding any input");
    tensors_.push_back(MaybeOwned<Tensor>::owned(output));
    num_outputs_++;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_owned_input(const Tensor& input) {
    tensors_.push_back(MaybeOwned<Tensor>::owned(input));
    num_inputs_++;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_owned_output(Tensor&& output) {
    RAXIS_INTERNAL_ASSERT(
      num_inputs_ == 0,
      "Keep in mind that you have to add all outputs first before adding any input");
    tensors_.push_back(MaybeOwned<Tensor>::owned(std::move(output)));
    num_outputs_++;
    return *this;
}

TensorIteratorConfig& TensorIteratorConfig::add_owned_input(Tensor&& input) {
    tensors_.push_back(MaybeOwned<Tensor>::owned(std::move(input)));
    num_inputs_++;
    return *this;
}

TensorIterator TensorIteratorConfig::build() {
    TensorIterator iter;
    iter.build(*this);
    return iter;
}

// ========================== Tensor Iterator ==========================
void TensorIterator::cast_outputs() {
    for (int out_idx = 0; out_idx < num_outputs_; out_idx++) {
        Operand& op = operands_[out_idx];
        if (!op.has_exchanged()) continue;
        
        const Tensor& original_tensor = op.original_tensor();
        const Tensor& tensor = op.tensor();

        original_tensor.copy_(tensor);
        op.restore_original_tensor();
    }
}

void TensorIterator::build(TensorIteratorConfig& config) {
    // Populate some persistent configuration fields
    is_reduction_ = config.is_reduction_;
    enforce_linear_iteration_ = config.enforce_linear_iteration_;

    // Fill in operands_ based on configuration
    populate_operands(config);
    // Check that the outputs can be written without overlapping writes or reads
    compute_mem_overlaps(config);
    // Compute the broadcasted shape
    compute_shape(config);
    // Compute the result dtype and device
    compute_types(config);

    if (!fast_setup()) {
        // Compute each tensor's stride after broadcasting
        compute_strides(config);
        // Re-order dimensions to improve coalescing
        reorder_dimensions();
        // Allocate the output tensor if it's not provided
        allocate_outputs();
        // Coalesce adjacent dimensions when possible
        coalesce_dimensions();
    }

    for (auto& op : operands_) {
        RAXIS_INTERNAL_ASSERT(op.tensor().defined());
        op.data = static_cast<char*>(op.tensor().mutable_data_ptr());
    }

    // zero out offsets
    // If the tensor is a scalar, we leave room for it
    // So index translations in reduction can access
    // a valid value for the offset
    int64_t ndim_offsets = (ndim() ? ndim() : 1);
    view_offsets_ = DimVector(ndim_offsets, 0);
}

void TensorIterator::populate_operands(TensorIteratorConfig& config) {
    num_outputs_ = config.num_outputs_;
    size_t num_operands = config.tensors_.size();
    
    for (size_t idx = 0; idx < num_operands; idx++) {
        auto& tensor = config.tensors_[idx];
        operands_.emplace_back(std::move(tensor));
        
        // check if tensor is an output
        if (idx >= num_outputs_) continue;
        
        // mark tensor as output
        operands_[idx].is_output = true;
        const Tensor& output = operands_[idx].tensor();
        if (!output.defined()) continue;

        // check if output is also an input
        for (size_t inp_idx = num_outputs_; inp_idx < num_operands; inp_idx++) {
            const Tensor& input = *config.tensors_[inp_idx];
            if (output.is_same(input)) {
                operands_[idx].is_read_write = true;
                break;
            }
        }
    }
}

void TensorIterator::compute_mem_overlaps(const TensorIteratorConfig& config) {
    if (!config.check_mem_overlap_) return;

    for (int out_idx = 0; out_idx < num_outputs_; out_idx++) {
        const Tensor& output = operands_[out_idx].tensor();
        if (!output.defined()) continue;
        assert_no_internal_overlap(output);
        for (int inp_idx = num_outputs_; inp_idx < num_operands(); inp_idx++) {
            const Tensor& input = operands_[inp_idx].tensor();
            if (!input.is_same(output)) {
                assert_no_partial_overlap(output, input);
            }
        }
    }
}

void TensorIterator::compute_shape(const TensorIteratorConfig& config) {
    if (config.static_shape_.has_value()) {
        shape_ = config.static_shape_.value();
        return;
    }

    all_ops_same_shape_ = true;
    bool has_scalars = false;
    bool has_tensors = false;

    // First pass: compute shape from inputs and defined outputs (with broadcasting)
    for (Operand& op: operands_) {
        if (!op.tensor().defined()) continue;
        
        IntArrayRef shape = op.tensor().shape();
        if (shape.empty()) has_scalars = true;
        else has_tensors = true;

        if (has_scalars && has_tensors) 
            all_ops_same_shape_ = false;

        if (shape_.empty()) {
            shape_ = DimVector(shape);
        } else if (shape != shape_) {
            all_ops_same_shape_ = false;
            shape_ = broadcast_shapes(shape_, op.tensor().shape());
        }
    }
    all_ops_are_scalars_ = !has_tensors;
}

void TensorIterator::compute_strides(const TensorIteratorConfig& config) {
    for (Operand& op: operands_) {
        if (!op.tensor().defined()) continue;

        IntArrayRef original_shape = config.static_shape_? shape_ : op.tensor().shape();
        IntArrayRef original_stride = op.tensor().strides();
        int64_t itemsize = static_cast<int64_t>(op.tensor().dtype().itemsize());
        
        int original_ndim = static_cast<int>(original_shape.size());
        int target_ndim = ndim();
        int offset = target_ndim - original_ndim;
        
        if (offset > 0)
            op.stride_bytes.resize(target_ndim, 0);
        else
            op.stride_bytes.resize(target_ndim);

        for (int i = 0; i < original_ndim; ++i) {
            int ti = offset + i;
            if (original_shape[i] == 1 && shape_[ti] != 1) {
                op.stride_bytes[ti] = 0;
            } else {
                op.stride_bytes[ti] = original_stride[i] * itemsize;
            }
        }
    }
}

void TensorIterator::compute_types(const TensorIteratorConfig& config) {
    // Reviews operands (1/2)
    //   - validates that all input tensors are defined
    //   - computes common device
    //   - determines if there are undefined outputs
    //   - determines if there are different dtypes and attempts
    //       to quickly acquire a common dtype
    Device common_device = DeviceType::CPU;
    common_dtype_ = dtypes::Undefined;
    Dtype output_dtype = dtypes::Undefined;
    bool has_different_input_dtypes = false;
    bool has_different_output_dtypes = false;
    bool has_undefined_outputs = false;
    
    for (auto& op : operands_) {
        if (!op.is_type_defined()) {
            RAXIS_INTERNAL_ASSERT(op.is_output, "Found type undefined input tensor");

            if (config.static_dtype_.has_value()) {
                op.target_dtype = config.static_dtype_.value();
            } else {
                has_undefined_outputs = true;
            }

            if (config.static_device_.has_value()) {
                op.device = config.static_device_.value();
            } else {
                RAXIS_INTERNAL_ASSERT(config.check_all_same_device_);
            }

            if (has_undefined_outputs || !op.device.has_value()) {
                continue;
            }
        }

        // Validates input tensors are defined
        if (!op.tensor().defined()) {
            RAXIS_INTERNAL_ASSERT(op.is_output, "Found undefined input tensor");
            continue;
        }

        RAXIS_INTERNAL_ASSERT(op.target_dtype == op.current_dtype);

        // Acquires the first non-CPU device (if any) as the common device
        if (common_device == DeviceType::CPU && !op.tensor().is_cpu()) {
            common_device = op.tensor().device();
        }

        if (!op.is_output) {
            // Determines if there are varying input dtypes
            // NOTE: the common dtype is set to the first defined input dtype observed
            if (op.target_dtype != common_dtype_) {
                if (common_dtype_ == dtypes::Undefined) {
                    common_dtype_ = op.target_dtype;
                } else {
                    has_different_input_dtypes = true;
                }
            }
        } else { // op.is_output = true
            // Determines if there are varying output dtypes
            // NOTE: the output dtype is set to the first defined output dtype observed
            if (op.target_dtype != output_dtype) {
                if (output_dtype == dtypes::Undefined) {
                    output_dtype = op.target_dtype;
                } else {
                    has_different_output_dtypes = true;
                }
            }
        }
    }

    RAXIS_INTERNAL_ASSERT(!(has_different_input_dtypes && !config.promote_inputs_to_common_dtype_ &&
                        (has_undefined_outputs || config.enforce_safe_casting_to_output_ ||
                        config.cast_common_dtype_to_outputs_)));

    // Checks that all inputs and defined outputs are the same dtype, if requested
    if (config.check_all_same_dtype_ &&
        (has_different_input_dtypes || has_different_output_dtypes ||
        (common_dtype_ != output_dtype && output_dtype != dtypes::Undefined))) {
        // Throws an informative error message
        for (auto& op : operands_) {
            if (!op.tensor().defined()) {
                continue;
            }

            RAXIS_CHECK(op.target_dtype == common_dtype_, 
                "Found dtype", op.target_dtype, " but expected ", common_dtype_);
        }
    }

    // Short-circuits if no additional work required. Not when devices are
    // checked: the loop below also gives undefined outputs their device
    if (!has_undefined_outputs && !config.check_all_same_device_ &&
        !config.promote_inputs_to_common_dtype_ && !config.cast_common_dtype_to_outputs_ &&
        !config.enforce_safe_casting_to_output_) {
        // Invalidates common_dtype_ if it could not be inferred
        common_dtype_ = has_different_input_dtypes ? dtypes::Undefined : common_dtype_;
        return;
    }
    
    // Computes a common dtype, if needed
    if ((has_different_input_dtypes || all_ops_are_scalars_) && config.promote_inputs_to_common_dtype_) {
        common_dtype_ = compute_common_dtype();
    }

    // Promotes common dtype to the default float scalar type, if needed
    if (config.promote_integer_inputs_to_float_ && common_dtype_.is_integral(/*include_bool=*/true)) {
        common_dtype_ = default_float_dtype();
    }
    
    // Reviews operands (2/2)
    //   - sets metadata for undefined outputs
    //   - checks that all tensors are on the same device, if requested
    //   - checks that the common dtype can safely cast to each output, if requested
    //   - creates temporaries for CPU operations, if needed and requested
    common_device_ = common_device;
    int max_cpu_scalars_on_non_cpu = config.allow_cpu_scalars_ ? 1 : 0;
    int current_cpu_scalars_on_non_cpu = 0;
    for (auto& op : operands_) {
        bool is_type_defined = op.is_type_defined();
        bool is_device_defined = op.is_device_defined();

        if (!is_type_defined) {
            op.target_dtype = common_dtype_;
        }

        if (!is_device_defined) {
            op.device = common_device;
        }

        if (!is_type_defined && !is_device_defined) {
            continue;
        }

        // Skips undefined tensors
        if (!op.tensor().defined()) continue;

        // Checks all tensors are on the same device, if requested
        if (config.check_all_same_device_) {
            // Handles CPU scalars on CUDA kernels that support them
            if (!common_device.is_cpu() &&
                config.allow_cpu_scalars_ && !op.is_output && op.tensor().ndim() == 0 &&
                op.tensor().is_cpu()) {
                RAXIS_CHECK(current_cpu_scalars_on_non_cpu < max_cpu_scalars_on_non_cpu,
                            "Trying to pass too many CPU scalars to non-CPU kernel!");
                ++current_cpu_scalars_on_non_cpu;
            } else {
                RAXIS_CHECK(op.device == common_device,
                            "Expected all tensors to be on the same device, but "
                            "found at least two devices, ", common_device, " and ", op.device.value(), "!");
            }
        }

        // Checks safe casting, if requested
        if (config.enforce_safe_casting_to_output_ && op.is_output && op.current_dtype != common_dtype_) {
            RAXIS_CHECK(can_cast(common_dtype_, op.current_dtype),
                  "result type ", common_dtype_, " can't be cast to the "
                  "desired output type ", op.current_dtype);
        }

        // Creates temporaries for CPU operations, if needed and requested
        if (common_device == DeviceType::CPU) {
            if (config.cast_common_dtype_to_outputs_ && op.is_output && op.current_dtype != common_dtype_) {
                RAXIS_INTERNAL_ASSERT(op.tensor().defined());
                op.exchange_tensor(MaybeOwned<Tensor>::owned(
                    empty_like(op.tensor(), common_dtype_, MemoryFormat::Contiguous)
                ));
                op.current_dtype = common_dtype_;
                op.target_dtype = common_dtype_;
            }
        }

        // Promotes inputs by creating temporaries of the correct dtype
        if (config.promote_inputs_to_common_dtype_ && !op.is_output && op.current_dtype != common_dtype_) {
            op.exchange_tensor(MaybeOwned<Tensor>::owned(op.tensor().to(common_dtype_)));
            op.current_dtype = common_dtype_;
            op.target_dtype = common_dtype_;
        }
    }
}

Dtype TensorIterator::compute_common_dtype() {
    PromotionState state;
    for (auto& op : operands_) {
        if (op.is_output) continue;
        state = update_promotion_state(op.tensor(), state);
    }
    return compute_promoted_dtype(state);
}

void TensorIterator::permute_dimensions(IntArrayRef perm) {
    auto reorder = [perm]<typename Vec>(const Vec& data) {
        Vec res(data.size(), 0);
        for (size_t dim = 0; dim < perm.size(); dim++) {
            res[dim] = data[perm[dim]];
        }
        return res;
    };

    // Update shape and strides
    shape_ = reorder(shape_);
    for (auto& op : operands_) {
        if (!op.stride_bytes.empty()) {
            op.stride_bytes = reorder(op.stride_bytes);
        }
    }
}

bool TensorIterator::fast_setup() {
    if (is_reduction_ || !all_ops_same_shape_) {
        return false;
    }

    // Check all defined operands are contiguous
    // Undefined outputs are skipped, they will be allocated contiguously
    for (const Operand& op : operands_) {
        if (!op.tensor().defined()) {
            continue;
        }
        if (!op.tensor().is_contiguous()) {
            return false;
        }
    }

    // Allocate undefined outputs as contiguous
    for (int i = 0; i < num_outputs_; i++) {
        Operand& op = operands_[i];
        if (!op.tensor().defined()) {
            op.set_tensor(MaybeOwned<Tensor>::owned(empty(shape_, op.target_dtype)));
        }
        op.current_dtype = op.target_dtype;
    }

    // Collapse to 1D: shape = [numel], stride = [itemsize] for each operand
    int64_t n = numel();
    shape_ = {n};
    for (Operand& op : operands_) {
        int64_t itemsize = static_cast<int64_t>(op.tensor().dtype().itemsize());
        op.stride_bytes = {itemsize};
    }

    has_coalesced_dimensions_ = true;
    return true;
}

void TensorIterator::reorder_dimensions() {
    perm_.resize(ndim());
    if (ndim() == 1) {
        perm_[0] = 0;
        return;
    }

    // initialize perm with n-1, n-2, ..., 1, 0
    std::iota(perm_.rbegin(), perm_.rend(), 0);

    if (enforce_linear_iteration_) {
        permute_dimensions(perm_);
        return;
    }

    // Returns 1 if the dim0 should come after dim1, -1 if dim0 should come
    // before dim1, and 0 if the comparison is ambiguous.
    auto should_swap = [&](size_t dim0, size_t dim1) {
        for (size_t idx = 0; idx < num_operands(); idx++) {
            // Ignore undefined
            if (operands_[idx].stride_bytes.empty()) {
                continue;
            }
            int64_t stride0 = operands_[idx].stride_bytes[dim0];
            int64_t stride1 = operands_[idx].stride_bytes[dim1];
            if (is_reduction_ && operands_[idx].is_output) {
                // move reduced dimensions to the front
                // strides of reduced dimensions are always set to 0 by review_reduce_result
                if ((stride0 == 0) != (stride1 == 0)) {
                    return stride1 == 0 ? 1 : -1;
                }
            }
            // move on to the next input if one of the dimensions is broadcasted
            if (stride0 == 0 || stride1 == 0) {
                continue;
            // it is important to return here only with strict comparisons, for equal strides we try to break the tie later
            // by comparing corresponding dimensions or if that does not work, moving on to the next tensor
            } else if (stride0 < stride1) {
                return -1;
            } else  if (stride0 > stride1) {
                return 1;
            } else { //equal strides, use dimensions themselves as the tie-breaker.
                //at this point, with zero strides out of the way, we are guaranteed that operand dimensions are equal to shape_
                auto t_dim0 = shape_[dim0];
                auto t_dim1 = shape_[dim1];
                //return only if dimensions should be swapped, otherwise move on to the next tensor
                if (t_dim0 > t_dim1) {
                    return 1;
                }
            }
        }
        return 0;
    };

    // Insertion sort with support for ambiguous comparisons
    for (int i = 1; i < ndim(); i++) {
        int dim1 = i;
        for (int dim0 = i - 1; dim0 >= 0; dim0--) {
            int comparison = should_swap(perm_[dim0], perm_[dim1]);
            if (comparison > 0) {
                std::swap(perm_[dim0], perm_[dim1]);
                dim1 = dim0;
            } else if (comparison < 0) {
                break;
            }
        }
    }

    // Perform re-ordering of shape and strides
    permute_dimensions(perm_);
}

StrideVector TensorIterator::compatible_stride(int64_t element_size) const {
    StrideVector stride;
    int64_t next_stride = element_size;
    for (size_t dim = 0; dim < ndim(); dim++) {
        stride.push_back(next_stride);
        next_stride *= shape_[dim];
    }
    return stride;
}

DimVector TensorIterator::invert_perm(IntArrayRef input) const {
    // Invert the permutation caused by reorder_dimensions. This is not valid
    // after coalesce_dimensions is called
    RAXIS_INTERNAL_ASSERT(!has_coalesced_dimensions_);
    RAXIS_INTERNAL_ASSERT(input.size() == perm_.size());

    DimVector res(input.size());
    for (size_t dim = 0; dim < ndim(); dim++) {
        res[perm_[dim]] = input[dim];
    }
    return res;
}

void TensorIterator::set_output(int64_t output_idx, IntArrayRef shape, IntArrayRef strides) {
    Operand& op = operands_[output_idx];
    if (op.tensor().defined()) return;
    
    if (strides.empty()) {
        op.set_tensor(MaybeOwned<Tensor>::owned(empty(shape, op.target_dtype)));
    } else {
        op.set_tensor(MaybeOwned<Tensor>::owned(empty_strided(shape, strides, op.target_dtype)));
    }

}

void TensorIterator::allocate_outputs() {
    for (int out_idx = 0; out_idx < num_outputs_; out_idx++) {
        Operand& op = operands_[out_idx];
        if (op.tensor().defined()) continue;
        
        RAXIS_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", out_idx);

        size_t element_size = op.target_dtype.itemsize();
        op.stride_bytes = compatible_stride(static_cast<int64_t>(element_size));
        // check if permutation is just an inverted order
        bool inverted = true;
        for (int j = 0; j < ndim(); j++) {
            if (perm_[j] != ndim() - j - 1) {
                inverted = false;
                break;
            }
        }
        auto tensor_shape = invert_perm(shape_);
        if (inverted) {
            // can just return contiguous output
            // it is faster because it avoids allocating 0 size tensor and
            // resizing and restriding it
            set_output(out_idx, tensor_shape, {});
        } else {
            auto tensor_stride = invert_perm(op.stride_bytes);
            for (size_t dim = 0; dim < ndim(); dim++) {
                tensor_stride[dim] /= static_cast<int64_t>(element_size);
            }
            set_output(out_idx, tensor_shape, tensor_stride);
        }
        
        op.current_dtype = op.target_dtype;
    }
}

void TensorIterator::coalesce_dimensions() {
    if (ndim() <= 1) return;

    auto can_coalesce = [&](int dim0, int dim1) {
        int64_t size0 = shape_[dim0];
        int64_t size1 = shape_[dim1];
        if (size0 == 1 || size1 == 1) {
            return true;
        }
        for (int i = 0; i < num_operands(); i++) {
            auto& stride = operands_[i].stride_bytes;
            if (size0 * stride[dim0] != stride[dim1]) {
                return false;
            }
        }
        return true;
    };

    auto replace_stride = [&](int dim0, int dim1) {
        for (int i = 0; i < num_operands(); i++) {
            auto& stride = operands_[i].stride_bytes;
            stride[dim0] = stride[dim1];
        }
    };

    int prev_dim = 0;
    for (int dim = 1; dim < ndim(); dim++) {
        if (can_coalesce(prev_dim, dim)) {
            if (shape_[prev_dim] == 1) {
                replace_stride(prev_dim, dim);
            }
            shape_[prev_dim] *= shape_[dim];
        } else {
            prev_dim++;
            if (prev_dim != dim) {
                replace_stride(prev_dim, dim);
                shape_[prev_dim] = shape_[dim];
            }
        }
    }

    shape_.resize(prev_dim + 1);
    for (int i = 0; i < num_operands(); i++) {
        operands_[i].stride_bytes.resize(ndim());
    }

    has_coalesced_dimensions_ = true;
}


TensorIterator TensorIterator::reduce_op(Tensor& out, const Tensor& a) {
    RAXIS_INTERNAL_ASSERT(out.defined());
    return TensorIteratorConfig()
        .is_reduction(true)
        .promote_inputs_to_common_dtype(true)
        .add_owned_output(out)
        .add_owned_input(a)
        .build();
}

bool TensorIterator::is_dim_reduced(int dim) const {
    for (const auto& op : operands_) {
        if (op.is_output && op.stride_bytes[dim] == 0 && shape_[dim] > 1) {
            return true;
        }
    }
    return false;
}

PtrVector TensorIterator::get_base_ptrs() const {
    SmallVector<char*, 4> ptrs(num_operands());
    ::raxis::get_base_ptrs(ptrs.data(), operands_);
    return ptrs;
}

StrideVector TensorIterator::get_strides() const {
    int dim = ndim();
    StrideVector strides(num_operands() * static_cast<size_t>(std::max(dim, 2)));
    ::raxis::get_strides(strides.data(), operands_, dim);
    return strides;
}

void TensorIterator::narrow(int dim, int64_t start, int64_t size) {
    RAXIS_INTERNAL_ASSERT(dim < ndim() && size >= 1);
    shape_[dim] = size;
    view_offsets_[dim] += start;
    for (auto& op : operands_) {
        op.data = ((char*)op.data) + op.stride_bytes[dim] * start;
    }
    if (size == 1 && !is_reduction_) {
        coalesce_dimensions();
    }
}


void TensorIterator::for_each(loop2d_t loop, int64_t grain_size) const {
    int64_t numel = this->numel();

    if (numel == 0) {
        return;
    } else if (numel < grain_size || get_num_threads() == 1) {
        return serial_for_each(loop, {0, numel});
    } else {
        parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
            serial_for_each(loop, {begin, end});
        });
    }
}

void TensorIterator::serial_for_each(loop2d_t loop, Range range) const {
    if (range.size() == 0) return;

    const size_t noperands = num_operands();
    const int ndim = this->ndim();

    SmallBuffer<char*, 4> ptrs(noperands);
    SmallBuffer<int64_t, 8> strides(noperands * static_cast<size_t>(std::max(ndim, 2)));

    ::raxis::get_base_ptrs(ptrs.data(), operands_);
    ::raxis::get_strides(strides.data(), operands_, ndim);

    detail::serial_for_each(shape_, strides, ptrs.data(), noperands, loop, range);
}

namespace {

void get_base_ptrs(char** ptrs, ArrayRef<Operand> operands) {
    for (int idx = 0; idx < operands.size(); idx++) {
        ptrs[idx] = static_cast<char*>(operands[idx].data);
    }
}

void get_strides(int64_t* strides, ArrayRef<Operand> operands, int ndim) {
    int noperands = static_cast<int>(operands.size());
    for (int dim = 0; dim < ndim; dim++) {
        for (int idx = 0; idx < noperands; idx++) {
            *strides++ = operands[idx].stride_bytes[dim];
        }
    }

    if (ndim < 2) {
        std::fill_n(strides, (2 - ndim) * noperands, 0);
    }
}

void get_data_ptrs(
    char** ptrs,
    ArrayRef<char*> base_ptrs,
    IntArrayRef strides,
    IntArrayRef counter_values
) {
    size_t noperands = base_ptrs.size();
    size_t ndim = counter_values.size();
    std::copy(base_ptrs.begin(), base_ptrs.end(), ptrs);
    
    for (size_t dim = 0; dim < ndim; dim++) {
        int64_t value = counter_values[dim];
        for (size_t idx = 0; idx < noperands; idx++) {
            ptrs[idx] += value * strides[dim * noperands + idx];
        }
    }
}

}

namespace detail {

void serial_for_each(
    IntArrayRef shape,
    IntArrayRef strides,
    char** base_ptrs,
    size_t noperands,
    loop2d_t loop,
    Range range
) {
    size_t ndim = shape.size();
    RAXIS_DEBUG_ASSERT(strides.size() == noperands * std::max(size_t{2}, ndim));

    if (ndim <= 1) {
        if (range.begin == 0) {
            loop(base_ptrs, strides.data(), range.size(), 1);
        } else {
            SmallBuffer<char*, 4> ptrs(noperands);
            get_data_ptrs(ptrs.data(), {base_ptrs, noperands}, strides, {range.begin});
            loop(ptrs.data(), strides.data(), range.size(), 1);
        }
    } else {
        SmallBuffer<char*, 4> ptrs(noperands);
        detail::DimCounter counter(shape, range);
        while(!counter.is_done()) {
            get_data_ptrs(ptrs.data(), {base_ptrs, noperands}, strides, counter.values);
            auto step = counter.max_2d_step();
            loop(ptrs.data(), strides.data(), step[0], step[1]);
            counter.increment(step);
        }
    }
}

} // namespace detail

void TensorIterator::print_state(std::ostream& os) const {
    os << "=== TensorIterator state ===\n";
    os << "ndim: " << ndim() << "\n";
    os << "numel: " << numel() << "\n";
    os << "num_operands: " << num_operands() << "\n";
    os << "num_outputs: " << num_outputs_ << "\n";
    os << "common_dtype: " << common_dtype_.name() << "\n";
    os << "is_reduction: " << (all_ops_same_shape_? "true": "false") << "\n";
    os << "enforce_linear_iteration: " << (enforce_linear_iteration_? "true": "false") << "\n";
    os << "all_ops_same_shape: " << (all_ops_same_shape_? "true": "false") << "\n";
    os << "all_ops_are_scalars: " << (all_ops_are_scalars_? "true": "false") << "\n";
    
    os << "computational shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        os << shape_[i];
        if (i + 1 < shape_.size()) os << ", ";
    }
    os << "]\n";

    os << "permutations: [";
    for (size_t i = 0; i < perm_.size(); ++i) {
        os << perm_[i];
        if (i + 1 < perm_.size()) os << ", ";
    }
    os << "]\n";

    for (int op = 0; op < num_operands(); ++op) {
        const Operand& operand = operands_[op];
        os << "operand " << op;
        if (operand.is_output) os << " (output)";
        else os << " (input)";
        os << "\n";

        os << "  is_read_write=" << operand.is_read_write? "true": "false";
        os << "\n";

        os << "  original shape: [";
        const auto& shp = operand.tensor().shape();
        for (size_t i = 0; i < shp.size(); ++i) {
            os << shp[i];
            if (i + 1 < shp.size()) os << ", ";
        }
        os << "]\n";

        os << "  original strides: [";
        const auto& strides = operand.tensor().strides();
        for (size_t i = 0; i < strides.size(); ++i) {
            os << strides[i];
            if (i + 1 < strides.size()) os << ", ";
        }
        os << "]\n";

        os << "  current_dtype: " << operand.current_dtype.name();
        os << "\n";
        os << "  target_dtype: " << operand.target_dtype.name();
        os << "\n";

        os << "  computational stride_bytes: [";
        for (size_t i = 0; i < operand.stride_bytes.size(); ++i) {
            os << operand.stride_bytes[i];
            if (i + 1 < operand.stride_bytes.size()) os << ", ";
        }
        os << "]\n";
    }

    os << "===========================\n";
}

}