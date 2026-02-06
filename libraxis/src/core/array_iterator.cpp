
#include <raxis/core/array_iterator.hpp>

#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <numeric>

#include <fmt/format.h>

#include <raxis/array.hpp>
#include <raxis/factories.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/core/utils/range.hpp>
#include <raxis/core/utils/dim_counter.hpp>
#include <raxis/core/utils/function_ref.hpp>
#include <raxis/core/utils/maybe_owned.hpp>


namespace raxis {

void Operand::set_array(detail::MaybeOwned<NDArray>&& new_array) {
    if (original_array_) {
        throw std::runtime_error("Cannot set array on an exchanged array");
    }

    array_ = std::move(new_array);
}

void Operand::exchange_array(detail::MaybeOwned<NDArray>&& new_array) {
    if (original_array_) {
        throw std::runtime_error("Array already exchanged");
    } 
    original_array_ = std::move(array_);
    array_ = std::move(new_array);
}

void Operand::restore_original_array() {
    if (!original_array_) {
        throw std::runtime_error("No original array to restore");
    }
    array_ = std::move(*original_array_);
    original_array_.reset();
}

bool Operand::has_exchanged() const {
    return original_array_.has_value();
}

bool Operand::is_type_defined() const {
    return target_dtype != dtypes::Undefined;
}


// ======================= Array Iterator Config =======================
ArrayIteratorConfig& ArrayIteratorConfig::add_borrowed_output(const NDArray& output) {
    arrays_.push_back(detail::MaybeOwned<NDArray>::borrowed(output));
    num_outputs_++;
    return *this;
}

ArrayIteratorConfig& ArrayIteratorConfig::add_borrowed_input(const NDArray& input) {
    arrays_.push_back(detail::MaybeOwned<NDArray>::borrowed(input));
    num_inputs_++;
    return *this;
}

ArrayIteratorConfig& ArrayIteratorConfig::add_owned_output(const NDArray& output) {
    arrays_.push_back(detail::MaybeOwned<NDArray>::owned(output));
    num_outputs_++;
    return *this;
}

ArrayIteratorConfig& ArrayIteratorConfig::add_owned_input(const NDArray& input) {
    arrays_.push_back(detail::MaybeOwned<NDArray>::owned(input));
    num_inputs_++;
    return *this;
}

ArrayIteratorConfig& ArrayIteratorConfig::add_owned_output(NDArray&& output) {
    arrays_.push_back(detail::MaybeOwned<NDArray>::owned(std::move(output)));
    num_outputs_++;
    return *this;
}

ArrayIteratorConfig& ArrayIteratorConfig::add_owned_input(NDArray&& input) {
    arrays_.push_back(detail::MaybeOwned<NDArray>::owned(std::move(input)));
    num_inputs_++;
    return *this;
}

ArrayIterator ArrayIteratorConfig::build() {
    ArrayIterator iter;
    iter.build(*this);
    return iter;
}

// ========================== Array Iterator ==========================
void ArrayIterator::cast_outputs() {
    for (int out_idx = 0; out_idx < num_outputs_; out_idx++) {
        Operand& op = operands_[out_idx];
        if (!op.has_exchanged()) continue;
        
        const NDArray& original_array = op.original_array();
        const NDArray& array = op.array();

        original_array.copy_(array);
        op.restore_original_array();
    }
}

void ArrayIterator::build(ArrayIteratorConfig& config) {
    // Populate some persistent configuration fields
    is_reduction_ = config.is_reduction_;
    enforce_linear_iteration_ = config.enforce_linear_iteration_;

    // Fill in operands_ based on configuration
    populate_operands(config);
    // Compute the broadcasted shape
    compute_shape(config);
    // Compute the result dtype and device
    compute_types(config);
    // Compute each array's stride after broadcasting
    compute_strides(config);

    // Re-order dimensions to improve coalescing
    reorder_dimensions();
    // Allocate the output array if it's not provided
    allocate_outputs();
    // Coalesce adjacent dimensions when possible
    coalesce_dimensions();
}

void ArrayIterator::populate_operands(ArrayIteratorConfig& config) {
    num_outputs_ = config.num_outputs_;
    size_t num_operands = config.arrays_.size();
    
    for (size_t idx = 0; idx < num_operands; idx++) {
        auto& arr = config.arrays_[idx];
        operands_.emplace_back(std::move(arr));
        
        // check if tensor is an output
        if (idx >= num_outputs_) continue;
        
        // mark tensor as output
        operands_[idx].is_output = true;
        const NDArray& output = operands_[idx].array();
        if (!output.defined()) continue;

        // check if output is also an input
        for (size_t inp_idx = num_outputs_; inp_idx < num_operands; inp_idx++) {
            const NDArray& input = *config.arrays_[inp_idx];
            if (output.is_same(input)) {
                operands_[idx].is_read_write = true;
                break;
            }
        }
    }
}

void ArrayIterator::compute_shape(const ArrayIteratorConfig& config) {
    if (config.static_shape_.has_value()) {
        shape_ = config.static_shape_.value();
        return;
    }

    all_ops_same_shape_ = true;
    bool has_scalars = false;
    bool has_arrays = false;
    for (Operand& op: operands_) {
        if (!op.array().defined()) continue;
        
        // Outputs don't participate in shape computation
        if (op.is_output) continue;
        
        IntArray shape = op.array().shape();
        if (shape.empty()) has_scalars = true;
        else has_arrays = true;

        if (has_scalars && has_arrays) 
            all_ops_same_shape_ = false;

        if (shape_.empty()) {
            shape_ = shape;
        } else if (shape != shape_) {
            all_ops_same_shape_ = false;
            shape_ = broadcast_shapes(shape_, op.array().shape());
        }
    }
    all_ops_are_scalars_ = !has_arrays;

    // Ensure defined outputs have the correct shape
    for (size_t i = 0; i < num_outputs_; i++) {
        const NDArray& output = operands_[i].array();
        if (output.defined() && (output.shape() != shape_) && !is_reduction_) {
            throw std::runtime_error("Invalid shape encountered in output");
        }
    }
}

void ArrayIterator::compute_strides(const ArrayIteratorConfig& config) {
    for (Operand& op: operands_) {
        if (!op.array().defined()) continue;

        const IntArray& original_shape = config.static_shape_? shape_ : op.array().shape();
        const IntArray& original_stride = op.array().strides();
        int64_t itemsize = static_cast<int64_t>(op.array().dtype().itemsize());
        
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

void ArrayIterator::compute_types(const ArrayIteratorConfig& config) {
    // Reviews operands (1/2)
    //   - validates that all input arrays are defined
    //   - computes common device
    //   - determines if there are undefined outputs
    //   - determines if there are different dtypes and attempts
    //       to quickly acquire a common dtype
    common_dtype_ = dtypes::Undefined;
    Dtype output_dtype = dtypes::Undefined;
    bool has_different_input_dtypes = false;
    bool has_different_output_dtypes = false;
    bool has_undefined_outputs = false;
    
    for (auto& op : operands_) {
        if (!op.is_type_defined()) {
            if (!op.is_output) {
                throw std::runtime_error("Found type undefined input array");
            }

            if (config.static_dtype_.has_value()) {
                op.target_dtype = config.static_dtype_.value();
            } else {
                has_undefined_outputs = true;
            }

            if (has_undefined_outputs) {
                continue;
            }
        }

        // Validate type defined arrays are defined
        if (!op.array().defined()) {
            throw std::runtime_error("Found undefined input array");
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

    // Checks that all inputs and defined outputs are the same dtype, if requested
    if (config.check_all_same_dtype_ &&
        (has_different_input_dtypes || has_different_output_dtypes ||
        (common_dtype_ != output_dtype && output_dtype != dtypes::Undefined))) {
        // Throws an informative error message
        for (auto& op : operands_) {
            if (!op.array().defined()) {
                continue;
            }
            if (op.target_dtype != common_dtype_) {
                std::string msg = fmt::format(
                    "Found dtype {} but expected {}", 
                    op.target_dtype.name(), common_dtype_.name()
                );
                throw std::runtime_error(msg);
            }
        }
    }

    // Short-circuits if no additional work required
    if (!has_undefined_outputs &&
        !config.promote_inputs_to_common_dtype_ && !config.cast_common_dtype_to_outputs_ &&
        !config.enforce_safe_casting_to_output_) {
        // Invalidates common_dtype_ if it could not be inferred
        common_dtype_ = has_different_input_dtypes ? dtypes::Undefined : common_dtype_;
        return;
    }
    
    // Computes a common dtype, if needed
    if ((has_different_input_dtypes || all_ops_are_scalars_) && config.promote_inputs_to_common_dtype_) {
        common_dtype_ = compute_common_dtype(common_dtype_);
    }

    // Promotes common dtype to the default float scalar type, if needed
    if (config.promote_integer_inputs_to_float_ && common_dtype_.is_integral(/*include_bool=*/true)) {
        common_dtype_ = default_float_dtype();
    }
    
    // Reviews operands (2/2)
    //   - sets metadata for undefined outputs
    //   - checks that all arrays are on the same device, if requested
    //   - checks that the common dtype can safely cast to each output, if requested
    //   - creates temporaries for CPU operations, if needed and requested
    for (auto& op : operands_) {
        bool is_type_defined = op.is_type_defined();

        if (!is_type_defined) {
            op.target_dtype = common_dtype_;
        }

        // Skips undefined arrays
        if (!op.array().defined()) continue;

        // Checks safe casting, if requested
        if (config.enforce_safe_casting_to_output_ && op.is_output && op.current_dtype != common_dtype_) {
            if (!can_cast(common_dtype_, op.current_dtype)) {
                std::string msg = fmt::format(
                    "result type {} can't be cast to the {} desired output type ", 
                    common_dtype_.name(), op.current_dtype.name()
                );
                throw std::runtime_error(msg);
            }
        }

        // Creates temporaries for CPU operations, if needed and requested
        // TODO: if (common_device == CPU) {
        if (config.cast_common_dtype_to_outputs_ && op.is_output && op.current_dtype != common_dtype_) {
            op.exchange_array(detail::MaybeOwned<NDArray>::owned(
                empty(op.array().shape(), common_dtype_)
            ));
            op.current_dtype = common_dtype_;
            op.target_dtype = common_dtype_;
        }
        // }

        // Promotes inputs by creating temporaries of the correct dtype
        if (config.promote_inputs_to_common_dtype_ && !op.is_output && op.current_dtype != common_dtype_) {
            op.exchange_array(detail::MaybeOwned<NDArray>::owned(op.array().to(common_dtype_)));
            op.current_dtype = common_dtype_;
            op.target_dtype = common_dtype_;
        }
    }
}

Dtype ArrayIterator::compute_common_dtype(Dtype base_dtype) {
    Dtype common_dtype = base_dtype;
    for (auto& op : operands_) {
        if (op.is_output) continue;
        
        if (op.current_dtype != common_dtype) {
            common_dtype = promote_types(common_dtype, op.current_dtype);
        }
    }
    return common_dtype;
}

void ArrayIterator::permute_dimensions(const IntArray& perm) {
    auto reorder = [perm](const IntArray& data) {
        IntArray res(data.size(), 0);
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

void ArrayIterator::reorder_dimensions() {
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

IntArray ArrayIterator::compatible_stride(int64_t element_size) const {
    IntArray stride;
    int64_t next_stride = element_size;
    for (size_t dim = 0; dim < ndim(); dim++) {
        stride.push_back(next_stride);
        next_stride *= shape_[dim];
    }
    return stride;
}

IntArray ArrayIterator::invert_perm(const IntArray& input) const {
    // Invert the permutation caused by reorder_dimensions. This is not valid
    // after coalesce_dimensions is called.
    if (has_coalesced_dimensions_) {
        throw std::runtime_error("coalesce_dimensions() has already been called");
    }

    if (input.size() != perm_.size()) {
        throw std::runtime_error("Input size is different from perm_ size");
    }

    IntArray res(input.size());
    for (size_t dim = 0; dim < ndim(); dim++) {
        res[perm_[dim]] = input[dim];
    }
    return res;
}

void ArrayIterator::set_output(int64_t output_idx, const IntArray& shape, const IntArray& strides) {
    Operand& op = operands_[output_idx];
    if (op.array().defined()) return;
    
    if (strides.empty()) {
        op.set_array(detail::MaybeOwned<NDArray>::owned(empty(shape, op.target_dtype)));
    } else {
        op.set_array(detail::MaybeOwned<NDArray>::owned(empty_strided(shape, strides, op.target_dtype)));
    }

}

void ArrayIterator::allocate_outputs() {
    for (int out_idx = 0; out_idx < num_outputs_; out_idx++) {
        Operand& op = operands_[out_idx];
        if (op.array().defined()) continue;
        
        if (!op.is_type_defined()) {
            throw std::runtime_error("No type for operand " + out_idx);
        }

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
        auto array_shape = invert_perm(shape_);
        if (inverted) {
            // can just return contiguous output
            // it is faster because it avoids allocating 0 size tensor and
            // resizing and restriding it
            set_output(out_idx, array_shape, {});
        } else {
            auto array_stride = invert_perm(op.stride_bytes);
            for (size_t dim = 0; dim < ndim(); dim++) {
                array_stride[dim] /= static_cast<int64_t>(element_size);
            }
            set_output(out_idx, array_shape, array_stride);
        }
        
        op.current_dtype = op.target_dtype;
    }
}

void ArrayIterator::coalesce_dimensions() {
    if (ndim() <= 1) return;

    auto can_coalesce = [&](int dim0, int dim1) {
        int64_t size0 = shape_[dim0];
        int64_t size1 = shape_[dim1];
        if (size0 == 1 || size1 == 1) {
            return true;
        }
        for (int i = 0; i < num_operands(); i++) {
            IntArray& stride = operands_[i].stride_bytes;
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

void ArrayIterator::for_each(loop2d_t loop, int64_t grain_size) const {
    int num_threads = 1; // TODO: implement get_num_threads()
    int64_t numel = this->numel();

    if (numel == 0) {
        return;
    } else if (numel < grain_size || num_threads == 1) {
        return serial_for_each(loop, {0, numel});
    } else {
        // TODO: parallel for_each()
    }
}

void get_base_ptrs(char** ptrs, const std::vector<Operand>& operands) {
    for (int idx = 0; idx < operands.size(); idx++) {
        ptrs[idx] = static_cast<char*>(operands[idx].array().mutable_data_ptr());
    }
}

void get_strides(int64_t* strides, const std::vector<Operand>& operands, int ndim) {
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

namespace internal {

inline void get_data_ptrs(
    char** ptrs,
    char** base_ptrs,
    size_t noperands,
    const IntArray& strides,
    const IntArray& counter_values
) {
    size_t ndim = counter_values.size();
    std::copy(base_ptrs, base_ptrs + noperands, ptrs);
    
    for (size_t dim = 0; dim < ndim; dim++) {
        int64_t value = counter_values[dim];
        for (size_t idx = 0; idx < noperands; idx++) {
            ptrs[idx] += value * strides[dim * noperands + idx];
        }
    }
}

void serial_for_each(
    const IntArray& shape,
    const IntArray& strides,
    char** base_ptrs,
    size_t noperands,
    ArrayIterator::loop2d_t loop,
    Range range
) {
    size_t ndim = shape.size();

    if (ndim <= 1) {
        if (range.begin == 0) {
            loop(base_ptrs, strides.data(), range.size(), 1);
        } else {
            std::vector<char*> ptrs(noperands);
            get_data_ptrs(ptrs.data(), base_ptrs, noperands, strides, {range.begin});
            loop(ptrs.data(), strides.data(), range.size(), 1);
        }
    } else {
        std::vector<char*> ptrs(noperands);
        DimCounter counter(shape, range);
        while(!counter.is_done()) {
            get_data_ptrs(ptrs.data(), base_ptrs, noperands, strides, counter.values);
            auto step = counter.max_2d_step();
            loop(ptrs.data(), strides.data(), step[0], step[1]);
            counter.increment(step);
        }
    }
}

} // namespace internal

void ArrayIterator::serial_for_each(loop2d_t loop, Range range) const {
    if (range.size() == 0) return;

    const size_t noperands = num_operands();
    const int ndim = this->ndim();

    std::vector<char*> ptrs(noperands);
    std::vector<int64_t> strides(noperands * static_cast<size_t>(std::max(ndim, 2)));

    get_base_ptrs(ptrs.data(), operands_);
    get_strides(strides.data(), operands_, ndim);

    internal::serial_for_each(shape_, strides, ptrs.data(), noperands, loop, range);
}


void ArrayIterator::print_state(std::ostream& os) const {
    os << "=== ArrayIterator state ===\n";
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
        const auto& shp = operand.array().shape();
        for (size_t i = 0; i < shp.size(); ++i) {
            os << shp[i];
            if (i + 1 < shp.size()) os << ", ";
        }
        os << "]\n";

        os << "  original strides: [";
        const auto& strides = operand.array().strides();
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