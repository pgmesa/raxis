
#include <raxis/core/array_iterator.hpp>

#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <cassert>

#include <raxis/array.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/core/utils/range.hpp>
#include <raxis/core/utils/dim_counter.hpp>
#include <raxis/core/utils/function_ref.hpp>
#include <raxis/core/utils/maybe_owned.hpp>


namespace raxis {

void Operand::exchange_array(detail::MaybeOwned<NDArray>&& new_array) {
    if (original_array_) {
        throw std::runtime_error("Array already exchanged");
    }
    original_array_ = std::move(array_);
    array_ = std::move(new_array);
    current_dtype = array_->dtype();
}

void Operand::restore_original_array() {
    if (!original_array_) {
        throw std::runtime_error("No original array to restore");
    }
    array_ = std::move(*original_array_);
    original_array_.reset();
    current_dtype = array_->dtype();
}

bool Operand::has_exchanged() const {
    return original_array_.has_value();
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

ArrayIteratorConfig& ArrayIteratorConfig::is_reduction(bool is_reduction) {
    is_reduction_ = is_reduction;
    return *this;
}

ArrayIterator ArrayIteratorConfig::build() {
    ArrayIterator iter;
    iter.build(*this);
    return iter;
}

// ========================== Array Iterator ==========================
int ArrayIterator::ndim() const {
    return static_cast<int>(shape_.size());
}

int64_t ArrayIterator::numel() const {
    return compute_numel(shape_);
}

int ArrayIterator::num_inputs() const {
    return num_operands() - num_outputs_;
}

int ArrayIterator::num_outputs() const {
    return num_outputs_;
}

int ArrayIterator::num_operands() const {
    return static_cast<int>(operands_.size());
}

Dtype ArrayIterator::common_dtype() const {
    return common_dtype_;
}

void ArrayIterator::build(ArrayIteratorConfig& config) {
    // Fill in operands_ based on configuration
    populate_operands(config);
    // Compute the broadcasted shape
    compute_shape();
    // Compute the result dtype and device
    compute_types();
    // Compute each array's stride after broadcasting
    compute_strides();

    // Re-order dimensions to improve coalescing
    reorder_dimensions();
    // Allocate the output array if it's not provided
    allocate_or_resize_outputs();
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

void ArrayIterator::compute_shape() {
    all_ops_same_shape_ = true;
    bool has_scalars = false;
    bool has_arrays = false;
    for (Operand& op: operands_) {
        if (!op.array().defined()) continue;
        
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
}

void ArrayIterator::compute_types() {
    bool has_different_input_dtypes = false;
    bool has_undefined_outputs = false;
    Dtype first_input_dtype = dtypes::Undefined;
    
    for (const auto& op : operands_) {
        // Validate inputs are defined
        if (!op.is_output) {
            if (!op.array().defined()) {
                throw std::runtime_error("Found undefined input array");
            }
            
            if (first_input_dtype == dtypes::Undefined) {
                first_input_dtype = op.current_dtype;
            } else if (op.current_dtype != first_input_dtype) {
                has_different_input_dtypes = true;
            }
        }
        
        // Check for undefined outputs
        if (op.is_output && !op.array().defined()) {
            has_undefined_outputs = true;
        }
    }
    
    // 1. Compute common dtype from inputs
    Dtype common_dtype = dtypes::Undefined;
    
    for (auto& op : operands_) {
        if (op.is_output) continue;
        
        if (common_dtype == dtypes::Undefined) {
            common_dtype = op.current_dtype;
        } else if (op.current_dtype != common_dtype) {
            common_dtype = promote_types(common_dtype, op.current_dtype);
        }
    }
    
    common_dtype_ = common_dtype;
    
    // 2. Promote inputs (creates temporary copies)
    for (auto& op : operands_) {
        if (!op.is_output && op.current_dtype != common_dtype_) {
            op.exchange_array(
                detail::MaybeOwned<NDArray>::owned(op.array().to(common_dtype_))
            );
        }
    }
    
    // 3. Set output dtype if undefined
    for (auto& op : operands_) {
        if (op.is_output && !op.array().defined()) {
            op.target_dtype = common_dtype_;
        }
    }
}

void ArrayIterator::compute_strides() {
    for (Operand& op: operands_) {
        if (!op.array().defined()) continue;

        const IntArray& original_shape = op.array().shape();
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

void ArrayIterator::reorder_dimensions() {
    std::reverse(shape_.begin(), shape_.end());

    for (Operand& op: operands_) {
        if (op.stride_bytes.empty()) continue;
        std::reverse(op.stride_bytes.begin(), op.stride_bytes.end());
    }
}

void ArrayIterator::allocate_or_resize_outputs() {}

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
    os << "all_ops_same_shape: " << (all_ops_same_shape_? "true": "false") << "\n";
    os << "all_ops_are_scalars: " << (all_ops_are_scalars_? "true": "false") << "\n";

    os << "computational shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        os << shape_[i];
        if (i + 1 < shape_.size()) os << ", ";
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