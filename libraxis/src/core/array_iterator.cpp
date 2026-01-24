
#include <raxis/core/array_iterator.hpp>

#include <ostream>

#include <raxis/array.hpp>
#include <raxis/core/shape.hpp>


namespace raxis {

    ArrayIteratorConfig& ArrayIteratorConfig::add_input(const NDArray& input) {
        arrays_.push_back(&input);
        num_inputs_++;
        return *this;
    }

    ArrayIteratorConfig& ArrayIteratorConfig::add_output(const NDArray& output) {
        arrays_.push_back(&output);
        num_outputs_++;
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
    size_t ArrayIterator::ndim() const {
        return shape_.size();
    }

    size_t ArrayIterator::numel() const {
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

    void ArrayIterator::build(const ArrayIteratorConfig& config) {
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

    void ArrayIterator::populate_operands(const ArrayIteratorConfig& config) {
        num_outputs_ = config.num_outputs_;
        size_t num_operands = config.arrays_.size();
        
        for (int idx = 0; idx < num_operands; idx++) {
            const NDArray* arr = config.arrays_[idx];
            operands_.emplace_back(arr);
            
            // check if tensor is an output
            if (idx >= num_outputs_) continue;
            
            // mark tensor as output
            operands_[idx].is_output = true;
            // const auto& output = arr(idx);
            // if (!output.defined()) continue;

            // for (int inp_idx = num_outputs_; inp_idx < num_operands; inp_idx++)) {
            //     const auto& input = arr(inp_idx);
            //     if (output.is_same(input)) {
            //         operands_[idx].is_read_write = true;
            //         break;
            //     }
            // }
        }
    }

    void ArrayIterator::compute_shape() {
        all_ops_same_shape_ = true;
        bool has_scalars = false;
        bool has_arrays = false;
        for (Operand& op: operands_) {
            //if (!op.array().defined()) continue;
            
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
        // Reviews operands (1/2)
        //   - validates that all input tensors are defined
        //   - computes common device
        //   - determines if there are undefined outputs
        //   - determines if there are different dtypes and attempts
        //       to quickly acquire a common dtype
        


        // Reviews operands (2/2)
        //   - sets metadata for undefined outputs
        //   - checks that all tensors are on the same device, if requested
        //   - checks that the common dtype can safely cast to each output, if requested
        //   - creates temporaries for CPU operations, if needed and requested
    }

    void ArrayIterator::compute_strides() {
        for (Operand& op: operands_) {
            // if (!op.array().defined()) continue;

            const IntArray& original_shape = op.array().shape();
            const IntArray& original_stride = op.array().strides();
            size_t itemsize = op.array().dtype().itemsize();
            
            size_t original_ndim = original_shape.size();
            size_t target_ndim = ndim();
            size_t offset = target_ndim - original_ndim;
            
            if (offset > 0)
                op.stride_bytes.resize(target_ndim, 0);
            else
                op.stride_bytes.resize(target_ndim);

            for (size_t i = 0; i < original_ndim; ++i) {
                size_t ti = offset + i;
                if (original_shape[i] == 1 && shape_[ti] != 1) {
                    op.stride_bytes[ti] = 0;
                } else {
                    op.stride_bytes[ti] = original_stride[i] * itemsize;
                }
            }
        }
    }

    void ArrayIterator::reorder_dimensions() {}

    void ArrayIterator::allocate_or_resize_outputs() {}

    void ArrayIterator::coalesce_dimensions() {
        if (ndim() <= 1) return;

        auto can_coalesce = [&](int dim0, int dim1) {
            Int size0 = shape_[dim0];
            Int size1 = shape_[dim1];
            if (size0 == 1 || size1 == 1) {
                return true;
            }
            for (size_t i = 0; i < num_operands(); i++) {
                IntArray& stride = operands_[i].stride_bytes;
                if (size0 * stride[dim0] != stride[dim1]) {
                    return false;
                }
            }
            return true;
        };

        auto replace_stride = [&](int dim0, int dim1) {
            for (size_t i = 0; i < num_operands(); i++) {
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

    void ArrayIterator::print_state(std::ostream& os) const {
        os << "=== ArrayIterator state ===\n";
        os << "ndim: " << ndim() << "\n";
        os << "numel: " << numel() << "\n";
        os << "computed shape: [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            os << shape_[i];
            if (i + 1 < shape_.size()) os << ", ";
        }
        os << "]\n";
        os << "num_operands: " << num_operands() << "\n";
        os << "num_outputs: " << num_outputs_ << "\n";
        os << "all_ops_same_shape: " << (all_ops_same_shape_? "true": "false") << "\n";
        os << "all_ops_are_scalars: " << (all_ops_are_scalars_? "true": "false") << "\n";
        os << "has_coalesced_dimensions: " << (has_coalesced_dimensions_? "true": "false") << "\n";

        os << "shape: [";
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

            os << "  original shape: [";
            const auto& shp = operand.array().shape();
            for (size_t i = 0; i < shp.size(); ++i) {
                os << shp[i];
                if (i + 1 < shp.size()) os << ", ";
            }
            os << "]\n";

            os << "  stride_bytes: [";
            for (size_t i = 0; i < operand.stride_bytes.size(); ++i) {
                os << operand.stride_bytes[i];
                if (i + 1 < operand.stride_bytes.size()) os << ", ";
            }
            os << "]\n";
        }

        os << "===========================\n";
    }

}