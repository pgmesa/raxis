#pragma once

#include <ostream>
#include <iostream>
#include <type_traits>
#include <optional>
#include <cassert>

#include <raxis/core/dtype.h>
#include <raxis/core/types.h>

#include <raxis/util/range.h>
#include <raxis/util/function_ref.h>
#include <raxis/util/maybe_owned.h>

#include <raxis/tensor/tensor.h>


namespace raxis {

namespace internal {
// From: aten/src/ATen/TensorIterator.h
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants parallelism. For example, when summing an tensor, it is
// deemed inefficient to parallelise over tensors shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr int64_t GRAIN_SIZE = 32768;
}

struct Operand {
    IntArray stride_bytes;
    bool is_output = false;
    bool is_read_write = false;
    Dtype target_dtype;
    Dtype current_dtype;
    // std::optional<Device>

    explicit Operand(detail::MaybeOwned<Tensor>&& arr) {
        if (arr->defined()) {
            target_dtype = arr->dtype();
            current_dtype = target_dtype;
        }
        tensor_ = std::move(arr);
    }

    const Tensor& tensor() const {
        return *tensor_;
    }

    const Tensor& original_tensor() const {
        return *(original_tensor_.value());
    }

    void set_tensor(detail::MaybeOwned<Tensor>&& new_tensor);
    void exchange_tensor(detail::MaybeOwned<Tensor>&& new_tensor);
    void restore_original_tensor();
    bool has_exchanged() const;
    bool is_type_defined() const;
    
private:
    detail::MaybeOwned<Tensor> tensor_;
    std::optional<detail::MaybeOwned<Tensor>> original_tensor_; // Original tensor (if swapped)
};


class TensorIterator;
class TensorIteratorConfig;


class TensorIteratorConfig {
public:
    friend struct TensorIterator;
    
    TensorIteratorConfig& add_output(const Tensor& output) {
        if (num_inputs_ != 0) {
            throw std::runtime_error("Cannot add output after inputs have been added");
        }
        return add_borrowed_output(output);
    }
    TensorIteratorConfig& add_input(const Tensor& input) {
        if (num_outputs_ == 0) {
            throw std::runtime_error("At least one output must be added before inputs");
        }
        return add_borrowed_input(input);
    }
    // Don't allow borrowing from temporaries
    TensorIteratorConfig& add_output(Tensor&& output) = delete;
    TensorIteratorConfig& add_input(Tensor&& input) = delete;

    TensorIteratorConfig& add_owned_output(const Tensor& output);
    TensorIteratorConfig& add_owned_input(const Tensor& input);
    TensorIteratorConfig& add_owned_output(Tensor&& output);
    TensorIteratorConfig& add_owned_input(Tensor&& input);

    TensorIteratorConfig& add_borrowed_output(const Tensor& output);
    TensorIteratorConfig& add_borrowed_input(const Tensor& input);
    // Don't allow borrowing from temporaries
    TensorIteratorConfig& add_borrowed_output(Tensor&& output) = delete;
    TensorIteratorConfig& add_borrowed_input(Tensor&& input) = delete;

    TensorIteratorConfig& is_reduction(bool is_reduction) {
        is_reduction_ = is_reduction;
        return *this;
    }

    // Sets the check_all_same_dtype_ flag, which is true by default
    // If true, checks that all inputs and defined outputs have the same dtype
    // Setting either of promote_inputs_to_common_dtype_
    //   or cast_common_dtype_to_outputs_ to true will set
    //   check_all_same_dtype_ to false.
    TensorIteratorConfig& check_all_same_dtype(const bool _check) {
        check_all_same_dtype_ = _check;
        return *this;
    }
    
    TensorIteratorConfig& enforce_safe_casting_to_output(const bool _enforce) {
        enforce_safe_casting_to_output_ = _enforce;
        return *this;
    }

    TensorIteratorConfig& enforce_linear_iteration(const bool _enforce = true) {
        enforce_linear_iteration_ = _enforce;
        return *this;
    }

    TensorIteratorConfig& promote_inputs_to_common_dtype(const bool _promote) {
        promote_inputs_to_common_dtype_ = _promote;
        if (_promote) {
            check_all_same_dtype_ = false;
        }
        return *this;
    }

    TensorIteratorConfig& promote_integer_inputs_to_float(const bool _promote) {
        promote_integer_inputs_to_float_ = _promote;
        if (promote_integer_inputs_to_float_ && !promote_inputs_to_common_dtype_) {
            throw std::runtime_error("'promote_inputs_to_common_dtype_' must also be true");
        }
        return *this;
    }

    TensorIteratorConfig& cast_common_dtype_to_outputs(const bool _cast) {
        cast_common_dtype_to_outputs_ = _cast;
        if (_cast) {
            check_all_same_dtype_ = false;
        }
        return *this;
    }

    TensorIteratorConfig& declare_static_dtype(Dtype dtype) {
        static_dtype_.emplace(dtype);
        return *this;
    }

    TensorIteratorConfig& declare_static_shape(const IntArray& shape) {
        static_shape_ = shape;
        return *this;
    }

    TensorIterator build();

private:
    std::vector<detail::MaybeOwned<Tensor>> tensors_;
    int num_outputs_ = 0;
    int num_inputs_ = 0;
    std::optional<Dtype> static_dtype_ = std::nullopt;
    std::optional<IntArray> static_shape_ = std::nullopt;
    bool is_reduction_ = false;
    bool check_all_same_dtype_ = true;
    bool enforce_safe_casting_to_output_ = false;
    bool enforce_linear_iteration_ = false;
    bool promote_inputs_to_common_dtype_ = false;
    bool promote_integer_inputs_to_float_ = false;
    bool cast_common_dtype_to_outputs_ = false;
};

namespace iterator {

inline TensorIteratorConfig binary_op_config() {
    return TensorIteratorConfig()
        // .set_check_mem_overlap(true)
        // .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .enforce_safe_casting_to_output(true);
}

inline TensorIteratorConfig binary_float_op_config() {
    return binary_op_config()
        .promote_integer_inputs_to_float(true);
}

} // namespace iterator

struct TensorIterator {
public:
    int ndim() const {
        return static_cast<int>(shape_.size());
    }

    int64_t numel() const {
        return compute_numel(shape_);
    }

    int num_inputs() const {
        return num_operands() - num_outputs_;
    }

    int num_outputs() const {
        return num_outputs_;
    }

    int num_operands() const {
        return static_cast<int>(operands_.size());
    }

    Dtype common_dtype() const {
        return common_dtype_;
    }

    const Operand& get_operand(int64_t idx) const {
        return operands_[idx]; 
    }

    const Tensor& get_tensor(int64_t idx) const {
        return get_operand(idx).tensor();
    }

    const Tensor& get_output(int64_t idx) const {
        assert(idx < num_outputs_);
        return get_tensor(idx);
    }

    Dtype dtype(int64_t idx) const{
        return operands_[idx].current_dtype;
    }

    void build(TensorIteratorConfig& config);
    void print_state(std::ostream& os = std::cout) const;

    // Type-erased 2D loop
    using loop2d_t = detail::function_ref<void(char** data, const int64_t* strides, int64_t size0, int64_t size1)>;
    
    // Templated 1D loop (inlined, optimizable)
    template <typename loop1d_t,
        std::enable_if_t<
            std::is_invocable_r_v<
                void, loop1d_t, char**, const int64_t*, int64_t>,
            int> = 0>
    void for_each(loop1d_t loop, int64_t grain_size = internal::GRAIN_SIZE) const {
        for_each(loop_2d_from_1d(loop), grain_size);
    }

    template <typename loop1d_t,
        std::enable_if_t<
            std::is_invocable_r_v<
                void, loop1d_t, char**, const int64_t*, int64_t>,
            int> = 0>
    void serial_for_each(loop1d_t loop, Range range) const {
        serial_for_each(loop_2d_from_1d(loop), range);
    }

    void for_each(loop2d_t loop, int64_t grain_size = internal::GRAIN_SIZE) const;
    void serial_for_each(loop2d_t loop, Range range) const;

    /// Create a strides array for a Tensor with shape of this iterator. The
    /// parameter `element_size` specifies the size of Tensor's data type in
    /// bytes (e.g. `4` for `float`)
    IntArray compatible_stride(int64_t element_size) const;

    /// Inverts the re-ordering done by reorder_dimensions. This can only be
    /// called *before* coalesce_dimensions() is called.
    IntArray invert_perm(const IntArray& input) const;

    void set_output(int64_t output_idx, const IntArray& sizes, const IntArray& strides);

    // Copies from temporary outputs back to the original outputs
    // NOTE: only used on CPU
    void cast_outputs();

private:
    std::vector<Operand> operands_;
    int num_outputs_ = 0;
    
    IntArray shape_;
    IntArray perm_;

    bool all_ops_same_shape_ = false;
    bool all_ops_are_scalars_ = false;

    Dtype common_dtype_;
    // common_device_
    
    /// From iterator config
    bool is_reduction_ = false;
    /// Whether iteration must be fixed. This disables dimension permuting and
    /// also changes how for_each divides work among threads.
    bool enforce_linear_iteration_ = false;

    /// Has coalesce_dimensions() been called? 
    /// This is SOLELY used to check validity of perm_.
    bool has_coalesced_dimensions_ = false;

    void populate_operands(TensorIteratorConfig&);
    void compute_shape(const TensorIteratorConfig&);
    void check_output_shapes();
    void compute_types(const TensorIteratorConfig&);
    Dtype compute_common_dtype(Dtype base_dtype);
    void compute_strides(const TensorIteratorConfig&);
    void reorder_dimensions();
    void permute_dimensions(const IntArray& perm);
    void allocate_outputs();
    void coalesce_dimensions();

    template <typename loop1d_t>
    auto loop_2d_from_1d(const loop1d_t& loop) const {
        // Convert 1D kernel to 2D
        return
            [loop, noperands = num_operands()](
                char** base, const int64_t* strides, int64_t size0, int64_t size1
            ) {
                // Adapt 1D signature to 2D
                const int64_t* outer_strides = &strides[noperands];
                std::vector<char*> data(base, base + noperands);
                for (int outer = 0; outer < size1; outer++) {
                    if (outer > 0) {
                        for (int idx = 0; idx < noperands; idx++) {
                            data[idx] += outer_strides[idx];
                        }
                    } 
                    loop(data.data(), strides, size0);
                }
            };
    }
};

}