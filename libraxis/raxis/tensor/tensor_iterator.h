#pragma once

#include <ostream>
#include <iostream>
#include <type_traits>
#include <optional>
#include <cassert>
#include <bitset>

#include <raxis/core/dtype.h>
#include <raxis/core/device.h>
#include <raxis/core/types.h>

#include <raxis/util/range.h>
#include <raxis/util/exception.h>
#include <raxis/util/function_ref.h>
#include <raxis/util/maybe_owned.h>
#include <raxis/util/small_vector.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/shape_utils.h>


namespace raxis {

namespace detail {

// From: aten/src/ATen/TensorIterator.h
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants parallelism. For example, when summing a tensor, it is
// deemed inefficient to parallelise over tensors shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr int64_t GRAIN_SIZE = 32768;

} // namespace detail

struct Operand {
    using StrideVector = SmallVector<int64_t, 6>;

    /// The data pointer. This may be different from tensor->data_ptr() if the
    /// iterator is split.
    void* data = nullptr;
    StrideVector stride_bytes;
    bool is_output = false;
    bool is_read_write = false;

    /// The desired device and type for the operand. For inputs, this specifies
    /// that the input should be converted to this type if necessary. For outputs,
    /// this specifies which type to allocate. target_dtype and device are
    /// initialized with the dtype and device of the tensor but during type
    /// promotion target_dtype value can become different from tensor's dtype
    /// also, during type promotion target_dtype and device can be set for an
    /// undefined tensor so that tensor can be properly constructed later.
    Dtype target_dtype;
    std::optional<Device> device;
    Dtype current_dtype;

    explicit Operand(MaybeOwned<Tensor>&& t) {
        if (t->defined()) {
            device = t->device();
            target_dtype = t->dtype();
            current_dtype = target_dtype;
        }
        tensor_ = std::move(t);
    }

    const Tensor& tensor() const {
        return *tensor_;
    }

    const Tensor& original_tensor() const {
        return *(original_tensor_.value());
    }

    void set_tensor(MaybeOwned<Tensor>&& new_tensor);
    void exchange_tensor(MaybeOwned<Tensor>&& new_tensor);
    void restore_original_tensor();
    bool has_exchanged() const;

    bool is_device_defined() const {
        return device.has_value();
    }

    bool is_type_defined() const {
        return target_dtype != dtypes::Undefined;
    }
    
private:
    MaybeOwned<Tensor> tensor_;
    std::optional<MaybeOwned<Tensor>> original_tensor_; // Original tensor (if swapped)
};


class TensorIterator;
class TensorIteratorConfig;


class TensorIteratorConfig {
public:
    friend struct TensorIterator;
    
    TensorIteratorConfig& add_output(const Tensor& output) {
        return add_borrowed_output(output);
    }
    TensorIteratorConfig& add_input(const Tensor& input) {
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

    // Sets the check_all_same_device_ flag, which is true by default
    // If true, all operands must be on the same device, with the possible
    //   exception of CPU scalars, which can be passed to some CUDA kernels
    //   as kernel arguments.
    TensorIteratorConfig& check_all_same_device(const bool _check) {
        check_all_same_device_ = _check;
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
        RAXIS_INTERNAL_ASSERT(
            !promote_integer_inputs_to_float_ || promote_inputs_to_common_dtype_);
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

    TensorIteratorConfig& declare_static_shape(IntArrayRef shape) {
        static_shape_ = DimVector(shape);
        return *this;
    }

    TensorIteratorConfig& declare_static_device(Device device) {
        static_device_ = device;
        return *this;
    }

    TensorIteratorConfig& allow_cpu_scalars(const bool _allow_cpu_scalars) {
        allow_cpu_scalars_ = _allow_cpu_scalars;
        return *this;
    }

    TensorIterator build();

private:
    SmallVector<MaybeOwned<Tensor>, 4> tensors_;
    int num_outputs_ = 0;
    int num_inputs_ = 0;
    std::optional<DimVector> static_shape_ = std::nullopt;
    std::optional<Dtype> static_dtype_ = std::nullopt;
    std::optional<Device> static_device_ = std::nullopt;
    bool allow_cpu_scalars_ = false;
    bool is_reduction_ = false;
    bool check_all_same_dtype_ = true;
    bool check_all_same_device_ = true;
    bool enforce_safe_casting_to_output_ = false;
    bool enforce_linear_iteration_ = false;
    bool promote_inputs_to_common_dtype_ = false;
    bool promote_integer_inputs_to_float_ = false;
    bool cast_common_dtype_to_outputs_ = false;
};

namespace iterator {

inline TensorIteratorConfig binary_op_config() {
    return TensorIteratorConfig()
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .enforce_safe_casting_to_output(true);
}

inline TensorIteratorConfig binary_float_op_config() {
    return binary_op_config()
        .promote_integer_inputs_to_float(true);
}

inline TensorIteratorConfig comparison_op_config() {
    return TensorIteratorConfig()
        .allow_cpu_scalars(true)
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(false)
        .enforce_safe_casting_to_output(false)
        .declare_static_dtype(dtypes::Bool);
}

inline TensorIteratorConfig unary_op_config() {
    return TensorIteratorConfig()
        .cast_common_dtype_to_outputs(false)
        .enforce_safe_casting_to_output(false)
        .check_all_same_dtype(true);
}
 
inline TensorIteratorConfig unary_float_op_config() {
    return TensorIteratorConfig()
        .promote_inputs_to_common_dtype(true)
        .cast_common_dtype_to_outputs(true)
        .enforce_safe_casting_to_output(true)
        .promote_integer_inputs_to_float(true);
}

} // namespace iterator

struct TensorIterator {
public:
    using DimMask = std::bitset<64>;
    using PtrVector = SmallVector<char*, 4>;
    using StrideVector = SmallVector<int64_t, 6>;
    // Type-erased 2D loop
    using loop2d_t = function_ref<
        void(char** data, const int64_t* strides, int64_t size0, int64_t size1)>;

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

    Dtype dtype(int64_t idx = 0) const{
        return operands_[idx].current_dtype;
    }

    Device device(int64_t idx = 0) const {
        return operands_[idx].device.value();
    }
    
    DeviceType device_type(int64_t idx = 0) const {
        return device(idx).type();
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

    const Tensor& get_input(int64_t idx = 0) const {
        assert(idx >= 0 && idx < num_inputs());
        return get_tensor(num_outputs_ + idx);
    }

    IntArrayRef shape() const {
        return shape_;
    }

    int64_t element_size(int64_t arg) const {
        return static_cast<int64_t>(dtype(arg).itemsize());
    }

    bool is_dim_reduced(int dim) const;

    PtrVector get_base_ptrs() const;
    StrideVector get_strides() const;

    void narrow(int dim, int64_t start, int64_t size);

    void build(TensorIteratorConfig& config);
    void print_state(std::ostream& os = std::cout) const;

    static TensorIterator reduce_op(Tensor& out, const Tensor& a);
    
    // Templated 1D loop (inlined, optimizable)
    template <typename loop1d_t,
        std::enable_if_t<
            std::is_invocable_r_v<
                void, loop1d_t, char**, const int64_t*, int64_t>,
            int> = 0>
    void for_each(loop1d_t loop, int64_t grain_size = detail::GRAIN_SIZE) const {
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

    void for_each(loop2d_t loop, int64_t grain_size = detail::GRAIN_SIZE) const;
    void serial_for_each(loop2d_t loop, Range range) const;

    void parallel_reduce(loop2d_t loop) const;

    /// Create a strides array for a Tensor with shape of this iterator. The
    /// parameter `element_size` specifies the size of Tensor's data type in
    /// bytes (e.g. `4` for `float`)
    StrideVector compatible_stride(int64_t element_size) const;

    /// Inverts the re-ordering done by reorder_dimensions. This can only be
    /// called *before* coalesce_dimensions() is called.
    DimVector invert_perm(IntArrayRef input) const;

    void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides);

    // Copies from temporary outputs back to the original outputs
    // NOTE: only used on CPU
    void cast_outputs();

private:
    SmallVector<Operand, 4> operands_;
    int num_outputs_ = 0;
    
    DimVector shape_;
    DimVector perm_;
    /// The index offsets into the original tensors for each dimension.
    /// This is only non-zero when you narrow() a TensorIterator (e.g.,
    /// when you make sub-TensorIterators).
    DimVector view_offsets_;

    bool all_ops_same_shape_ = false;
    bool all_ops_are_scalars_ = false;

    Dtype common_dtype_ = dtypes::Undefined;
    Device common_device_ = DeviceType::CPU;
    
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
    Dtype compute_common_dtype();
    void compute_strides(const TensorIteratorConfig&);
    bool fast_setup();
    void reorder_dimensions();
    void permute_dimensions(IntArrayRef perm);
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
                PtrVector data(base, base + noperands);
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

namespace detail {

void get_base_ptrs(char** ptrs, ArrayRef<Operand> operands);
void get_strides(int64_t* strides, ArrayRef<Operand> operands, int ndim);

void get_data_ptrs(
    char** ptrs,
    ArrayRef<char*> base_ptrs,
    IntArrayRef strides,
    IntArrayRef counter_values
);

void serial_for_each(
    IntArrayRef shape,
    IntArrayRef strides,
    char** base_ptrs,
    size_t noperands,
    TensorIterator::loop2d_t loop,
    Range range
);

} // namespace detail

} // namespace raxis