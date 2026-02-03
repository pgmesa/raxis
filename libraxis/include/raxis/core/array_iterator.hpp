#pragma once

#include <ostream>
#include <iostream>
#include <type_traits>
#include <optional>

#include <raxis/array.hpp>
#include <raxis/dtype.hpp>
#include <raxis/array_types.hpp>
#include <raxis/core/utils/range.hpp>
#include <raxis/core/utils/function_ref.hpp>
#include <raxis/core/utils/maybe_owned.hpp>


namespace raxis {

namespace internal {
// From: aten/src/ATen/TensorIterator.h
// This parameter is heuristically chosen to determine the minimum number of
// work that warrants parallelism. For example, when summing an array, it is
// deemed inefficient to parallelise over arrays shorter than 32768. Further,
// no parallel algorithm (such as parallel_reduce) should split work into
// smaller than GRAIN_SIZE chunks.
constexpr int64_t GRAIN_SIZE = 32768;
}

struct Operand {
    IntArray stride_bytes;
    bool is_output = false;
    bool is_read_write = false;
    bool will_resize = false;
    Dtype target_dtype;
    Dtype current_dtype;
    // std::optional<Device>

    explicit Operand(detail::MaybeOwned<NDArray>&& arr) {
        if (arr->defined()) {
            target_dtype = arr->dtype();
            current_dtype = target_dtype;
        }
        array_ = std::move(arr);
    }

    const NDArray& array() const {
        return *array_;
    }

    void exchange_array(detail::MaybeOwned<NDArray>&& new_array);
    void restore_original_array();
    bool has_exchanged() const;
    
private:
    detail::MaybeOwned<NDArray> array_;
    std::optional<detail::MaybeOwned<NDArray>> original_array_; // Original array (if swapped)
};


class ArrayIterator;
class ArrayIteratorConfig;


class ArrayIteratorConfig {
public:
    friend struct ArrayIterator;
    
    ArrayIteratorConfig& add_output(const NDArray& output) {
        return add_borrowed_output(output);
    }
    ArrayIteratorConfig& add_input(const NDArray& input) {
        return add_borrowed_input(input);
    }
    // Don't allow borrowing from temporaries
    ArrayIteratorConfig& add_output(NDArray&& output) = delete;
    ArrayIteratorConfig& add_input(NDArray&& input) = delete;

    ArrayIteratorConfig& add_owned_output(const NDArray& output);
    ArrayIteratorConfig& add_owned_input(const NDArray& input);
    ArrayIteratorConfig& add_owned_output(NDArray&& output);
    ArrayIteratorConfig& add_owned_input(NDArray&& input);

    ArrayIteratorConfig& add_borrowed_output(const NDArray& output);
    ArrayIteratorConfig& add_borrowed_input(const NDArray& input);
    // Don't allow borrowing from temporaries
    ArrayIteratorConfig& add_borrowed_output(NDArray&& output) = delete;
    ArrayIteratorConfig& add_borrowed_input(NDArray&& input) = delete;

    ArrayIteratorConfig& is_reduction(bool is_reduction);

    ArrayIterator build();

private:
    std::vector<detail::MaybeOwned<NDArray>> arrays_;
    int num_outputs_ = 0;
    int num_inputs_ = 0;
    bool is_reduction_ = false;
};


struct ArrayIterator {
public:
    int ndim() const;
    int64_t numel() const;
    IntArray& shape() const;
    int num_inputs() const;
    int num_outputs() const;
    int num_operands() const;
    Dtype common_dtype() const;

    void build(ArrayIteratorConfig& config);
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

private:
    std::vector<Operand> operands_;
    int num_outputs_ = 0;
    
    IntArray shape_;
    // IntArray perm_;

    bool all_ops_same_shape_ = false;
    bool all_ops_are_scalars_ = false;

    Dtype common_dtype_;
    // common_device_

    bool is_reduction_ = false;

    void populate_operands(ArrayIteratorConfig&);
    void compute_shape();
    void compute_types();
    void compute_strides();
    void reorder_dimensions();
    void allocate_or_resize_outputs();
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