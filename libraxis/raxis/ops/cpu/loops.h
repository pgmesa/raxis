#pragma once

#include <stdexcept>
#include <type_traits>

#include <raxis/macros.h>
#include <raxis/util/exception.h>
#include <raxis/util/function_traits.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace {

template <typename T>
inline T load(const void* ptr) {
    return *reinterpret_cast<const T*>(ptr);
}

// Dereference implementation
template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_impl(
    char* RAXIS_RESTRICT data[],
    const int64_t* strides,
    int64_t i,
    std::index_sequence<INDEX...>
) {
    return std::make_tuple(
        load<typename traits::template arg<INDEX>::type>(
            data[INDEX] + i * strides[INDEX]
        )...
    );
}

// Dereference: loads all function arguments from memory
template <typename traits>
typename traits::ArgsTuple
dereference(
    char* RAXIS_RESTRICT data[],
    const int64_t* strides,
    int64_t i
) {
    using Indices = std::make_index_sequence<traits::arity>;
    return dereference_impl<traits>(data, strides, i, Indices{});
}

// Execute operation: version with return value
template <typename func_t,
    std::enable_if_t<!std::is_void_v<typename function_traits<func_t>::result_type>>* = nullptr>
inline void execute_op(
    char* RAXIS_RESTRICT data[],
    const int64_t* strides,
    int64_t i,
    int64_t n,
    func_t&& op
) {
    using traits = function_traits<func_t>;
    using result_type = typename traits::result_type;
    
    for (; i < n; i++) {
        result_type* out_ptr = reinterpret_cast<result_type*>(data[0] + i * strides[0]);
        *out_ptr = std::apply(op, dereference<traits>(&data[1], &strides[1], i));
    }
}

// Execute operation: version for void
template <typename func_t,
    std::enable_if_t<std::is_void_v<typename function_traits<func_t>::result_type>>* = nullptr>
inline void execute_op(
    char* RAXIS_RESTRICT data[],
    const int64_t* strides,
    int64_t i,
    int64_t n,
    func_t&& op
) {
    using traits = function_traits<func_t>;
    
    for (; i < n; i++) {
        std::apply(op, dereference<traits>(&data[0], &strides[0], i));
    }
}

} // namespace

// Basic loop operation (one output, N inputs). May be auto-vectorized
// by the compiler. Supports inputs and outputs of different types.
template <typename func_t>
inline void
basic_loop(char* RAXIS_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
    using traits = function_traits<func_t>;
    constexpr int ntensors = traits::arity + 1;

    // Copying strides to temporary array helps auto vectorization in older GCC versions.
    int64_t strides[ntensors];
    for (int i = 0; i < ntensors; i++) {
        strides[i] = strides_[i];
    }

    execute_op(data, strides, i, n, std::forward<func_t>(op));
}

template <typename func_t>
void cpu_kernel(TensorIterator& iter, func_t&& op, int64_t grain_size = detail::GRAIN_SIZE) {
    using traits = function_traits<func_t>;
    
    RAXIS_INTERNAL_ASSERT(iter.num_inputs() == traits::arity);
    RAXIS_INTERNAL_ASSERT(iter.num_outputs() == 1);
    
    iter.for_each([&](char** data, const int64_t* strides, int64_t size) {
        // Process 1D slices. This is all that iter.for_each is ever sending to the loop lambda
        execute_op(data, strides, 0, size, op);
    }, grain_size);
    iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op, const Range& range) {
    using traits = function_traits<func_t>;
    
    constexpr bool result_void = std::is_void_v<typename traits::result_type>;
    RAXIS_INTERNAL_ASSERT(
        iter.num_inputs() == traits::arity &&
        ((result_void && iter.num_outputs() == 0) || (!result_void && iter.num_outputs() == 1))
    );

    iter.serial_for_each([&](char** data, const int64_t* strides, int64_t n) {
        basic_loop(data, strides, 0, n, op);
    }, range);
    iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op) {
    cpu_serial_kernel(iter, std::forward<func_t>(op), {0, iter.numel()});
}

} // namespace raxis
