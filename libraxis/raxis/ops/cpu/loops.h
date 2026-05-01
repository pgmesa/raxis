#pragma once

#include <stdexcept>
#include <type_traits>

#include <raxis/macros.h>

#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/is_contiguous.h>

#include <raxis/util/exception.h>
#include <raxis/util/function_traits.h>

#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

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

template <typename traits, std::size_t... INDEX>
typename traits::ArgsTuple
dereference_vec_impl(
    char* RAXIS_RESTRICT data[],
    const typename traits::result_type& opt_scalar,
    size_t S,
    int64_t i,
    std::index_sequence<INDEX...>
) {
    using Vec = typename traits::result_type;
    using scalar_t = typename Vec::value_type;
    return std::make_tuple(
        (S == INDEX + 1
            ? opt_scalar
            : Vec::loadu(data[INDEX] + i * sizeof(scalar_t)))...
    );
}

template <typename traits>
typename traits::ArgsTuple
dereference_vec(
    char* RAXIS_RESTRICT data[],
    const typename traits::result_type& opt_scalar,
    size_t S,
    int64_t i
) {
    using Indices = std::make_index_sequence<traits::arity>;
    return dereference_vec_impl<traits>(data, opt_scalar, S, i, Indices{});
}

template <typename func_t>
inline void execute_op(
    char* RAXIS_RESTRICT data[],
    const int64_t* strides,
    int64_t i,
    int64_t n,
    func_t&& op
) {
    using traits = function_traits<func_t>;

    for (; i < n; i++) {
        if constexpr (std::is_void_v<typename traits::result_type>) {
            std::apply(op, dereference<traits>(&data[0], &strides[0], i));
        } else {
            using result_type = typename traits::result_type;
            auto* out_ptr = reinterpret_cast<result_type*>(data[0] + i * strides[0]);
            *out_ptr = std::apply(op, dereference<traits>(&data[1], &strides[1], i));
        }
    }
}

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

template <typename func_t, typename vec_func_t>
inline void
vectorized_loop(
    char** RAXIS_RESTRICT data_,
    int64_t n,
    int64_t S,
    func_t&& op,
    vec_func_t&& vop
) {
    using traits = function_traits<vec_func_t>;
    using scalar_t = typename function_traits<func_t>::result_type;
    using Vec = Vectorized<scalar_t>;
    constexpr int ntensors = traits::arity + 1;

    char* RAXIS_RESTRICT data[ntensors];
    for (int arg = 0; arg < ntensors; arg++) {
        data[arg] = data_[arg];
    }

    Vec opt_scalar = Vec(S > 0 ? load<scalar_t>(data[S]) : scalar_t(0));
    int64_t i = 0;
    for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
        auto args1 = dereference_vec<traits>(&data[1], opt_scalar, S, i);
        auto args2 = dereference_vec<traits>(&data[1], opt_scalar, S, i + Vec::size());
        auto out1 = std::apply(vop, std::move(args1));
        auto out2 = std::apply(vop, std::move(args2));
        out1.store(data[0] + i * sizeof(scalar_t));
        out2.store(data[0] + (i + Vec::size()) * sizeof(scalar_t));
    }
    if (i < n) {
        int64_t strides[ntensors];
        for (int arg = 0; arg < ntensors; arg++) {
            strides[arg] = (S > 0 && arg == static_cast<int>(S)) ? 0 : sizeof(scalar_t);
        }
        basic_loop(data, strides, i, n, std::forward<func_t>(op));
    }
}

template <typename traits, typename cb_t>
inline void unroll_contiguous_scalar_checks(
    const int64_t* /*strides*/,
    std::index_sequence<>,
    cb_t&& cb
) {
    cb(0);
}

template <typename traits, typename cb_t, size_t INDEX0, size_t... INDEX>
inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    std::index_sequence<INDEX0, INDEX...>,
    cb_t&& cb
) {
    if (is_contiguous_scalar<traits, INDEX0 + 1>(strides)) {
        cb(INDEX0 + 1);
    } else {
        unroll_contiguous_scalar_checks<traits>(
            strides,
            std::index_sequence<INDEX...>{},
            std::forward<cb_t>(cb)
        );
    }
}

template <typename op_t, typename vop_t>
struct VectorizedLoop2d {
    op_t op;
    vop_t vop;

    using traits = function_traits<op_t>;
    static constexpr int ntensors = traits::arity + 1;
    using data_t = std::array<char*, ntensors>;

    VectorizedLoop2d(op_t op, vop_t vop)
        : op(std::move(op))
        , vop(std::move(vop))
    {}

    static void advance(data_t& data, const int64_t* outer_strides) {
        for (int arg = 0; arg < ntensors; arg++) {
            data[arg] += outer_strides[arg];
        }
    }

    void operator()(
        char** base,
        const int64_t* strides,
        int64_t size0,
        int64_t size1
    ) {
        data_t data;
        std::copy_n(base, ntensors, data.data());
        const int64_t* outer_strides = &strides[ntensors];

        if (is_contiguous<traits>(strides)) {
            for (int64_t i = 0; i < size1; i++) {
                vectorized_loop(data.data(), size0, 0, op, vop);
                advance(data, outer_strides);
            }
        } else {
            using Indices = std::make_index_sequence<traits::arity>;
            unroll_contiguous_scalar_checks<traits>(
                strides, Indices{}, [&](size_t idx) {
                    if (idx) {
                        for (int64_t i = 0; i < size1; i++) {
                            vectorized_loop(data.data(), size0, idx, op, vop);
                            advance(data, outer_strides);
                        }
                    } else {
                        for (int64_t i = 0; i < size1; i++) {
                            basic_loop(data.data(), strides, 0, size0, op);
                            advance(data, outer_strides);
                        }
                    }
                }
            );
        }
    }
};

template <typename op_t, typename vop_t>
VectorizedLoop2d<op_t, vop_t> make_vectorized_loop2d(op_t&& op, vop_t&& vop) {
    return VectorizedLoop2d<op_t, vop_t>(
        std::forward<op_t>(op),
        std::forward<vop_t>(vop)
    );
}

template <typename func_t>
void cpu_kernel(TensorIterator& iter, func_t&& op, int64_t grain_size = detail::GRAIN_SIZE) {
    using traits = function_traits<func_t>;
    
    RAXIS_INTERNAL_ASSERT(iter.num_inputs() == traits::arity);
    RAXIS_INTERNAL_ASSERT(iter.num_outputs() == 1);
    
    iter.for_each([&](char** data, const int64_t* strides, int64_t size) {
        // Process 1D slices. This is all that iter.for_each is ever sending to the loop lambda
        basic_loop(data, strides, 0, size, op);
    }, grain_size);
    iter.cast_outputs();
}

template <typename func_t, typename vec_func_t>
void cpu_kernel_vec(
    TensorIterator& iter,
    func_t&& op,
    vec_func_t&& vop,
    int64_t grain_size = detail::GRAIN_SIZE
) {
    using traits = function_traits<func_t>;
    
    RAXIS_INTERNAL_ASSERT(iter.num_inputs() == traits::arity);
    RAXIS_INTERNAL_ASSERT(iter.num_outputs() == 1);

    iter.for_each(
        make_vectorized_loop2d(
            std::forward<func_t>(op),
            std::forward<vec_func_t>(vop)
        ),
        grain_size
    );
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

    iter.serial_for_each([&](char** data, const int64_t* strides, int64_t size) {
        basic_loop(data, strides, 0, size, op);
    }, range);
    iter.cast_outputs();
}

template <typename func_t>
void cpu_serial_kernel(TensorIterator& iter, func_t&& op) {
    cpu_serial_kernel(iter, std::forward<func_t>(op), {0, iter.numel()});
}


template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIterator& iter, func_t&& op, vec_func_t&& vop, const Range& range) {
    using traits = function_traits<func_t>;
    RAXIS_INTERNAL_ASSERT(iter.num_inputs() == traits::arity);
    RAXIS_INTERNAL_ASSERT(iter.num_outputs() == 1);

    iter.serial_for_each(make_vectorized_loop2d(std::forward<func_t>(op), std::forward<vec_func_t>(vop)), range);
    iter.cast_outputs();
}

template <typename func_t, typename vec_func_t>
void cpu_serial_kernel_vec(TensorIterator& iter, func_t&& op, vec_func_t&& vop) {
    cpu_serial_kernel_vec(iter, std::forward<func_t>(op), std::forward<vec_func_t>(vop), {0, iter.numel()});
}

}} // namespace raxis::CPU_CAPABILITY
