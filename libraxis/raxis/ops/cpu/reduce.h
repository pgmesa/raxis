#pragma once

#include <raxis/core/scalar.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>

#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/vec/vec.h>

#include <algorithm>
#include <cstdint>
#include <utility>


namespace raxis {

template <typename F>
inline void unary_outer_loop(char* data[2], const int64_t strides[2],
                             int64_t n, F&& f) {
    for (int64_t j = 0; j < n; ++j) {
        f();
        data[0] += strides[0];
        data[1] += strides[1];
    }
}

using namespace vec;

template <typename func_t, typename vec_func_t>
inline void vectorized_reduction(char** data, int64_t n, int64_t stride,
                                 func_t op, vec_func_t vop, bool reduce) {
    using traits = function_traits<func_t>;
    using scalar_t = typename traits::result_type;
    using Vec = Vectorized<scalar_t>;

    char* out_ptr = data[0];
    const char* in_ptr = data[1];

    Vec acc[4];
    for (int j = 0; j < 4; ++j) {
        acc[j] = Vec::loadu(in_ptr + j * Vec::size() * sizeof(scalar_t));
    }
    for (int64_t i = 1; i < n; ++i) {
        const char* ptr = in_ptr + stride * i;
        acc[0] = vop(acc[0], Vec::loadu(ptr + 0 * Vec::size() * sizeof(scalar_t)));
        acc[1] = vop(acc[1], Vec::loadu(ptr + 1 * Vec::size() * sizeof(scalar_t)));
        acc[2] = vop(acc[2], Vec::loadu(ptr + 2 * Vec::size() * sizeof(scalar_t)));
        acc[3] = vop(acc[3], Vec::loadu(ptr + 3 * Vec::size() * sizeof(scalar_t)));
    }

    if (reduce) {
        scalar_t buffer[Vec::size()];
        acc[0] = vop(vop(acc[0], acc[1]), vop(acc[2], acc[3]));
        acc[0].store(buffer);
        for (int j = 1; j < Vec::size(); ++j) {
            buffer[0] = op(buffer[0], buffer[j]);
        }
        auto* dst = reinterpret_cast<scalar_t*>(out_ptr);
        *dst = op(*dst, buffer[0]);
    } else {
        for (int j = 0; j < 4; ++j) {
            auto* dst = out_ptr + j * Vec::size() * sizeof(scalar_t);
            acc[j] = vop(acc[j], Vec::loadu(dst));
            acc[j].store(dst);
        }
    }
}

template <typename func_t, typename vec_func_t>
inline void vectorized_inner_reduction(char** data, int64_t n,
                                       func_t op, vec_func_t vop) {
    using traits = function_traits<func_t>;
    using scalar_t = typename traits::result_type;
    using Vec = Vectorized<scalar_t>;

    constexpr int64_t vector_stride = 4 * Vec::size() * sizeof(scalar_t);
    int64_t count = n / (4 * Vec::size());

    if (count > 0) {
        vectorized_reduction(data, count, vector_stride, op, vop, /*reduce=*/true);
    }

    char* ptrs[3] = { data[0], data[0], data[1] };
    int64_t strides[3] = { 0, 0, static_cast<int64_t>(sizeof(scalar_t)) };
    basic_loop(ptrs, strides, count * 4 * Vec::size(), n, op);
}

template <typename func_t, typename vec_func_t>
inline void vectorized_outer_reduction(char** data, int64_t inner_stride,
                                       int64_t size0, int64_t size1,
                                       func_t op, vec_func_t vop) {
    using traits = function_traits<func_t>;
    using scalar_t = typename traits::result_type;
    using Vec = Vectorized<scalar_t>;

    constexpr int64_t vector_stride = 4 * Vec::size() * sizeof(scalar_t);
    int64_t outer_strides[2] = { vector_stride, vector_stride };

    unary_outer_loop(data, outer_strides, size1 / (4 * Vec::size()), [&] {
        vectorized_reduction(data, size0, inner_stride, op, vop, /*reduce=*/false);
    });

    int64_t step[2] = { static_cast<int64_t>(sizeof(scalar_t)),
                        static_cast<int64_t>(sizeof(scalar_t)) };
    int64_t remaining = size1 % (4 * Vec::size());
    unary_outer_loop(data, step, remaining, [&] {
        char* ptrs[3] = { data[0], data[0], data[1] };
        int64_t inner_strides[3] = { 0, 0, inner_stride };
        basic_loop(ptrs, inner_strides, 0, size0, op);
    });
}


template <typename scalar_t, typename func_t, typename vec_func_t>
void cpu_reduce_kernel_vec(TensorIterator& iter, func_t op, vec_func_t vop, scalar_t ident) {
    iter.get_output(0).fill_(ident);
    iter.parallel_reduce(
        [&op, &vop](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
            int64_t out_stride_0 = strides[0];
            int64_t in_stride_0  = strides[1];
            int64_t out_stride_1 = strides[2];
            int64_t in_stride_1  = strides[3];

            bool is_contiguous_inner =
                (out_stride_0 == 0) &&
                (in_stride_0 == static_cast<int64_t>(sizeof(scalar_t)));
            bool is_contiguous_outer =
                (out_stride_0 == 0) &&
                (out_stride_1 == static_cast<int64_t>(sizeof(scalar_t))) &&
                (in_stride_1 == static_cast<int64_t>(sizeof(scalar_t)));

            if (is_contiguous_inner) {
                char* data_copy[2] = { data[0], data[1] };
                int64_t outer_strides[] = { out_stride_1, in_stride_1 };
                unary_outer_loop(data_copy, outer_strides, size1, [&] {
                    vectorized_inner_reduction(data_copy, size0, op, vop);
                });
            } else if (is_contiguous_outer) {
                vectorized_outer_reduction(data, in_stride_0, size0, size1, op, vop);
            } else {
                char* data_copy[2] = { data[0], data[1] };
                int64_t outer_strides[] = { out_stride_1, in_stride_1 };
                unary_outer_loop(data_copy, outer_strides, size1, [&] {
                    char* ptrs[3] = { data[0], data[0], data[1] };
                    int64_t inner_strides[3] = { out_stride_0, out_stride_0, in_stride_0 };
                    basic_loop(ptrs, inner_strides, 0, size0, op);
                });
            }
        }
    );
}

} // namespace raxis