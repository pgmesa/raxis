#pragma once

#include <raxis/core/scalar.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>

#include <raxis/ops/cpu/loops.h>

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


template <typename scalar_t, typename func_t>
inline void scalar_inner_reduction(char** data, int64_t size0, func_t op) {
    char* out_ptr = data[0];
    char* in_ptr = data[1];
    auto* out = reinterpret_cast<scalar_t*>(out_ptr);

    for (int64_t i = 0; i < size0; ++i) {
        auto val = *reinterpret_cast<const scalar_t*>(in_ptr);
        *out = op(*out, val);
        in_ptr += sizeof(scalar_t);
    }
}


template <typename scalar_t, typename func_t>
inline void scalar_outer_reduction(char** data, int64_t inner_stride,
                                   int64_t size0, int64_t size1, func_t op) {
    for (int64_t i = 0; i < size0; ++i) {
        char* out_ptr = data[0];
        char* in_ptr = data[1] + i * inner_stride;

        for (int64_t j = 0; j < size1; ++j) {
            auto* out = reinterpret_cast<scalar_t*>(out_ptr);
            auto val = *reinterpret_cast<const scalar_t*>(in_ptr);
            *out = op(*out, val);
            out_ptr += sizeof(scalar_t);
            in_ptr += sizeof(scalar_t);
        }
    }
}

template <typename scalar_t, typename func_t>
void cpu_reduce_kernel(
    TensorIterator& iter, func_t op, double ident)
{
    iter.get_output(0).fill_(ident);
    iter.parallel_reduce(
        [&op](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
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
                // input is contiguous in dim 0, output is reduced in dim 0
                char* data_copy[2] = { data[0], data[1] };
                int64_t outer_strides[] = { out_stride_1, in_stride_1 };

                unary_outer_loop(data_copy, outer_strides, size1, [&] {
                    scalar_inner_reduction<scalar_t>(data_copy, size0, op);
                });
            } else if (is_contiguous_outer) {
                // input and output are contiguous in dim 1
                scalar_outer_reduction<scalar_t>(data, in_stride_0, size0, size1, op);
            } else {
                char* data_copy[2] = { data[0], data[1] };
                int64_t outer_strides[] = { out_stride_1, in_stride_1 };

                // Generic fallback: no contiguity assumptions
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