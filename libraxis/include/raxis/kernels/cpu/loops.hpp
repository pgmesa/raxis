#pragma once

#include <raxis/core/array_iterator.hpp>


namespace raxis::kernels::cpu {

// For now, cpu_kernel is aware of the type directly and only 2 inputs and 1 output supported
template <typename scalar_t, typename func_t>
void cpu_kernel(const ArrayIterator& iter, func_t&& op) {
    iter.for_each([&](char** data, const int64_t* strides, int64_t size) {
        for (int64_t i = 0; i < size; i++) {
            scalar_t* out_ptr = reinterpret_cast<scalar_t*>(data[0] + i * strides[0]);
            scalar_t* in1_ptr = reinterpret_cast<scalar_t*>(data[1] + i * strides[1]);
            scalar_t* in2_ptr = reinterpret_cast<scalar_t*>(data[2] + i * strides[2]);
            
            *out_ptr = op(*in1_ptr, *in2_ptr);
        }
    });
}

}