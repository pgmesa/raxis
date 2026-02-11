#pragma once

#include <vector>
#include <functional>
#include <stdexcept>

#include <raxis/core/export.h>
#include <raxis/core/types.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

using Deleter = std::function<void(void*)>;

RAXIS_API Tensor from_blob(
    void* data, 
    const IntArray& shape, 
    Dtype dtype = default_dtype()
);

RAXIS_API Tensor from_blob(
    void* data,
    const IntArray& shape,          
    Dtype dtype,
    detail::DeleterFnPtr deleter
);

RAXIS_API Tensor empty(
    const IntArray& shape, 
    Dtype dtype = default_dtype()
);

RAXIS_API Tensor empty_strided(
    const IntArray& shape,
    const IntArray& strides,
    Dtype dtype = default_dtype()
);

RAXIS_API Tensor full(
    const IntArray& shape, 
    double value, 
    Dtype dtype = default_dtype()
);

RAXIS_API Tensor ones(const IntArray& shape, Dtype dtype = default_dtype());
RAXIS_API Tensor zeros(const IntArray& shape, Dtype dtype = default_dtype());

template<typename T>
Tensor tensor(
    const std::vector<T>& data,
    const IntArray& shape,
    Dtype dtype = default_dtype()
) {
    int64_t n = compute_numel(shape);
    if (data.size() != n) {
        throw std::runtime_error("data size mismatch");
    }

    Tensor out = empty(shape, dtype);
    T* out_data = out.mutable_data_ptr<T>();
    
    std::copy(data.begin(), data.end(), out_data);
    return out;
}


// Tensor scalar_tensor(Scalar s);

}
