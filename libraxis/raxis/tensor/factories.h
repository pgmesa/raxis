#pragma once

#include <vector>
#include <functional>
#include <stdexcept>

#include <raxis/export.h>
#include <raxis/core/types.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>
#include <raxis/core/scalar.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_options.h>

namespace raxis {

using Deleter = std::function<void(void*)>;

RAXIS_API Tensor from_blob(
    void* data, 
    const IntArray& shape, 
    Dtype dtype,
    Device device = DeviceType::CPU
);

RAXIS_API Tensor from_blob(
    void* data,
    DeleterFnPtr deleter,
    const IntArray& shape,          
    Dtype dtype,
    Device device = DeviceType::CPU
);

RAXIS_API Tensor empty(const IntArray& shape, TensorOptions options = {});
RAXIS_API Tensor empty_like(const Tensor& other, TensorOptions options = {});
RAXIS_API Tensor empty_strided(
    const IntArray& shape, const IntArray& strides, TensorOptions options = {});

RAXIS_API Tensor full(const IntArray& shape, Scalar value, TensorOptions options = {});
RAXIS_API Tensor full_like(const Tensor& other, Scalar value, TensorOptions options = {});

RAXIS_API Tensor zeros(const IntArray& shape, TensorOptions options = {});
RAXIS_API Tensor zeros_like(const Tensor& other, TensorOptions options = {});

RAXIS_API Tensor ones(const IntArray& shape, TensorOptions options = {});
RAXIS_API Tensor ones_like(const Tensor& other, TensorOptions options = {});

RAXIS_API Tensor scalar_tensor(Scalar s, TensorOptions options = {});

// template<typename T>
// Tensor tensor(
//     const std::vector<T>& data,
//     const IntArray& shape,
//     Dtype dtype = default_dtype(),
//     Device device = DeviceType::CPU
// ) {
//     int64_t n = compute_numel(shape);
//     if (data.size() != n) {
//         throw std::runtime_error("data size mismatch");
//     }

//     Tensor out = empty(shape, dtype, device);
//     T* out_data = out.mutable_data_ptr<T>();
    
//     std::copy(data.begin(), data.end(), out_data);
//     return out;
// }

}
