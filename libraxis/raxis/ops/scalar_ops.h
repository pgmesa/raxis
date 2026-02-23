#pragma once

#include <raxis/core/scalar.h>
#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_options.h>

namespace raxis {

namespace detail {

Tensor& scalar_fill(Tensor& self, Scalar value);

Tensor scalar_tensor_static(
    Scalar s,
    std::optional<Dtype> dtype_opt,
    std::optional<Device> device_opt);

} // namespace detail

inline Tensor scalar_to_tensor(Scalar s, Device device = DeviceType::CPU) {
    if (device == DeviceType::CPU) {
        return detail::scalar_tensor_static(s, s.dtype(), DeviceType::CPU);
    }
    return scalar_tensor(s, ::raxis::dtype(s.dtype()).device(device));
}

inline Tensor wrapped_scalar_tensor(Scalar scalar, Device device = DeviceType::CPU) {
    auto tensor = scalar_to_tensor(scalar, device);
    tensor.unsafe_get_tensor_impl()->set_wrapped_number(true);
    return tensor;
}

} // namespace raxis


