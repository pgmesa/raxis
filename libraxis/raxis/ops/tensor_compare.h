#pragma once

#include <optional>

#include <raxis/export.h>
#include <raxis/core/scalar.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

struct TensorIterator;
using where_fn = void(*)(TensorIterator&);
using clamp_scalar_fn = void(*)(TensorIterator&, Scalar, Scalar);
using clamp_bound_fn = void(*)(TensorIterator&, Scalar);

DECLARE_DISPATCH(where_fn, where_stub);
DECLARE_DISPATCH(clamp_scalar_fn, clamp_scalar_stub);
DECLARE_DISPATCH(clamp_bound_fn, clamp_min_scalar_stub);
DECLARE_DISPATCH(clamp_bound_fn, clamp_max_scalar_stub);

namespace ops {

RAXIS_API Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other);
RAXIS_API Tensor where(const Tensor& condition, const Tensor& self, Scalar other);
RAXIS_API Tensor where(const Tensor& condition, Scalar self, const Tensor& other);

RAXIS_API Tensor clamp(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max);
RAXIS_API Tensor& clamp_(Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max);

} // namespace ops

RAXIS_API Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other);
RAXIS_API Tensor where(const Tensor& condition, const Tensor& self, Scalar other);
RAXIS_API Tensor where(const Tensor& condition, Scalar self, const Tensor& other);
RAXIS_API Tensor where(const Tensor& condition, Scalar self, Scalar other);

RAXIS_API Tensor clamp(
    const Tensor& self,
    std::optional<Scalar> min = std::nullopt,
    std::optional<Scalar> max = std::nullopt
);
RAXIS_API Tensor& clamp_(
    Tensor& self,
    std::optional<Scalar> min = std::nullopt,
    std::optional<Scalar> max = std::nullopt
);
RAXIS_API Tensor clamp_min(const Tensor& self, Scalar min);
RAXIS_API Tensor clamp_max(const Tensor& self, Scalar max);

} // namespace raxis
