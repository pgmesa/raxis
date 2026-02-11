#pragma once

#include <raxis/core/export.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

RAXIS_API Tensor& copy(const Tensor& a, Tensor& out);
RAXIS_API Tensor clone(const Tensor& a);
RAXIS_API Tensor to(const Tensor& a, Dtype dtype);

RAXIS_API Tensor& add(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& sub(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& mul(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& div(const Tensor& a, const Tensor& b, Tensor& out);

RAXIS_API Tensor add(const Tensor& a, const Tensor& b);
// RAXIS_API Tensor add(const Tensor& a, Scalar b);
RAXIS_API Tensor sub(const Tensor& a, const Tensor& b);
// RAXIS_API Tensor sub(const Tensor& a, Scalar b);
RAXIS_API Tensor mul(const Tensor& a, const Tensor& b);
// RAXIS_API Tensor mul(const Tensor& a, Scalar b);
RAXIS_API Tensor div(const Tensor& a, const Tensor& b);
// RAXIS_API Tensor div(const Tensor& a, Scalar b);

// In-place
RAXIS_API Tensor& add_(Tensor& a, const Tensor& b);
// RAXIS_API Tensor& add_(Tensor& a, Scalar b);
RAXIS_API Tensor& sub_(Tensor& a, const Tensor& b);
// RAXIS_API Tensor& sub_(Tensor& a, Scalar b);
RAXIS_API Tensor& mul_(Tensor& a, const Tensor& b);
// RAXIS_API Tensor& mul_(Tensor& a, Scalar b);
RAXIS_API Tensor& div_(Tensor& a, const Tensor& b);
// RAXIS_API Tensor& div_(Tensor& a, Scalar b);

// Reversed
RAXIS_API Tensor rsub(const Tensor& a, const Tensor& b);
// RAXIS_API Tensor rsub(const Tensor& a, Scalar b);
RAXIS_API Tensor rdiv(const Tensor& a, const Tensor& b);
// RAXIS_API Tensor rdiv(const Tensor& a, Scalar b);

RAXIS_API Tensor& rsub_(Tensor& a, const Tensor& b);
// RAXIS_API Tensor& rsub_(Tensor& a, Scalar b);
RAXIS_API Tensor& rdiv_(Tensor& a, const Tensor& b);
// RAXIS_API Tensor& rsub_(Tensor& a, Scalar b);

}