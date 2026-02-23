#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

class TensorIterator;
using binary_fn = void(*)(TensorIterator&);

DECLARE_DISPATCH(binary_fn, add_stub);
DECLARE_DISPATCH(binary_fn, sub_stub);
DECLARE_DISPATCH(binary_fn, mul_stub);
DECLARE_DISPATCH(binary_fn, div_stub);

RAXIS_API Tensor& add(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& sub(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& mul(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& div(const Tensor& a, const Tensor& b, Tensor& out);

RAXIS_API Tensor add(const Tensor& a, const Tensor& b);
RAXIS_API Tensor add(const Tensor& a, Scalar s);
RAXIS_API Tensor sub(const Tensor& a, const Tensor& b);
RAXIS_API Tensor sub(const Tensor& a, Scalar s);
RAXIS_API Tensor mul(const Tensor& a, const Tensor& b);
RAXIS_API Tensor mul(const Tensor& a, Scalar s);
RAXIS_API Tensor div(const Tensor& a, const Tensor& b);
RAXIS_API Tensor div(const Tensor& a, Scalar s);

// In-place
RAXIS_API Tensor& add_(Tensor& a, const Tensor& b);
RAXIS_API Tensor& add_(Tensor& a, Scalar s);
RAXIS_API Tensor& sub_(Tensor& a, const Tensor& b);
RAXIS_API Tensor& sub_(Tensor& a, Scalar s);
RAXIS_API Tensor& mul_(Tensor& a, const Tensor& b);
RAXIS_API Tensor& mul_(Tensor& a, Scalar s);
RAXIS_API Tensor& div_(Tensor& a, const Tensor& b);
RAXIS_API Tensor& div_(Tensor& a, Scalar s);

// Reversed
RAXIS_API Tensor rsub(const Tensor& a, const Tensor& b);
RAXIS_API Tensor rsub(const Tensor& a, Scalar s);
RAXIS_API Tensor rdiv(const Tensor& a, const Tensor& b);
RAXIS_API Tensor rdiv(const Tensor& a, Scalar s);

RAXIS_API Tensor& rsub_(Tensor& a, const Tensor& b);
RAXIS_API Tensor& rsub_(Tensor& a, Scalar s);
RAXIS_API Tensor& rdiv_(Tensor& a, const Tensor& b);
RAXIS_API Tensor& rdiv_(Tensor& a, Scalar s);

}