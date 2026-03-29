#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

class TensorIterator;
using unary_fn = void(*)(TensorIterator&);

DECLARE_DISPATCH(unary_fn, neg_stub);
DECLARE_DISPATCH(unary_fn, abs_stub);
DECLARE_DISPATCH(unary_fn, ceil_stub);
DECLARE_DISPATCH(unary_fn, floor_stub);
DECLARE_DISPATCH(unary_fn, round_stub);

DECLARE_DISPATCH(unary_fn, exp_stub);
DECLARE_DISPATCH(unary_fn, log_stub);
DECLARE_DISPATCH(unary_fn, sqrt_stub);
DECLARE_DISPATCH(unary_fn, sigmoid_stub);

DECLARE_DISPATCH(unary_fn, logical_not_stub);

RAXIS_API Tensor isnan(const Tensor& self);

RAXIS_API Tensor neg(const Tensor& self);
RAXIS_API Tensor& neg_(Tensor& self);

RAXIS_API Tensor abs(const Tensor& self);
RAXIS_API Tensor& abs_(Tensor& self);

RAXIS_API Tensor ceil(const Tensor& self);
RAXIS_API Tensor& ceil_(Tensor& self);

RAXIS_API Tensor floor(const Tensor& self);
RAXIS_API Tensor& floor_(Tensor& self);

RAXIS_API Tensor round(const Tensor& self);
RAXIS_API Tensor& round_(Tensor& self);

RAXIS_API Tensor exp(const Tensor& self);
RAXIS_API Tensor& exp_(Tensor& self);

RAXIS_API Tensor log(const Tensor& self);
RAXIS_API Tensor& log_(Tensor& self);

RAXIS_API Tensor sqrt(const Tensor& self);
RAXIS_API Tensor& sqrt_(Tensor& self);

RAXIS_API Tensor sigmoid(const Tensor& self);
RAXIS_API Tensor& sigmoid_(Tensor& self);

RAXIS_API Tensor logical_not(const Tensor& self);
RAXIS_API Tensor& logical_not_(Tensor& self);

} // namespace raxis