#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

struct TensorIterator;
using unary_fn = void(*)(TensorIterator&);

inline namespace CPU_CAPABILITY {

void neg_kernel(TensorIterator &iter);
void sqrt_kernel(TensorIterator& iter);
void rsqrt_kernel(TensorIterator& iter);
void reciprocal_kernel(TensorIterator& iter);

} // namespace CPU_CAPABILITY

DECLARE_DISPATCH(unary_fn, neg_stub);
DECLARE_DISPATCH(unary_fn, abs_stub);
DECLARE_DISPATCH(unary_fn, ceil_stub);
DECLARE_DISPATCH(unary_fn, floor_stub);
DECLARE_DISPATCH(unary_fn, round_stub);

DECLARE_DISPATCH(unary_fn, exp_stub);
DECLARE_DISPATCH(unary_fn, expm1_stub);
DECLARE_DISPATCH(unary_fn, log_stub);
DECLARE_DISPATCH(unary_fn, log2_stub);
DECLARE_DISPATCH(unary_fn, log10_stub);
DECLARE_DISPATCH(unary_fn, log1p_stub);
DECLARE_DISPATCH(unary_fn, sqrt_stub);
DECLARE_DISPATCH(unary_fn, rsqrt_stub);
DECLARE_DISPATCH(unary_fn, reciprocal_stub);
DECLARE_DISPATCH(unary_fn, sigmoid_stub);
DECLARE_DISPATCH(unary_fn, logical_not_stub);

DECLARE_DISPATCH(unary_fn, sin_stub);
DECLARE_DISPATCH(unary_fn, cos_stub);
DECLARE_DISPATCH(unary_fn, tan_stub);
DECLARE_DISPATCH(unary_fn, asin_stub);
DECLARE_DISPATCH(unary_fn, acos_stub);
DECLARE_DISPATCH(unary_fn, atan_stub);
DECLARE_DISPATCH(unary_fn, sinh_stub);
DECLARE_DISPATCH(unary_fn, cosh_stub);
DECLARE_DISPATCH(unary_fn, tanh_stub);
DECLARE_DISPATCH(unary_fn, erf_stub);
DECLARE_DISPATCH(unary_fn, erfc_stub);

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

RAXIS_API Tensor expm1(const Tensor& self);
RAXIS_API Tensor& expm1_(Tensor& self);

RAXIS_API Tensor log(const Tensor& self);
RAXIS_API Tensor& log_(Tensor& self);

RAXIS_API Tensor log2(const Tensor& self);
RAXIS_API Tensor& log2_(Tensor& self);

RAXIS_API Tensor log10(const Tensor& self);
RAXIS_API Tensor& log10_(Tensor& self);

RAXIS_API Tensor log1p(const Tensor& self);
RAXIS_API Tensor& log1p_(Tensor& self);

RAXIS_API Tensor sqrt(const Tensor& self);
RAXIS_API Tensor& sqrt_(Tensor& self);

RAXIS_API Tensor rsqrt(const Tensor& a);
RAXIS_API Tensor& rsqrt_(Tensor& a);

RAXIS_API Tensor reciprocal(const Tensor& a);
RAXIS_API Tensor& reciprocal_(Tensor& a);

RAXIS_API Tensor sigmoid(const Tensor& self);
RAXIS_API Tensor& sigmoid_(Tensor& self);

RAXIS_API Tensor sin(const Tensor& self);
RAXIS_API Tensor& sin_(Tensor& self);

RAXIS_API Tensor cos(const Tensor& self);
RAXIS_API Tensor& cos_(Tensor& self);

RAXIS_API Tensor tan(const Tensor& self);
RAXIS_API Tensor& tan_(Tensor& self);

RAXIS_API Tensor asin(const Tensor& self);
RAXIS_API Tensor& asin_(Tensor& self);

RAXIS_API Tensor acos(const Tensor& self);
RAXIS_API Tensor& acos_(Tensor& self);

RAXIS_API Tensor atan(const Tensor& self);
RAXIS_API Tensor& atan_(Tensor& self);

RAXIS_API Tensor sinh(const Tensor& self);
RAXIS_API Tensor& sinh_(Tensor& self);

RAXIS_API Tensor cosh(const Tensor& self);
RAXIS_API Tensor& cosh_(Tensor& self);

RAXIS_API Tensor tanh(const Tensor& self);
RAXIS_API Tensor& tanh_(Tensor& self);

RAXIS_API Tensor erf(const Tensor& self);
RAXIS_API Tensor& erf_(Tensor& self);

RAXIS_API Tensor erfc(const Tensor& self);
RAXIS_API Tensor& erfc_(Tensor& self);

RAXIS_API Tensor logical_not(const Tensor& self);
RAXIS_API Tensor& logical_not_(Tensor& self);

} // namespace raxis