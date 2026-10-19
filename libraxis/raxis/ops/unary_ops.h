#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/op_macros.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

struct TensorIterator;
using unary_fn = void(*)(TensorIterator&);

inline namespace CPU_CAPABILITY {

void neg_kernel(TensorIterator& iter);
void sqrt_kernel(TensorIterator& iter);
void rsqrt_kernel(TensorIterator& iter);
void reciprocal_kernel(TensorIterator& iter);

} // namespace CPU_CAPABILITY

DECLARE_DISPATCH(unary_fn, neg_stub);
DECLARE_DISPATCH(unary_fn, abs_stub);
DECLARE_DISPATCH(unary_fn, sign_stub);
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

// Differentiable unary ops
RAXIS_DECLARE_UNARY_ARITHMETIC(neg)
RAXIS_DECLARE_UNARY_ARITHMETIC(abs)
RAXIS_DECLARE_UNARY_ARITHMETIC(sign)
RAXIS_DECLARE_UNARY_ARITHMETIC(ceil)
RAXIS_DECLARE_UNARY_ARITHMETIC(floor)
RAXIS_DECLARE_UNARY_ARITHMETIC(round)
RAXIS_DECLARE_UNARY_ARITHMETIC(exp)
RAXIS_DECLARE_UNARY_ARITHMETIC(expm1)
RAXIS_DECLARE_UNARY_ARITHMETIC(log)
RAXIS_DECLARE_UNARY_ARITHMETIC(log2)
RAXIS_DECLARE_UNARY_ARITHMETIC(log10)
RAXIS_DECLARE_UNARY_ARITHMETIC(log1p)
RAXIS_DECLARE_UNARY_ARITHMETIC(sqrt)
RAXIS_DECLARE_UNARY_ARITHMETIC(rsqrt)
RAXIS_DECLARE_UNARY_ARITHMETIC(reciprocal)
RAXIS_DECLARE_UNARY_ARITHMETIC(sigmoid)
RAXIS_DECLARE_UNARY_ARITHMETIC(sin)
RAXIS_DECLARE_UNARY_ARITHMETIC(cos)
RAXIS_DECLARE_UNARY_ARITHMETIC(tan)
RAXIS_DECLARE_UNARY_ARITHMETIC(asin)
RAXIS_DECLARE_UNARY_ARITHMETIC(acos)
RAXIS_DECLARE_UNARY_ARITHMETIC(atan)
RAXIS_DECLARE_UNARY_ARITHMETIC(sinh)
RAXIS_DECLARE_UNARY_ARITHMETIC(cosh)
RAXIS_DECLARE_UNARY_ARITHMETIC(tanh)
RAXIS_DECLARE_UNARY_ARITHMETIC(erf)
RAXIS_DECLARE_UNARY_ARITHMETIC(erfc)

// Non-differentiable
RAXIS_API Tensor isnan(const Tensor& self);
RAXIS_API Tensor logical_not(const Tensor& self);
RAXIS_API Tensor& logical_not_(Tensor& self);

} // namespace raxis