#pragma once

#include <raxis/export.h>
#include <raxis/ops/op_macros.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

struct TensorIterator;
using binary_fn = void(*)(TensorIterator&);

DECLARE_DISPATCH(binary_fn, add_stub);
DECLARE_DISPATCH(binary_fn, sub_stub);
DECLARE_DISPATCH(binary_fn, mul_stub);
DECLARE_DISPATCH(binary_fn, div_stub);
DECLARE_DISPATCH(binary_fn, gt_stub);
DECLARE_DISPATCH(binary_fn, ge_stub);
DECLARE_DISPATCH(binary_fn, lt_stub);
DECLARE_DISPATCH(binary_fn, le_stub);
DECLARE_DISPATCH(binary_fn, eq_stub);
DECLARE_DISPATCH(binary_fn, ne_stub);
DECLARE_DISPATCH(binary_fn, maximum_stub);
DECLARE_DISPATCH(binary_fn, minimum_stub);
DECLARE_DISPATCH(binary_fn, logical_and_stub);
DECLARE_DISPATCH(binary_fn, logical_or_stub);


RAXIS_API Tensor& add_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& sub_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& mul_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& div_out(const Tensor& a, const Tensor& b, Tensor& out);

// --- Differentiable ---
// Arithmetic
RAXIS_DECLARE_BINARY_ARITHMETIC(add);
RAXIS_DECLARE_BINARY_ARITHMETIC(sub);
RAXIS_DECLARE_BINARY_ARITHMETIC(mul);
RAXIS_DECLARE_BINARY_ARITHMETIC(div);

// Reversed
RAXIS_DECLARE_BINARY_REVERSED(sub)
RAXIS_DECLARE_BINARY_REVERSED(div)

// Elementwise
RAXIS_DECLARE_BINARY_OP(maximum);
RAXIS_DECLARE_BINARY_OP(minimum);

// --- Non-differentiable ---
// Comparison
RAXIS_API Tensor gt(const Tensor& a, const Tensor& b);
RAXIS_API Tensor gt(const Tensor& a, Scalar s);
RAXIS_API Tensor ge(const Tensor& a, const Tensor& b);
RAXIS_API Tensor ge(const Tensor& a, Scalar s);
RAXIS_API Tensor lt(const Tensor& a, const Tensor& b);
RAXIS_API Tensor lt(const Tensor& a, Scalar s);
RAXIS_API Tensor le(const Tensor& a, const Tensor& b);
RAXIS_API Tensor le(const Tensor& a, Scalar s);
RAXIS_API Tensor eq(const Tensor& a, const Tensor& b);
RAXIS_API Tensor eq(const Tensor& a, Scalar s);
RAXIS_API Tensor ne(const Tensor& a, const Tensor& b);
RAXIS_API Tensor ne(const Tensor& a, Scalar s);
 
// Logical
RAXIS_API Tensor logical_and(const Tensor& a, const Tensor& b);
RAXIS_API Tensor logical_or(const Tensor& a, const Tensor& b);

}