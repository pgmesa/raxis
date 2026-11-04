#pragma once

#include <string_view>

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
DECLARE_DISPATCH(binary_fn, div_trunc_stub);
DECLARE_DISPATCH(binary_fn, div_floor_stub);
DECLARE_DISPATCH(binary_fn, remainder_stub);
DECLARE_DISPATCH(binary_fn, fmod_stub);
DECLARE_DISPATCH(binary_fn, bitwise_and_stub);
DECLARE_DISPATCH(binary_fn, bitwise_or_stub);
DECLARE_DISPATCH(binary_fn, bitwise_xor_stub);
DECLARE_DISPATCH(binary_fn, lshift_stub);
DECLARE_DISPATCH(binary_fn, rshift_stub);


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
RAXIS_DECLARE_BINARY_OP_OUT(maximum);
RAXIS_DECLARE_BINARY_OP_OUT(minimum);

// Python's % (sign of the divisor) and C's fmod (sign of the dividend)
RAXIS_DECLARE_BINARY_ARITHMETIC(remainder)
RAXIS_DECLARE_BINARY_ARITHMETIC(fmod)
// s % t, as torch.remainder(Scalar, Tensor): no derivative
namespace ops { RAXIS_API Tensor remainder(Scalar s, const Tensor& b); }
RAXIS_API Tensor remainder(Scalar s, const Tensor& b);

// div with rounding_mode "trunc" or "floor" (an empty mode is the true
// division above); the result has a zero derivative, as in torch
namespace ops {
RAXIS_API Tensor& div_out(const Tensor& a, const Tensor& b, std::string_view rounding_mode, Tensor& out);
RAXIS_API Tensor div(const Tensor& a, const Tensor& b, std::string_view rounding_mode);
RAXIS_API Tensor div(const Tensor& a, Scalar s, std::string_view rounding_mode);
RAXIS_API Tensor& div_(Tensor& a, const Tensor& b, std::string_view rounding_mode);
RAXIS_API Tensor& div_(Tensor& a, Scalar s, std::string_view rounding_mode);
}
RAXIS_API Tensor& div_out(const Tensor& a, const Tensor& b, std::string_view rounding_mode, Tensor& out);
RAXIS_API Tensor& div_out(const Tensor& a, Scalar s, std::string_view rounding_mode, Tensor& out);
RAXIS_API Tensor div(const Tensor& a, const Tensor& b, std::string_view rounding_mode);
RAXIS_API Tensor div(const Tensor& a, Scalar s, std::string_view rounding_mode);
RAXIS_API Tensor& div_(Tensor& a, const Tensor& b, std::string_view rounding_mode);
RAXIS_API Tensor& div_(Tensor& a, Scalar s, std::string_view rounding_mode);

// --- Non-differentiable ---
// Their out= variants need no autograd check: only the version of out moves
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
RAXIS_API Tensor& gt_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& gt_out(const Tensor& a, Scalar s, Tensor& out);
RAXIS_API Tensor& ge_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& ge_out(const Tensor& a, Scalar s, Tensor& out);
RAXIS_API Tensor& lt_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& lt_out(const Tensor& a, Scalar s, Tensor& out);
RAXIS_API Tensor& le_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& le_out(const Tensor& a, Scalar s, Tensor& out);
RAXIS_API Tensor& eq_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& eq_out(const Tensor& a, Scalar s, Tensor& out);
RAXIS_API Tensor& ne_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& ne_out(const Tensor& a, Scalar s, Tensor& out);
 
// Logical
RAXIS_API Tensor logical_and(const Tensor& a, const Tensor& b);
RAXIS_API Tensor logical_or(const Tensor& a, const Tensor& b);
RAXIS_API Tensor& logical_and_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& logical_or_out(const Tensor& a, const Tensor& b, Tensor& out);

// Python's //: div(a, b, "floor"), but without a derivative (torch raises
// at backward)
RAXIS_API Tensor floor_divide(const Tensor& a, const Tensor& b);
RAXIS_API Tensor floor_divide(const Tensor& a, Scalar s);
RAXIS_API Tensor floor_divide(Scalar s, const Tensor& b);
RAXIS_API Tensor& floor_divide_(Tensor& a, const Tensor& b);
RAXIS_API Tensor& floor_divide_(Tensor& a, Scalar s);
RAXIS_API Tensor& floor_divide_out(const Tensor& a, const Tensor& b, Tensor& out);
RAXIS_API Tensor& floor_divide_out(const Tensor& a, Scalar s, Tensor& out);

// Bitwise ops: integer and bool tensors. A shift by a negative amount or by
// the bit width or more gives 0 (>> of a negative value: -1), as in torch
#define RAXIS_DECLARE_BITWISE_OP(op_name)                                      \
    RAXIS_API Tensor op_name(const Tensor& a, const Tensor& b);                \
    RAXIS_API Tensor op_name(const Tensor& a, Scalar s);                       \
    RAXIS_API Tensor op_name(Scalar s, const Tensor& b);                       \
    RAXIS_API Tensor& op_name##_(Tensor& a, const Tensor& b);                  \
    RAXIS_API Tensor& op_name##_(Tensor& a, Scalar s);                         \
    RAXIS_API Tensor& op_name##_out(const Tensor& a, const Tensor& b, Tensor& out); \
    RAXIS_API Tensor& op_name##_out(const Tensor& a, Scalar s, Tensor& out);

RAXIS_DECLARE_BITWISE_OP(bitwise_and)
RAXIS_DECLARE_BITWISE_OP(bitwise_or)
RAXIS_DECLARE_BITWISE_OP(bitwise_xor)
RAXIS_DECLARE_BITWISE_OP(bitwise_left_shift)
RAXIS_DECLARE_BITWISE_OP(bitwise_right_shift)

#undef RAXIS_DECLARE_BITWISE_OP

}