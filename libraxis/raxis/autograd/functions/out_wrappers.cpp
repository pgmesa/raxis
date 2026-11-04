#include <raxis/ops/binary_ops.h>
#include <raxis/ops/unary_ops.h>
#include <raxis/ops/pow.h>
#include <raxis/ops/scalar_ops.h>

#include <raxis/autograd/variable.h>


namespace raxis {

// From: the out= kernels of PyTorch's VariableType
// The public out= variants of differentiable ops: refuse arguments that
// require grad (the result could not be recorded), compute into out, and
// move its version, as an in-place op does

#define RAXIS_AUTOGRAD_UNARY_OUT(op_name)                                      \
    Tensor& op_name##_out(const Tensor& self, Tensor& out) {                   \
        autograd::check_out_no_grad(#op_name, {&self, &out});                  \
        ops::op_name##_out(self, out);                                         \
        autograd::bump_version(out);                                           \
        return out;                                                            \
    }

#define RAXIS_AUTOGRAD_BINARY_OUT(op_name)                                     \
    Tensor& op_name##_out(const Tensor& a, const Tensor& b, Tensor& out) {     \
        autograd::check_out_no_grad(#op_name, {&a, &b, &out});                 \
        ops::op_name##_out(a, b, out);                                         \
        autograd::bump_version(out);                                           \
        return out;                                                            \
    }

#define RAXIS_AUTOGRAD_BINARY_SCALAR_OUT(op_name)                              \
    Tensor& op_name##_out(const Tensor& a, Scalar s, Tensor& out) {            \
        autograd::check_out_no_grad(#op_name, {&a, &out});                     \
        ops::op_name##_out(a, s, out);                                         \
        autograd::bump_version(out);                                           \
        return out;                                                            \
    }

RAXIS_AUTOGRAD_UNARY_OUT(neg)
RAXIS_AUTOGRAD_UNARY_OUT(abs)
RAXIS_AUTOGRAD_UNARY_OUT(sign)
RAXIS_AUTOGRAD_UNARY_OUT(ceil)
RAXIS_AUTOGRAD_UNARY_OUT(floor)
RAXIS_AUTOGRAD_UNARY_OUT(round)
RAXIS_AUTOGRAD_UNARY_OUT(trunc)
RAXIS_AUTOGRAD_UNARY_OUT(frac)
RAXIS_AUTOGRAD_UNARY_OUT(exp)
RAXIS_AUTOGRAD_UNARY_OUT(expm1)
RAXIS_AUTOGRAD_UNARY_OUT(log)
RAXIS_AUTOGRAD_UNARY_OUT(log2)
RAXIS_AUTOGRAD_UNARY_OUT(log10)
RAXIS_AUTOGRAD_UNARY_OUT(log1p)
RAXIS_AUTOGRAD_UNARY_OUT(sqrt)
RAXIS_AUTOGRAD_UNARY_OUT(rsqrt)
RAXIS_AUTOGRAD_UNARY_OUT(reciprocal)
RAXIS_AUTOGRAD_UNARY_OUT(sigmoid)
RAXIS_AUTOGRAD_UNARY_OUT(sin)
RAXIS_AUTOGRAD_UNARY_OUT(cos)
RAXIS_AUTOGRAD_UNARY_OUT(tan)
RAXIS_AUTOGRAD_UNARY_OUT(asin)
RAXIS_AUTOGRAD_UNARY_OUT(acos)
RAXIS_AUTOGRAD_UNARY_OUT(atan)
RAXIS_AUTOGRAD_UNARY_OUT(sinh)
RAXIS_AUTOGRAD_UNARY_OUT(cosh)
RAXIS_AUTOGRAD_UNARY_OUT(tanh)
RAXIS_AUTOGRAD_UNARY_OUT(erf)
RAXIS_AUTOGRAD_UNARY_OUT(erfc)

RAXIS_AUTOGRAD_BINARY_OUT(add)
RAXIS_AUTOGRAD_BINARY_SCALAR_OUT(add)
RAXIS_AUTOGRAD_BINARY_OUT(sub)
RAXIS_AUTOGRAD_BINARY_SCALAR_OUT(sub)
RAXIS_AUTOGRAD_BINARY_OUT(mul)
RAXIS_AUTOGRAD_BINARY_SCALAR_OUT(mul)
RAXIS_AUTOGRAD_BINARY_OUT(div)
RAXIS_AUTOGRAD_BINARY_SCALAR_OUT(div)
RAXIS_AUTOGRAD_BINARY_OUT(maximum)
RAXIS_AUTOGRAD_BINARY_OUT(minimum)
RAXIS_AUTOGRAD_BINARY_OUT(pow)
RAXIS_AUTOGRAD_BINARY_SCALAR_OUT(pow)
RAXIS_AUTOGRAD_BINARY_OUT(remainder)
RAXIS_AUTOGRAD_BINARY_SCALAR_OUT(remainder)
RAXIS_AUTOGRAD_BINARY_OUT(fmod)
RAXIS_AUTOGRAD_BINARY_SCALAR_OUT(fmod)

Tensor& div_out(const Tensor& a, const Tensor& b, std::string_view rounding_mode, Tensor& out) {
    autograd::check_out_no_grad("div", {&a, &b, &out});
    ops::div_out(a, b, rounding_mode, out);
    autograd::bump_version(out);
    return out;
}

Tensor& div_out(const Tensor& a, Scalar s, std::string_view rounding_mode, Tensor& out) {
    return div_out(a, wrapped_scalar_tensor(s), rounding_mode, out);
}

// No derivative: nothing to refuse, only the version of out moves
Tensor& floor_divide_out(const Tensor& a, const Tensor& b, Tensor& out) {
    ops::div_out(a, b, "floor", out);
    autograd::bump_version(out);
    return out;
}

Tensor& floor_divide_out(const Tensor& a, Scalar s, Tensor& out) {
    return floor_divide_out(a, wrapped_scalar_tensor(s), out);
}

Tensor& pow_out(Scalar base, const Tensor& exp, Tensor& out) {
    autograd::check_out_no_grad("pow", {&exp, &out});
    ops::pow_out(base, exp, out);
    autograd::bump_version(out);
    return out;
}

#undef RAXIS_AUTOGRAD_UNARY_OUT
#undef RAXIS_AUTOGRAD_BINARY_OUT
#undef RAXIS_AUTOGRAD_BINARY_SCALAR_OUT

} // namespace raxis
