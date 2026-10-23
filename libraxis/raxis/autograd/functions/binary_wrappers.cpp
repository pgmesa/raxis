
#include <raxis/ops/binary_ops.h>
#include <raxis/ops/pow.h>
#include <raxis/ops/scalar_ops.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_iterator.h>
 
#include <raxis/autograd/engine.h>
#include <raxis/autograd/variable.h>
#include <raxis/autograd/functions/binary_backward.h>
#include <raxis/autograd/autograd_macros.h>


namespace raxis {

RAXIS_AUTOGRAD_BINARY_ALL(add, AddBackward0)
RAXIS_AUTOGRAD_BINARY_ALL(sub, SubBackward0)
RAXIS_AUTOGRAD_BINARY_ALL(mul, MulBackward0)
RAXIS_AUTOGRAD_BINARY_ALL(div, DivBackward0)
 
// -- Reversed ops --
Tensor rsub(const Tensor& a, const Tensor& b) {
    Tensor result = ops::rsub(a, b);
    if (autograd::requires_autograd(a, b)) {
        auto grad_fn = std::make_shared<autograd::SubBackward0>(b, a);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, b, a);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}
 
Tensor rsub(const Tensor& a, Scalar s) {
    Tensor b = wrapped_scalar_tensor(s);
    Tensor result = ops::rsub(a, b);
    if (autograd::requires_autograd(a)) {
        auto grad_fn = std::make_shared<autograd::SubBackward0>(b, a);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, b, a);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}
 
Tensor rdiv(const Tensor& a, const Tensor& b) {
    Tensor result = ops::rdiv(a, b);
    if (autograd::requires_autograd(a, b)) {
        auto grad_fn = std::make_shared<autograd::DivBackward0>(b, a);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, b, a);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}
 
Tensor rdiv(const Tensor& a, Scalar s) {
    Tensor b = wrapped_scalar_tensor(s);
    Tensor result = ops::rdiv(a, b);
    if (autograd::requires_autograd(a)) {
        auto grad_fn = std::make_shared<autograd::DivBackward0>(b, a);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, b, a);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}
 
// In-place reversed
RAXIS_AUTOGRAD_INPLACE_NO_GRAD_BINARY_TENSOR(rsub)
RAXIS_AUTOGRAD_INPLACE_NO_GRAD_BINARY_TENSOR(rdiv)
RAXIS_AUTOGRAD_INPLACE_NO_GRAD_BINARY_SCALAR(rsub)
RAXIS_AUTOGRAD_INPLACE_NO_GRAD_BINARY_SCALAR(rdiv)

// -- Pow --
Tensor pow(const Tensor& base, Scalar exp) {
    Tensor result = ops::pow(base, exp);
    if (autograd::requires_autograd(base)) {
        auto grad_fn = std::make_shared<autograd::PowBackward0>(base, exp);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, base);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor pow(const Tensor& base, const Tensor& exp) {
    Tensor result = ops::pow(base, exp);
    if (autograd::requires_autograd(base, exp)) {
        auto grad_fn = std::make_shared<autograd::PowBackward1>(base, exp, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, base, exp);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor pow(Scalar base, const Tensor& exp) {
    Tensor result = ops::pow(base, exp);
    if (autograd::requires_autograd(exp)) {
        auto grad_fn = std::make_shared<autograd::PowBackward2>(base, exp, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, exp);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

RAXIS_AUTOGRAD_BINARY(maximum, MaximumBackward0)
RAXIS_AUTOGRAD_BINARY(minimum, MinimumBackward0)
 
} // namespace raxis
