
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

// -- remainder, fmod --
RAXIS_AUTOGRAD_BINARY(remainder, RemainderBackward1)
RAXIS_AUTOGRAD_BINARY_SCALAR(remainder, RemainderBackward0)
RAXIS_AUTOGRAD_INPLACE_BINARY(remainder, RemainderBackward1)
RAXIS_AUTOGRAD_INPLACE_BINARY_SCALAR(remainder, RemainderBackward0)
RAXIS_AUTOGRAD_BINARY(fmod, FmodBackward1)
RAXIS_AUTOGRAD_BINARY_SCALAR(fmod, FmodBackward0)
RAXIS_AUTOGRAD_INPLACE_BINARY(fmod, FmodBackward1)
RAXIS_AUTOGRAD_INPLACE_BINARY_SCALAR(fmod, FmodBackward0)

// torch has no formula for remainder(Scalar, Tensor) either
Tensor remainder(Scalar s, const Tensor& b) {
    Tensor result = ops::remainder(s, b);
    if (autograd::requires_autograd(b)) {
        auto grad_fn = std::make_shared<autograd::NotImplementedBackward>("remainder");
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, b);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

// -- div with a rounding mode --
Tensor div(const Tensor& a, const Tensor& b, std::string_view rounding_mode) {
    Tensor result = ops::div(a, b, rounding_mode);
    if (autograd::requires_autograd(a, b)) {
        auto grad_fn = std::make_shared<autograd::DivBackward2>(a, b);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, a, b);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor div(const Tensor& a, Scalar s, std::string_view rounding_mode) {
    Tensor b = wrapped_scalar_tensor(s);
    Tensor result = ops::div(a, b, rounding_mode);
    if (autograd::requires_autograd(a)) {
        auto grad_fn = std::make_shared<autograd::DivBackward2>(a, b);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, a, b);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor& div_(Tensor& a, const Tensor& b, std::string_view rounding_mode) {
    const bool requires_grad = autograd::requires_autograd(a, b);
    autograd::check_inplace(a, requires_grad);
    std::shared_ptr<autograd::DivBackward2> grad_fn;
    if (requires_grad) {
        grad_fn = std::make_shared<autograd::DivBackward2>(a, b);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, a, b);
        autograd::save_original_self(*grad_fn, a);
    }
    ops::div_(a, b, rounding_mode);
    autograd::finish_inplace(a, grad_fn);
    return a;
}

Tensor& div_(Tensor& a, Scalar s, std::string_view rounding_mode) {
    const bool requires_grad = autograd::requires_autograd(a);
    autograd::check_inplace(a, requires_grad);
    Tensor b = wrapped_scalar_tensor(s);
    std::shared_ptr<autograd::DivBackward2> grad_fn;
    if (requires_grad) {
        grad_fn = std::make_shared<autograd::DivBackward2>(a, b);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, a, b);
        autograd::save_original_self(*grad_fn, a);
    }
    ops::div_(a, b, rounding_mode);
    autograd::finish_inplace(a, grad_fn);
    return a;
}

// -- floor_divide: div(a, b, "floor") with no derivative, as torch --
namespace {

Tensor not_differentiable(Tensor result, const char* name, const Tensor& a, const Tensor& b) {
    if (autograd::requires_autograd(a, b)) {
        auto grad_fn = std::make_shared<autograd::NotImplementedBackward>(name);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, a, b);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

} // anonymous namespace

Tensor floor_divide(const Tensor& a, const Tensor& b) {
    return not_differentiable(ops::div(a, b, "floor"), "floor_divide", a, b);
}

Tensor floor_divide(const Tensor& a, Scalar s) {
    Tensor b = wrapped_scalar_tensor(s);
    return not_differentiable(ops::div(a, b, "floor"), "floor_divide", a, b);
}

Tensor floor_divide(Scalar s, const Tensor& b) {
    Tensor a = wrapped_scalar_tensor(s);
    return not_differentiable(ops::div(a, b, "floor"), "floor_divide", a, b);
}

Tensor& floor_divide_(Tensor& a, const Tensor& b) {
    RAXIS_CHECK(!autograd::requires_autograd(a, b),
        "the derivative for 'floor_divide_' is not implemented");
    ops::div_(a, b, "floor");
    autograd::bump_version(a);
    return a;
}

Tensor& floor_divide_(Tensor& a, Scalar s) {
    RAXIS_CHECK(!autograd::requires_autograd(a),
        "the derivative for 'floor_divide_' is not implemented");
    ops::div_(a, s, "floor");
    autograd::bump_version(a);
    return a;
}
 
} // namespace raxis
