
#include <raxis/autograd/functions/binary_backward.h>

#include <raxis/autograd/variable.h>
#include <raxis/ops/binary_ops.h>


namespace raxis::autograd {

AddBackward0::AddBackward0(const Tensor& a, const Tensor& b)
    : a_requires_grad_(a.requires_grad()),
      b_requires_grad_(b.requires_grad())
{
    if (a_requires_grad_) a_shape_ = DimVector(a.shape());
    if (b_requires_grad_) b_shape_ = DimVector(b.shape());
}

std::string AddBackward0::name() const { return "AddBackward0"; }

TensorList AddBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_a, grad_b;

    if (a_requires_grad_)
        grad_a = unbroadcast(grad, a_shape_);
    if (b_requires_grad_)
        grad_b = unbroadcast(grad, b_shape_);

    return {grad_a, grad_b};
}


SubBackward0::SubBackward0(const Tensor& a, const Tensor& b)
    : a_requires_grad_(a.requires_grad()),
      b_requires_grad_(b.requires_grad())
{
    if (a_requires_grad_) a_shape_ = DimVector(a.shape());
    if (b_requires_grad_) b_shape_ = DimVector(b.shape());
}

std::string SubBackward0::name() const { return "SubBackward0"; }

TensorList SubBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_a, grad_b;

    if (a_requires_grad_)
        grad_a = unbroadcast(grad, a_shape_);
    if (b_requires_grad_)
        grad_b = unbroadcast(grad, b_shape_).neg();

    return {grad_a, grad_b};
}


MulBackward0::MulBackward0(const Tensor& a, const Tensor& b)
    : a_requires_grad_(a.requires_grad()),
      b_requires_grad_(b.requires_grad())
{
    if (a_requires_grad_) {
        b_ = b;
        a_shape_ = DimVector(a.shape());
    }
    if (b_requires_grad_) {
        a_ = a;
        b_shape_ = DimVector(b.shape());
    }
}

std::string MulBackward0::name() const { return "MulBackward0"; }

TensorList MulBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_a, grad_b;

    if (a_requires_grad_)
        grad_a = unbroadcast(grad * b_, a_shape_);
    if (b_requires_grad_)
        grad_b = unbroadcast(grad * a_, b_shape_);

    return {grad_a, grad_b};
}


DivBackward0::DivBackward0(const Tensor& a, const Tensor& b)
    : a_requires_grad_(a.requires_grad()),
      b_requires_grad_(b.requires_grad())
{
    if (a_requires_grad_) {
        b_ = b;
        a_shape_ = DimVector(a.shape());
    }
    if (b_requires_grad_) {
        a_ = a;
        b_ = b;
        b_shape_ = DimVector(b.shape());
    }
}

std::string DivBackward0::name() const { return "DivBackward0"; }

TensorList DivBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_a, grad_b;

    if (a_requires_grad_)
        grad_a = unbroadcast(grad / b_, a_shape_);

    if (b_requires_grad_) {
        // d/db(a/b) = -a / b^2
        grad_b = unbroadcast(-(grad * a_) / (b_ * b_), b_shape_);
    }

    return {grad_a, grad_b};
}


PowBackward0::PowBackward0(const Tensor& base, Scalar exp)
    : base_(base), exp_(exp) {}

std::string PowBackward0::name() const { return "PowBackward0"; }

TensorList PowBackward0::apply(const TensorList& grads) {
    // d/dbase = exp * base^(exp-1)
    double e = exp_.to<double>();
    return {grads[0] * e * base_.pow(e - 1.0)};
}

PowBackward1::PowBackward1(const Tensor& base, const Tensor& exp, const Tensor& result)
    : base_requires_grad_(base.requires_grad()),
      exp_requires_grad_(exp.requires_grad()),
      base_(base), exp_(exp), result_(result, /*is_output=*/true)
{
    if (base_requires_grad_) base_shape_ = DimVector(base.shape());
    if (exp_requires_grad_) exp_shape_ = DimVector(exp.shape());
}

std::string PowBackward1::name() const { return "PowBackward1"; }

TensorList PowBackward1::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_base, grad_exp;

    if (base_requires_grad_) {
        // d/dbase = exp * base^(exp-1) * grad
        grad_base = unbroadcast(grad * exp_ * base_.pow(exp_ - 1.0), base_shape_);
    }
    if (exp_requires_grad_) {
        // d/dexp = result * log(base) * grad
        grad_exp = unbroadcast(grad * result_.unpack() * base_.log(), exp_shape_);
    }

    return {grad_base, grad_exp};
}

PowBackward2::PowBackward2(Scalar base, const Tensor& exp, const Tensor& result)
    : base_(base), result_(result, /*is_output=*/true) {}

std::string PowBackward2::name() const { return "PowBackward2"; }

TensorList PowBackward2::apply(const TensorList& grads) {
    // d/dexp = result * log(base)
    return {grads[0] * result_.unpack() * std::log(base_.to<double>())};
}


MaximumBackward0::MaximumBackward0(const Tensor& a, const Tensor& b)
    : a_requires_grad_(a.requires_grad()),
      b_requires_grad_(b.requires_grad()),
      a_(a), b_(b)
{
    if (a_requires_grad_) a_shape_ = DimVector(a.shape());
    if (b_requires_grad_) b_shape_ = DimVector(b.shape());
}

std::string MaximumBackward0::name() const { return "MaximumBackward0"; }

TensorList MaximumBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor mask = (a_ >= b_).to(grad.dtype());
    Tensor grad_a, grad_b;

    if (a_requires_grad_)
        grad_a = unbroadcast(grad * mask, a_shape_);
    if (b_requires_grad_)
        grad_b = unbroadcast(grad * mask.rsub(1.0), b_shape_);

    return {grad_a, grad_b};
}


MinimumBackward0::MinimumBackward0(const Tensor& a, const Tensor& b)
    : a_requires_grad_(a.requires_grad()),
      b_requires_grad_(b.requires_grad()),
      a_(a), b_(b)
{
    if (a_requires_grad_) a_shape_ = DimVector(a.shape());
    if (b_requires_grad_) b_shape_ = DimVector(b.shape());
}

std::string MinimumBackward0::name() const { return "MinimumBackward0"; }

TensorList MinimumBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor mask = (a_ <= b_).to(grad.dtype());
    Tensor grad_a, grad_b;

    if (a_requires_grad_)
        grad_a = unbroadcast(grad * mask, a_shape_);
    if (b_requires_grad_)
        grad_b = unbroadcast(grad * mask.rsub(1.0), b_shape_);

    return {grad_a, grad_b};
}

} // namespace raxis::autograd