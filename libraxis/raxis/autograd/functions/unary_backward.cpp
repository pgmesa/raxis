#include <raxis/autograd/functions/unary_backward.h>

#include <cmath>

#include <raxis/ops/unary_ops.h>
#include <raxis/tensor/factories.h>


namespace raxis::autograd {

namespace {

constexpr double kLn2 = 0.6931471805599453;
constexpr double kLn10 = 2.302585092994046;
constexpr double kTwoOverSqrtPi = 1.1283791670955126;

} // namespace

std::string NegBackward0::name() const { return "NegBackward0"; }

TensorList NegBackward0::apply(const TensorList& grads) {
    return {-grads[0]};
}

std::string FracBackward0::name() const { return "FracBackward0"; }

TensorList FracBackward0::apply(const TensorList& grads) {
    return {grads[0]};
}


AbsBackward0::AbsBackward0(const Tensor& self) {
    self_.save(self);
}
std::string AbsBackward0::name() const { return "AbsBackward0"; }

// d|x|/dx = sign(x), with 0 at x = 0 (subgradient), as in torch
TensorList AbsBackward0::apply(const TensorList& grads) {
    return {grads[0] * sign(self_.unpack())};
}


std::string SignBackward0::name() const { return "SignBackward0"; }

// sign is piecewise constant
TensorList SignBackward0::apply(const TensorList& grads) {
    return {zeros_like(grads[0])};
}


ExpBackward0::ExpBackward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string ExpBackward0::name() const { return "ExpBackward0"; }

TensorList ExpBackward0::apply(const TensorList& grads) {
    return {grads[0] * result_.unpack()};
}


Expm1Backward0::Expm1Backward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string Expm1Backward0::name() const { return "Expm1Backward0"; }

TensorList Expm1Backward0::apply(const TensorList& grads) {
    return {grads[0] * (result_.unpack() + 1.0)};
}


LogBackward0::LogBackward0(const Tensor& self) {
    self_.save(self);
}
std::string LogBackward0::name() const { return "LogBackward0"; }

TensorList LogBackward0::apply(const TensorList& grads) {
    return {grads[0] / self_.unpack()};
}


Log2Backward0::Log2Backward0(const Tensor& self) {
    self_.save(self);
}
std::string Log2Backward0::name() const { return "Log2Backward0"; }

TensorList Log2Backward0::apply(const TensorList& grads) {
    return {grads[0] / (self_.unpack() * kLn2)};
}


Log10Backward0::Log10Backward0(const Tensor& self) {
    self_.save(self);
}
std::string Log10Backward0::name() const { return "Log10Backward0"; }

TensorList Log10Backward0::apply(const TensorList& grads) {
    return {grads[0] / (self_.unpack() * kLn10)};
}


Log1PBackward0::Log1PBackward0(const Tensor& self) {
    self_.save(self);
}
std::string Log1PBackward0::name() const { return "Log1PBackward0"; }

TensorList Log1PBackward0::apply(const TensorList& grads) {
    return {grads[0] / (self_.unpack() + 1.0)};
}


SqrtBackward0::SqrtBackward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string SqrtBackward0::name() const { return "SqrtBackward0"; }

TensorList SqrtBackward0::apply(const TensorList& grads) {
    return {grads[0] / (result_.unpack() * 2.0)};
}


RsqrtBackward0::RsqrtBackward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string RsqrtBackward0::name() const { return "RsqrtBackward0"; }

TensorList RsqrtBackward0::apply(const TensorList& grads) {
    return {grads[0] * result_.unpack().pow(3) * -0.5};
}


ReciprocalBackward0::ReciprocalBackward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string ReciprocalBackward0::name() const { return "ReciprocalBackward0"; }

TensorList ReciprocalBackward0::apply(const TensorList& grads) {
    return {-(grads[0] * result_.unpack().pow(2))};
}


SigmoidBackward0::SigmoidBackward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string SigmoidBackward0::name() const { return "SigmoidBackward0"; }

TensorList SigmoidBackward0::apply(const TensorList& grads) {
    const Tensor& result = result_.unpack();
    return {grads[0] * result * result.rsub(1.0)};
}


SinBackward0::SinBackward0(const Tensor& self) {
    self_.save(self);
}
std::string SinBackward0::name() const { return "SinBackward0"; }

TensorList SinBackward0::apply(const TensorList& grads) {
    return {grads[0] * self_.unpack().cos()};
}


CosBackward0::CosBackward0(const Tensor& self) {
    self_.save(self);
}
std::string CosBackward0::name() const { return "CosBackward0"; }

TensorList CosBackward0::apply(const TensorList& grads) {
    return {-(grads[0] * self_.unpack().sin())};
}


TanBackward0::TanBackward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string TanBackward0::name() const { return "TanBackward0"; }

TensorList TanBackward0::apply(const TensorList& grads) {
    return {grads[0] * (result_.unpack().pow(2) + 1.0)};
}


AsinBackward0::AsinBackward0(const Tensor& self) {
    self_.save(self);
}
std::string AsinBackward0::name() const { return "AsinBackward0"; }

TensorList AsinBackward0::apply(const TensorList& grads) {
    return {grads[0] / self_.unpack().pow(2).rsub(1.0).sqrt()};
}


AcosBackward0::AcosBackward0(const Tensor& self) {
    self_.save(self);
}
std::string AcosBackward0::name() const { return "AcosBackward0"; }

TensorList AcosBackward0::apply(const TensorList& grads) {
    return {-(grads[0] / self_.unpack().pow(2).rsub(1.0).sqrt())};
}


AtanBackward0::AtanBackward0(const Tensor& self) {
    self_.save(self);
}
std::string AtanBackward0::name() const { return "AtanBackward0"; }

TensorList AtanBackward0::apply(const TensorList& grads) {
    return {grads[0] / (self_.unpack().pow(2) + 1.0)};
}


SinhBackward0::SinhBackward0(const Tensor& self) {
    self_.save(self);
}
std::string SinhBackward0::name() const { return "SinhBackward0"; }

TensorList SinhBackward0::apply(const TensorList& grads) {
    return {grads[0] * self_.unpack().cosh()};
}


CoshBackward0::CoshBackward0(const Tensor& self) {
    self_.save(self);
}
std::string CoshBackward0::name() const { return "CoshBackward0"; }

TensorList CoshBackward0::apply(const TensorList& grads) {
    return {grads[0] * self_.unpack().sinh()};
}


TanhBackward0::TanhBackward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string TanhBackward0::name() const { return "TanhBackward0"; }

TensorList TanhBackward0::apply(const TensorList& grads) {
    return {grads[0] * result_.unpack().pow(2).rsub(1.0)};
}


ErfBackward0::ErfBackward0(const Tensor& self) {
    self_.save(self);
}
std::string ErfBackward0::name() const { return "ErfBackward0"; }

TensorList ErfBackward0::apply(const TensorList& grads) {
    return {grads[0] * (-self_.unpack().pow(2)).exp() * kTwoOverSqrtPi};
}


ErfcBackward0::ErfcBackward0(const Tensor& self) {
    self_.save(self);
}
std::string ErfcBackward0::name() const { return "ErfcBackward0"; }

TensorList ErfcBackward0::apply(const TensorList& grads) {
    return {-(grads[0] * (-self_.unpack().pow(2)).exp() * kTwoOverSqrtPi)};
}

} // namespace raxis::autograd