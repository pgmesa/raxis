#pragma once

#include <numbers>
#include <string_view>

#include <raxis/export.h>
#include <raxis/core/scalar.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/op_macros.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

struct TensorIterator;

// gelu(x) = x * Phi(x), Phi the standard normal CDF, computed exactly with
// erf (None) or with the tanh approximation of GPT-2 and BERT (Tanh)
enum class GeluType { None, Tanh };

// From: aten/src/ATen/native/cpu/Gelu.h and Activation.cpp. Spelled as the
// M_* products of PyTorch, so the constants round the same way
// sqrt(2 / pi), the scale inside the tanh approximation
inline constexpr double kGeluBeta = std::numbers::sqrt2 * (2 * std::numbers::inv_sqrtpi) * 0.5;
inline constexpr double kGeluKappa = 0.044715;
// 1 / sqrt(2), so that Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
inline constexpr double kGeluAlpha = std::numbers::sqrt2 / 2;
// 1 / sqrt(2 pi), the scale of the normal PDF
inline constexpr double kGeluPdfScale = (2 * std::numbers::inv_sqrtpi) * (std::numbers::sqrt2 / 2) * 0.5;

using activation_fn = void(*)(TensorIterator&);
using threshold_backward_fn = void(*)(TensorIterator&, Scalar threshold);
using gelu_fn = void(*)(TensorIterator&, GeluType);

DECLARE_DISPATCH(activation_fn, relu_stub);
DECLARE_DISPATCH(threshold_backward_fn, threshold_backward_stub);
DECLARE_DISPATCH(gelu_fn, gelu_stub);
// Inputs (grad, self)
DECLARE_DISPATCH(gelu_fn, gelu_backward_stub);

// "none" or "tanh", as torch's approximate argument
RAXIS_API GeluType gelu_type_from_string(std::string_view approximate);

RAXIS_DECLARE_UNARY_ARITHMETIC(relu)

// Differentiable through ThresholdBackwardBackward0 (double backward of relu)
RAXIS_API Tensor threshold_backward(const Tensor& grad, const Tensor& self, Scalar threshold);

RAXIS_API Tensor gelu(const Tensor& self, std::string_view approximate = "none");
// Differentiable through GeluBackwardBackward0
RAXIS_API Tensor gelu_backward(const Tensor& grad, const Tensor& self, std::string_view approximate = "none");

namespace ops {

// Gradient of relu-like ops: grad where self > threshold, 0 elsewhere.
// From: aten/src/ATen/native/Activation.cpp (threshold_backward)
RAXIS_API Tensor threshold_backward(const Tensor& grad, const Tensor& self, Scalar threshold);

// From: aten/src/ATen/native/Activation.cpp (gelu_out_cpu, gelu_backward_out_cpu)
RAXIS_API Tensor gelu(const Tensor& self, GeluType approximate);
RAXIS_API Tensor gelu_backward(const Tensor& grad, const Tensor& self, GeluType approximate);

} // namespace ops

} // namespace raxis
