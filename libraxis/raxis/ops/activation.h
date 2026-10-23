#pragma once

#include <raxis/export.h>
#include <raxis/core/scalar.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/op_macros.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

struct TensorIterator;
using activation_fn = void(*)(TensorIterator&);
using threshold_backward_fn = void(*)(TensorIterator&, Scalar threshold);

DECLARE_DISPATCH(activation_fn, relu_stub);
DECLARE_DISPATCH(threshold_backward_fn, threshold_backward_stub);

RAXIS_DECLARE_UNARY_ARITHMETIC(relu)

// Differentiable through ThresholdBackwardBackward0 (double backward of relu)
RAXIS_API Tensor threshold_backward(const Tensor& grad, const Tensor& self, Scalar threshold);

namespace ops {

// Gradient of relu-like ops: grad where self > threshold, 0 elsewhere.
// From: aten/src/ATen/native/Activation.cpp (threshold_backward)
RAXIS_API Tensor threshold_backward(const Tensor& grad, const Tensor& self, Scalar threshold);

} // namespace ops

} // namespace raxis
