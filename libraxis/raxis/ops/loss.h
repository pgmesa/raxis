#pragma once

#include <cstdint>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// From: aten/src/ATen/core/Reduction.h
enum class Reduction : int64_t {
    None,
    Mean,
    Sum,
};

RAXIS_API Reduction parse_reduction(const std::string& reduction);

// Differentiable through a dedicated backward node (NllLossBackward0).
// weight (optional, one per class) scales each sample's loss by the weight
// of its target class; the mean then divides by the sum of those weights.
RAXIS_API Tensor nll_loss(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight = {},
    Reduction reduction = Reduction::Mean,
    int64_t ignore_index = -100);

// Composite: differentiable through the ops they are made of
RAXIS_API Tensor mse_loss(const Tensor& self, const Tensor& target, Reduction reduction = Reduction::Mean);
RAXIS_API Tensor cross_entropy(
    const Tensor& self,
    const Tensor& target,
    const Tensor& weight = {},
    Reduction reduction = Reduction::Mean,
    int64_t ignore_index = -100);
RAXIS_API Tensor binary_cross_entropy(const Tensor& self, const Tensor& target, Reduction reduction = Reduction::Mean);
RAXIS_API Tensor binary_cross_entropy_with_logits(
    const Tensor& self, const Tensor& target, Reduction reduction = Reduction::Mean);

// The backward kernel, differentiable in grad (NllLossBackwardBackward0)
RAXIS_API Tensor nll_loss_backward(
    const Tensor& grad, const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index, const Tensor& total_weight);

namespace ops {

struct NllLossResult {
    Tensor output;
    // Sum of the weights of the targets not equal to ignore_index (their
    // count without weight): the mean divisor. A 0-d tensor on self's
    // device, as in PyTorch, so a GPU forward never waits to read it.
    Tensor total_weight;
};

// self: [N, C] or [C] log-probabilities, target: [N] or [] class indices,
// weight: [C] or undefined.
// From: aten/src/ATen/native/LossNLL.cpp
RAXIS_API NllLossResult nll_loss_forward(
    const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index);

RAXIS_API Tensor nll_loss_backward(
    const Tensor& grad, const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index, const Tensor& total_weight);

} // namespace ops

// The kernels, on contiguous tensors: output is [N] for reduction None on
// a 2-D input, else 0-d; grad_input comes zeroed.
using nll_loss_forward_fn = void(*)(
    const Tensor& output, const Tensor& total_weight, const Tensor& self, const Tensor& target,
    const Tensor& weight, Reduction reduction, int64_t ignore_index);
using nll_loss_backward_fn = void(*)(
    const Tensor& grad_input, const Tensor& grad, const Tensor& target, const Tensor& weight,
    const Tensor& total_weight, Reduction reduction, int64_t ignore_index);
DECLARE_DISPATCH(nll_loss_forward_fn, nll_loss_forward_stub);
DECLARE_DISPATCH(nll_loss_backward_fn, nll_loss_backward_stub);

} // namespace raxis
