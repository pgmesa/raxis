#pragma once

#include <cstdint>

#include <raxis/export.h>
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
    Reduction reduction, int64_t ignore_index, double total_weight);

namespace ops {

struct NllLossResult {
    Tensor output;
    // Sum of the weights of the targets not equal to ignore_index (their
    // count without weight): the mean divisor. Holds a scalar_t value.
    double total_weight;
};

// self: [N, C] or [C] log-probabilities, target: [N] or [] class indices,
// weight: [C] or undefined.
// From: aten/src/ATen/native/LossNLL.cpp
RAXIS_API NllLossResult nll_loss_forward(
    const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index);

RAXIS_API Tensor nll_loss_backward(
    const Tensor& grad, const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index, double total_weight);

} // namespace ops

} // namespace raxis
