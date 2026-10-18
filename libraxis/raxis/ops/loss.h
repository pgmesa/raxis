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

// Differentiable through a dedicated backward node (NllLossBackward0)
RAXIS_API Tensor nll_loss(
    const Tensor& self,
    const Tensor& target,
    Reduction reduction = Reduction::Mean,
    int64_t ignore_index = -100);

// Composite: differentiable through the ops they are made of
RAXIS_API Tensor mse_loss(const Tensor& self, const Tensor& target, Reduction reduction = Reduction::Mean);
RAXIS_API Tensor cross_entropy(
    const Tensor& self,
    const Tensor& target,
    Reduction reduction = Reduction::Mean,
    int64_t ignore_index = -100);
RAXIS_API Tensor binary_cross_entropy(const Tensor& self, const Tensor& target, Reduction reduction = Reduction::Mean);
RAXIS_API Tensor binary_cross_entropy_with_logits(
    const Tensor& self, const Tensor& target, Reduction reduction = Reduction::Mean);

namespace ops {

struct NllLossResult {
    Tensor output;
    // Number of targets not equal to ignore_index (the mean divisor)
    double total_weight;
};

// self: [N, C] or [C] log-probabilities, target: [N] or [] class indices.
// From: aten/src/ATen/native/LossNLL.cpp
RAXIS_API NllLossResult nll_loss_forward(
    const Tensor& self, const Tensor& target, Reduction reduction, int64_t ignore_index);

RAXIS_API Tensor nll_loss_backward(
    const Tensor& grad, const Tensor& self, const Tensor& target,
    Reduction reduction, int64_t ignore_index, double total_weight);

} // namespace ops

} // namespace raxis
