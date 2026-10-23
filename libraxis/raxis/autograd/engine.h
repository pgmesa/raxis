#pragma once

#include <optional>

#include <raxis/export.h>
#include <raxis/tensor/tensor.h>
#include <raxis/autograd/graph.h>


namespace raxis::autograd {

// From: torch/csrc/autograd/autograd.h (backward)
// Accumulates in .grad of the leaves the gradient of `tensors` (weighted by
// grad_tensors: one per tensor, undefined means ones for a scalar). The
// graph is freed unless retain_graph, which defaults to create_graph. With
// create_graph the backward itself is recorded, so the gradients can be
// differentiated again.
RAXIS_API void backward(
    ArrayRef<Tensor> tensors,
    ArrayRef<Tensor> grad_tensors = {},
    std::optional<bool> retain_graph = std::nullopt,
    bool create_graph = false);

// From: torch/csrc/autograd/autograd.h (grad)
// Returns the gradients of `outputs` with respect to `inputs` instead of
// accumulating them in .grad. Only the part of the graph between them runs.
// An input the outputs do not depend on raises, or gets an undefined tensor
// with allow_unused.
RAXIS_API TensorList grad(
    ArrayRef<Tensor> outputs,
    ArrayRef<Tensor> inputs,
    ArrayRef<Tensor> grad_outputs = {},
    std::optional<bool> retain_graph = std::nullopt,
    bool create_graph = false,
    bool allow_unused = false);

// Tensor::backward
RAXIS_API void run_backward(
    const Tensor& tensor,
    const Tensor& grad = {},
    bool retain_graph = false,
    bool create_graph = false);

} // namespace raxis::autograd
