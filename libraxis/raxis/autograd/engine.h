#pragma once

#include <stack>

#include <raxis/export.h>
#include <raxis/tensor/tensor.h>
#include <raxis/autograd/graph.h>


namespace raxis::autograd {

RAXIS_API void run_backward(
    const Tensor& tensor,
    const Tensor& grad = {},
    bool retain_graph = false);

} // namespace raxis::autograd