#pragma once

#include <cstdint>
#include <optional>
#include <tuple>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// Geometry of a 2-D pooling over one plane
struct Pool2dParams {
    int64_t input_height, input_width;
    int64_t output_height, output_width;
    int64_t kernel_height, kernel_width;
    int64_t stride_height, stride_width;
    int64_t pad_height, pad_width;
    int64_t dilation_height, dilation_width;
};

// Kernels take contiguous tensors whose last two dims are the planes and
// treat every leading dim (N, C) as one more plane. indices hold, for each
// output element, the flat position of the max inside its input plane.
using max_pool2d_fn = void(*)(
    const Tensor& output, const Tensor& indices, const Tensor& input, const Pool2dParams& params);
using max_pool2d_backward_fn = void(*)(
    const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices);
using avg_pool2d_fn = void(*)(
    const Tensor& output, const Tensor& input, const Pool2dParams& params,
    bool count_include_pad, std::optional<int64_t> divisor_override);
using avg_pool2d_backward_fn = void(*)(
    const Tensor& grad_input, const Tensor& grad_output, const Pool2dParams& params,
    bool count_include_pad, std::optional<int64_t> divisor_override);

DECLARE_DISPATCH(max_pool2d_fn, max_pool2d_stub);
DECLARE_DISPATCH(max_pool2d_backward_fn, max_pool2d_backward_stub);
DECLARE_DISPATCH(avg_pool2d_fn, avg_pool2d_stub);
DECLARE_DISPATCH(avg_pool2d_backward_fn, avg_pool2d_backward_stub);

// input: [N, C, H, W] or [C, H, W]. An empty stride means stride = kernel_size.
// Differentiable through MaxPool2DWithIndicesBackward0 and AvgPool2DBackward0.
RAXIS_API Tensor max_pool2d(
    const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride = {},
    IntArrayRef padding = {0}, IntArrayRef dilation = {1}, bool ceil_mode = false);
// Also returns the int64 indices of the maxima (return_indices=True in torch)
RAXIS_API std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride = {},
    IntArrayRef padding = {0}, IntArrayRef dilation = {1}, bool ceil_mode = false);
RAXIS_API Tensor avg_pool2d(
    const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride = {},
    IntArrayRef padding = {0}, bool ceil_mode = false, bool count_include_pad = true,
    std::optional<int64_t> divisor_override = std::nullopt);

namespace ops {

// From: aten/src/ATen/native/DilatedMaxPool2d.cpp
RAXIS_API std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode);

// grad_input has the shape of the forward input: each output gradient goes
// to the position of its max
RAXIS_API Tensor max_pool2d_with_indices_backward(
    const Tensor& grad_output, IntArrayRef input_shape, const Tensor& indices);

// From: aten/src/ATen/native/AveragePool2d.cpp
RAXIS_API Tensor avg_pool2d(
    const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override);

RAXIS_API Tensor avg_pool2d_backward(
    const Tensor& grad_output, IntArrayRef input_shape, IntArrayRef kernel_size,
    IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override);

} // namespace ops

} // namespace raxis
