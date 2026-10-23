#pragma once

#include <array>
#include <cstdint>
#include <tuple>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// Geometry of a 2-D convolution over one image (no batch dim). A 1-D
// convolution is the same with height 1.
struct Unfold2dParams {
    int64_t channels;
    int64_t input_height, input_width;
    int64_t output_height, output_width;
    int64_t kernel_height, kernel_width;
    int64_t stride_height, stride_width;
    int64_t pad_height, pad_width;
    int64_t dilation_height, dilation_width;
};

// unfold (im2col): copies every receptive field of a contiguous
// [channels, input_height, input_width] image into the columns of a
// contiguous [channels * kernel_height * kernel_width, output_height * output_width]
// matrix, with zeros where the field overlaps the padding. A convolution is
// then one gemm: weight [out_channels, channels * kh * kw] @ columns.
using unfolded2d_copy_fn = void(*)(
    Dtype dtype, void* columns, const void* input, const Unfold2dParams& params);

// fold (col2im), the adjoint of unfold: adds every column back into the
// image positions it was copied from. Overlapping fields accumulate.
using unfolded2d_acc_fn = void(*)(
    Dtype dtype, const void* columns, void* input, const Unfold2dParams& params);

DECLARE_DISPATCH(unfolded2d_copy_fn, unfolded2d_copy_stub);
DECLARE_DISPATCH(unfolded2d_acc_fn, unfolded2d_acc_stub);

// Differentiable through ConvolutionBackward0. input: [N, C, L] or
// [N, C, H, W]; weight: [O, C / groups, k...]; bias: [O] or undefined.
// stride, padding and dilation hold one value per spatial dim or a single
// value for all of them.
RAXIS_API Tensor convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups);

// Also accept unbatched input ([C, L] and [C, H, W]), as torch does
RAXIS_API Tensor conv1d(
    const Tensor& input, const Tensor& weight, const Tensor& bias = {},
    IntArrayRef stride = {1}, IntArrayRef padding = {0}, IntArrayRef dilation = {1},
    int64_t groups = 1);
RAXIS_API Tensor conv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias = {},
    IntArrayRef stride = {1}, IntArrayRef padding = {0}, IntArrayRef dilation = {1},
    int64_t groups = 1);

// No double backward yet: recorded as NotImplementedBackward
RAXIS_API std::tuple<Tensor, Tensor, Tensor> convolution_backward(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups,
    std::array<bool, 3> output_mask);

namespace ops {

// From: aten/src/ATen/native/Convolution.cpp (_convolution, Slow2d backend)
// and aten/src/ATen/native/ConvolutionMM2d.cpp
RAXIS_API Tensor convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups);

// Gradients wrt (input, weight, bias); output_mask selects which ones are
// computed, the others are undefined.
RAXIS_API std::tuple<Tensor, Tensor, Tensor> convolution_backward(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups,
    std::array<bool, 3> output_mask);

} // namespace ops

} // namespace raxis
