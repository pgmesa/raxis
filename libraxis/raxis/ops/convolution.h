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

// A convolution as the 2-D problem the kernels solve: 1-D inputs get a
// height of 1, input and weight are contiguous.
struct Conv2dProblem {
    Tensor input;    // [N, C, H, W]
    Tensor weight;   // [O, C / groups, kh, kw]
    Unfold2dParams params;
    int64_t groups;
    int64_t batch_size;
    int64_t out_channels;
    bool is_1d;
    // 1x1 kernel, stride 1, no padding: the image already is its columns
    bool is_pointwise;

    int64_t output_plane() const { return params.output_height * params.output_width; }
    int64_t input_numel() const { return params.channels * params.input_height * params.input_width; }
    // Rows of columns (gemm inner dim) per group
    int64_t group_rows() const {
        return params.channels / groups * params.kernel_height * params.kernel_width;
    }
    int64_t group_out_channels() const { return out_channels / groups; }
};

// output: contiguous [N, O, OH, OW]; bias: [O] or undefined
using conv2d_forward_fn = void(*)(const Tensor& output, const Conv2dProblem& problem, const Tensor& bias);
// grad_input: contiguous, shaped as problem.input; grad_output: contiguous [N, O, OH, OW]
using conv2d_backward_input_fn = void(*)(
    const Tensor& grad_input, const Tensor& grad_output, const Conv2dProblem& problem);
// Returns grad_weight flat, [O * C / groups * kh * kw]
using conv2d_backward_weight_fn = Tensor(*)(const Tensor& grad_output, const Conv2dProblem& problem);

DECLARE_DISPATCH(conv2d_forward_fn, conv2d_forward_stub);
DECLARE_DISPATCH(conv2d_backward_input_fn, conv2d_backward_input_stub);
DECLARE_DISPATCH(conv2d_backward_weight_fn, conv2d_backward_weight_stub);

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
