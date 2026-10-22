#pragma once

#include <array>
#include <tuple>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// Kernels take contiguous [N, C, *] tensors, viewed as [N, C, image_size].
// weight, bias, running_mean and running_var may be undefined.

// output = input * alpha[c] + beta[c], with alpha = invstd * weight and
// beta = bias - mean * alpha. The mean and invstd are save_mean and
// save_invstd when training, the running statistics otherwise.
using batch_norm_fn = void(*)(
    const Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps);

// Per channel: mean and sum((x - mean)^2)
using batch_norm_collect_stats_fn = void(*)(
    const Tensor& mean, const Tensor& var_sum, const Tensor& input);

// Undefined grad_* outputs are skipped
using batch_norm_backward_fn = void(*)(
    const Tensor& grad_input, const Tensor& grad_weight, const Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd, bool train, double eps);

DECLARE_DISPATCH(batch_norm_fn, batch_norm_stub);
DECLARE_DISPATCH(batch_norm_collect_stats_fn, batch_norm_collect_stats_stub);
DECLARE_DISPATCH(batch_norm_backward_fn, batch_norm_backward_stub);

// Normalizes each channel (dim 1) of input: [N, C, *]. When training it
// uses the batch statistics and, if given, updates running_mean and
// running_var in place with momentum (running_var with the unbiased
// variance); otherwise it uses the running statistics. Differentiable
// through NativeBatchNormBackward0.
RAXIS_API Tensor batch_norm(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    bool training, double momentum, double eps);

namespace ops {

// Returns (output, save_mean, save_invstd); the last two are empty when not
// training.
// From: aten/src/ATen/native/Normalization.cpp (batch_norm_cpu)
RAXIS_API std::tuple<Tensor, Tensor, Tensor> native_batch_norm(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    bool training, double momentum, double eps);

// From: aten/src/ATen/native/Normalization.cpp (batch_norm_backward_cpu)
RAXIS_API std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd,
    bool training, double eps, std::array<bool, 3> output_mask);

} // namespace ops

} // namespace raxis
