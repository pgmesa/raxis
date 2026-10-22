#pragma once

#include <cstdint>
#include <tuple>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// Elements are drawn in batches of kDropoutBatch from kDropoutLanes Philox
// blocks: element j of batch b is word j / kDropoutLanes of block
// offset + b * kDropoutLanes + j % kDropoutLanes (philox4x32_10_lanes order).
inline constexpr int kDropoutLanes = 16;
inline constexpr int64_t kDropoutBatch = 4 * kDropoutLanes;

// Philox blocks to reserve from the generator for n elements
inline uint64_t dropout_philox_blocks(int64_t n) {
    return static_cast<uint64_t>((n + kDropoutBatch - 1) / kDropoutBatch * kDropoutLanes);
}

// One pass over a contiguous input: draws a Philox number per element from
// the blocks [offset, offset + dropout_philox_blocks(n)) of the stream keyed
// by seed, keeps the element with probability p_keep and writes mask (bool)
// and output = input * mask / p_keep.
using fused_dropout_fn = void(*)(
    const Tensor& output, const Tensor& mask, const Tensor& input,
    double p_keep, uint64_t seed, uint64_t offset);

// grad_input = grad * mask * scale, all contiguous
using dropout_backward_fn = void(*)(
    const Tensor& grad_input, const Tensor& grad, const Tensor& mask, double scale);

DECLARE_DISPATCH(fused_dropout_fn, fused_dropout_stub);
DECLARE_DISPATCH(dropout_backward_fn, dropout_backward_stub);

// Zeroes each element with probability p and scales the rest by 1 / (1 - p)
// so the expected value is unchanged. Identity when !train or p == 0.
// Differentiable through NativeDropoutBackward0.
RAXIS_API Tensor dropout(const Tensor& input, double p, bool train);

namespace ops {

// Returns (output, mask). Not training: (input, all-true mask).
// From: aten/src/ATen/native/Dropout.cpp (native_dropout_cpu)
RAXIS_API std::tuple<Tensor, Tensor> native_dropout(const Tensor& input, double p, bool train);

// From: aten/src/ATen/native/Dropout.cpp (native_dropout_backward)
RAXIS_API Tensor native_dropout_backward(const Tensor& grad, const Tensor& mask, double scale);

} // namespace ops

} // namespace raxis
