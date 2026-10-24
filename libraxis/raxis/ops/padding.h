#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include <raxis/export.h>
#include <raxis/core/scalar.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// How F.pad fills the new positions: with a value, mirroring the input
// without repeating the edge (reflect), repeating the edge (replicate), or
// wrapping around (circular)
enum class PaddingMode { Constant, Reflect, Replicate, Circular };

// Reflect, replicate and circular kernels. input and output are contiguous
// [planes, D, H, W] (D and H are 1 when fewer dims are padded); padding is
// in F.pad order: {left, right, top, bottom, front, back}.
using padding_fn = void(*)(const Tensor& output, const Tensor& input, IntArrayRef padding, PaddingMode mode);
// grad_input must be zero: every output position adds into the input
// position it was read from
using padding_backward_fn = void(*)(
    const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding, PaddingMode mode);

DECLARE_DISPATCH(padding_fn, padding_stub);
DECLARE_DISPATCH(padding_backward_fn, padding_backward_stub);

namespace ops {

// From: aten/src/ATen/native/PadNd.cpp (constant_pad_nd)
// pad = {left, right} pairs from the last dim backwards; negative values crop
RAXIS_API Tensor constant_pad_nd(const Tensor& self, IntArrayRef pad, Scalar value = 0);

// reflection_pad{1,2,3}d / replication_pad{1,2,3}d / _pad_circular:
// pad.size() / 2 trailing dims are padded, after an optional batch dim and a
// channel dim
RAXIS_API Tensor padding(const Tensor& self, IntArrayRef pad, PaddingMode mode);
RAXIS_API Tensor padding_backward(
    const Tensor& grad_output, IntArrayRef input_shape, IntArrayRef pad, PaddingMode mode);

} // namespace ops

/// Pads self by value, cropping where pad is negative
RAXIS_API Tensor constant_pad_nd(const Tensor& self, IntArrayRef pad, Scalar value = 0);

/// reflect / replicate / circular padding of the pad.size() / 2 trailing dims
RAXIS_API Tensor padding(const Tensor& self, IntArrayRef pad, PaddingMode mode);
RAXIS_API Tensor padding_backward(
    const Tensor& grad_output, IntArrayRef input_shape, IntArrayRef pad, PaddingMode mode);

/// torch.nn.functional.pad: mode is "constant", "reflect", "replicate" or
/// "circular"; pad holds a (left, right) pair per dim, starting from the last
RAXIS_API Tensor pad(
    const Tensor& self, IntArrayRef pad, std::string_view mode = "constant",
    std::optional<double> value = std::nullopt);

} // namespace raxis
