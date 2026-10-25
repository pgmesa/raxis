#pragma once

#include <optional>
#include <tuple>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// query, key and value are [B, L or S, H, E] (heads after the sequence, the
// layout the projections produce), each with unit last stride; output is
// written in that same layout and logsumexp is [B, L, H]. attn_mask is
// undefined or an additive mask of the query dtype, viewed as 4-D
// [B or 1, H or 1, L, S] (broadcast dims with stride 0). scale is resolved.
using flash_attention_fn = void(*)(
    const Tensor& output, const Tensor& logsumexp, const Tensor& query, const Tensor& key,
    const Tensor& value, bool is_causal, const Tensor& attn_mask, double scale);

// Same layout as the forward; grad_query, grad_key and grad_value are zeroed
// and accumulated into
using flash_attention_backward_fn = void(*)(
    const Tensor& grad_query, const Tensor& grad_key, const Tensor& grad_value,
    const Tensor& grad_output, const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output, const Tensor& logsumexp, bool is_causal, const Tensor& attn_mask,
    double scale);

DECLARE_DISPATCH(flash_attention_fn, flash_attention_stub);
DECLARE_DISPATCH(flash_attention_backward_fn, flash_attention_backward_stub);

namespace ops {

// Returns (output, logsumexp): softmax(query @ key^T * scale + mask) @ value
// for query [B, H, L, E], key [B, H, S, E] and value [B, H, S, E], and the
// log of each softmax row's normalizer [B, H, L], kept for the backward.
// attn_mask is undefined, bool (true takes part) or additive, 2-D or 4-D.
// From: aten/src/ATen/native/transformers/attention.cpp (_scaled_dot_product_flash_attention_cpu)
RAXIS_API std::tuple<Tensor, Tensor> scaled_dot_product_flash_attention_for_cpu(
    const Tensor& query, const Tensor& key, const Tensor& value, double dropout_p = 0.0,
    bool is_causal = false, const Tensor& attn_mask = {}, std::optional<double> scale = std::nullopt);

// Returns (grad_query, grad_key, grad_value)
// From: aten/src/ATen/native/transformers/attention.cpp (_scaled_dot_product_flash_attention_cpu_backward)
RAXIS_API std::tuple<Tensor, Tensor, Tensor> scaled_dot_product_flash_attention_for_cpu_backward(
    const Tensor& grad_output, const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output, const Tensor& logsumexp, double dropout_p, bool is_causal,
    const Tensor& attn_mask, std::optional<double> scale);

} // namespace ops

/// Fused attention: the [L, S] scores are never materialized, keys are
/// processed in blocks with an online softmax. Differentiable through
/// ScaledDotProductFlashAttentionForCpuBackward0 (query, key and value; the
/// mask gets no gradient). logsumexp does not require grad.
RAXIS_API std::tuple<Tensor, Tensor> scaled_dot_product_flash_attention_for_cpu(
    const Tensor& query, const Tensor& key, const Tensor& value, double dropout_p = 0.0,
    bool is_causal = false, const Tensor& attn_mask = {}, std::optional<double> scale = std::nullopt);
/// No double backward: recorded as NotImplementedBackward
RAXIS_API std::tuple<Tensor, Tensor, Tensor> scaled_dot_product_flash_attention_for_cpu_backward(
    const Tensor& grad_output, const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output, const Tensor& logsumexp, double dropout_p, bool is_causal,
    const Tensor& attn_mask, std::optional<double> scale);

} // namespace raxis
