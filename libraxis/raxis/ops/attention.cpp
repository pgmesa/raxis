#include <raxis/ops/attention.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <raxis/ops/tensor_compare.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

namespace {

// From: aten/src/ATen/native/transformers/sdp_utils_cpp.h (calculate_scale)
double calculate_scale(const Tensor& query, std::optional<double> scale) {
    return scale.has_value() ? *scale : 1.0 / std::sqrt(static_cast<double>(query.shape()[3]));
}

Tensor unit_last_stride(const Tensor& t) {
    return t.stride(-1) == 1 ? t : t.contiguous();
}

// The mask as the kernels take it: additive, in the query dtype, viewed as
// [B or 1, H or 1, L, S] with stride 0 in the broadcast dims.
// From: aten/src/ATen/native/transformers/attention.cpp (convert_boolean_attn_mask)
// and aten/src/ATen/native/cpu/FlashAttentionKernel.cpp (reshape_attn_mask_to_4d)
Tensor attn_mask_to_4d(const Tensor& attn_mask, const Tensor& query, int64_t kv_size) {
    if (!attn_mask.defined() || attn_mask.numel() == 0) {
        return {};
    }
    RAXIS_CHECK(attn_mask.dtype() == dtypes::Bool || attn_mask.dtype() == dtypes::Float32
                || attn_mask.dtype() == query.dtype(),
        "scaled_dot_product_attention_flash_attention: Attention mask is the same data type as query");
    RAXIS_CHECK(attn_mask.ndim() == 2 || attn_mask.ndim() == 4,
        "scaled_dot_product_attention_flash_attention: Attention mask dim in {2, 4}");
    Tensor mask = attn_mask;
    if (mask.dtype() == dtypes::Bool) {
        // true takes part (+0), false is masked out (-inf)
        mask = ops::where(mask, Scalar(0.0), full({}, -std::numeric_limits<double>::infinity(), query.options()));
    } else if (mask.dtype() != query.dtype()) {
        mask = mask.to(query.dtype());
    }
    const int64_t batch_size = query.shape()[0];
    const int64_t num_head = query.shape()[1];
    const int64_t q_size = query.shape()[2];
    int64_t mask_size_0 = 1;
    int64_t mask_size_1 = 1;
    if (mask.ndim() == 4) {
        if (mask.shape()[0] == batch_size) {
            mask_size_0 = batch_size;
        }
        if (mask.shape()[1] == num_head) {
            mask_size_1 = num_head;
        }
    }
    mask = mask.view({mask_size_0, mask_size_1, mask.shape()[mask.ndim() - 2], mask.shape()[mask.ndim() - 1]})
               .expand({mask_size_0, mask_size_1, q_size, kv_size});
    if (mask.stride(-1) != 1 && mask.stride(-1) != 0) {
        mask = mask.contiguous();
    }
    return mask;
}

} // namespace


namespace ops {

std::tuple<Tensor, Tensor> scaled_dot_product_flash_attention_for_cpu(
    const Tensor& query, const Tensor& key, const Tensor& value, double dropout_p, bool is_causal,
    const Tensor& attn_mask, std::optional<double> scale
) {
    const Dtype dtype = query.dtype();
    RAXIS_CHECK(dtype.is_floating(),
        "scaled_dot_product_attention_flash_attention: Expected data type in FP32, FP64, but got ",
        dtype, " instead.");
    RAXIS_CHECK(key.dtype() == dtype && value.dtype() == dtype,
        "scaled_dot_product_attention_flash_attention: Expected query, key and value to have the same dtype");
    RAXIS_CHECK(query.ndim() == 4 && key.ndim() == 4 && value.ndim() == 4,
        "scaled_dot_product_attention_flash_attention: Accept only 4 dims inputs shape of {B, H, T, K}");
    RAXIS_CHECK(dropout_p == 0.0,
        "scaled_dot_product_attention_flash_attention: Currently do not support dropout > 0");
    RAXIS_CHECK(query.shape()[3] == value.shape()[3] && key.shape()[3] == value.shape()[3],
        "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
    RAXIS_CHECK(key.shape()[0] == query.shape()[0] && value.shape()[0] == query.shape()[0]
                && key.shape()[1] == query.shape()[1] && value.shape()[1] == query.shape()[1]
                && value.shape()[2] == key.shape()[2],
        "scaled_dot_product_attention_flash_attention: expected query, key and value with the same "
        "batch size and number of heads, and key and value with the same sequence length");

    const int64_t batch_size = query.shape()[0];
    const int64_t q_size = query.shape()[2];
    const int64_t num_head = query.shape()[1];
    const Tensor mask = attn_mask_to_4d(attn_mask, query, key.shape()[2]);

    // The kernel works on [B, L, H, E]: the output is allocated with the
    // query's layout and transposed, so for the usual (B, L, H, E) storage of
    // q, k and v (a view after splitting the projection into heads) the
    // output is contiguous in that same layout and the reshape back to
    // (B, L, H * E) costs nothing
    Tensor output = empty_like(query).transpose(1, 2);
    Tensor logsumexp = empty({batch_size, q_size, num_head}, query.options());
    if (output.numel() > 0) {
        flash_attention_stub(
            query.device().type(), output, logsumexp, unit_last_stride(query.transpose(1, 2)),
            unit_last_stride(key.transpose(1, 2)), unit_last_stride(value.transpose(1, 2)),
            is_causal, mask, calculate_scale(query, scale));
    }
    return {output.transpose(1, 2), logsumexp.transpose(1, 2)};
}

std::tuple<Tensor, Tensor, Tensor> scaled_dot_product_flash_attention_for_cpu_backward(
    const Tensor& grad_output, const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output, const Tensor& logsumexp, double dropout_p, bool is_causal,
    const Tensor& attn_mask, std::optional<double> scale
) {
    if (!grad_output.defined()) {
        return {Tensor(), Tensor(), Tensor()};
    }
    RAXIS_CHECK(dropout_p == 0.0,
        "scaled_dot_product_attention_flash_attention_backward: Currently do not support dropout > 0");
    const Tensor q_t = unit_last_stride(query.transpose(1, 2));
    const Tensor k_t = unit_last_stride(key.transpose(1, 2));
    const Tensor v_t = unit_last_stride(value.transpose(1, 2));
    Tensor grad_q = zeros(q_t.shape(), query.options());
    Tensor grad_k = zeros(k_t.shape(), key.options());
    Tensor grad_v = zeros(v_t.shape(), value.options());
    if (grad_q.numel() > 0 && grad_k.numel() > 0) {
        // grad_output may come with zero strides (from a sum, a broadcast):
        // a gemm operand must not
        flash_attention_backward_stub(
            query.device().type(), grad_q, grad_k, grad_v, grad_output.transpose(1, 2).contiguous(),
            q_t, k_t, v_t, unit_last_stride(output.transpose(1, 2)), logsumexp.transpose(1, 2),
            is_causal, attn_mask_to_4d(attn_mask, query, key.shape()[2]), calculate_scale(query, scale));
    }
    return {grad_q.transpose(1, 2), grad_k.transpose(1, 2), grad_v.transpose(1, 2)};
}

} // namespace ops

DEFINE_DISPATCH(flash_attention_stub);
DEFINE_DISPATCH(flash_attention_backward_stub);

} // namespace raxis
