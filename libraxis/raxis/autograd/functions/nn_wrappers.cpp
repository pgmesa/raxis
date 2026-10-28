#include <raxis/ops/activation.h>
#include <raxis/ops/attention.h>
#include <raxis/ops/convolution.h>
#include <raxis/ops/dropout.h>
#include <raxis/ops/embedding.h>
#include <raxis/ops/loss.h>
#include <raxis/ops/normalization.h>
#include <raxis/ops/padding.h>
#include <raxis/ops/pooling.h>
#include <raxis/ops/softmax.h>

#include <raxis/tensor/shape_utils.h>

#include <raxis/autograd/autograd_macros.h>
#include <raxis/autograd/functions/common.h>
#include <raxis/autograd/functions/nn_backward.h>


namespace raxis {

RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(relu, ReluBackward0)

Tensor embedding(const Tensor& weight, const Tensor& indices, int64_t padding_idx, bool scale_grad_by_freq) {
    Tensor result = ops::embedding(weight, indices, padding_idx, scale_grad_by_freq);
    if (autograd::requires_autograd(weight)) {
        auto grad_fn = std::make_shared<autograd::EmbeddingBackward0>(
            indices, weight.shape()[0], padding_idx, scale_grad_by_freq);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, weight);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor gelu(const Tensor& self, std::string_view approximate) {
    const GeluType type = gelu_type_from_string(approximate);
    Tensor result = ops::gelu(self, type);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::GeluBackward0>(self, type);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor softmax(const Tensor& self, int64_t dim) {
    Tensor result = ops::softmax(self, dim);
    if (autograd::requires_autograd(self)) {
        int64_t wrapped = wrap_dim(dim, std::max<int64_t>(self.ndim(), 1));
        auto grad_fn = std::make_shared<autograd::SoftmaxBackward0>(result, wrapped);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor log_softmax(const Tensor& self, int64_t dim) {
    Tensor result = ops::log_softmax(self, dim);
    if (autograd::requires_autograd(self)) {
        int64_t wrapped = wrap_dim(dim, std::max<int64_t>(self.ndim(), 1));
        auto grad_fn = std::make_shared<autograd::LogSoftmaxBackward0>(result, wrapped);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor nll_loss(
    const Tensor& self, const Tensor& target, const Tensor& weight, Reduction reduction, int64_t ignore_index
) {
    auto [result, total_weight] = ops::nll_loss_forward(self, target, weight, reduction, ignore_index);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::NllLossBackward0>(
            self, target, weight, reduction, ignore_index, total_weight);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups
) {
    Tensor result = ops::convolution(input, weight, bias, stride, padding, dilation, groups);
    if (autograd::requires_autograd({input, weight, bias})) {
        auto grad_fn = std::make_shared<autograd::ConvolutionBackward0>(
            input, weight, bias, stride, padding, dilation, groups);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, {input, weight, bias});
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

// From: aten/src/ATen/native/Convolution.cpp (conv1d, conv2d): unbatched
// inputs get a batch dim of 1 for the convolution
Tensor conv1d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups
) {
    RAXIS_CHECK(input.ndim() == 2 || input.ndim() == 3,
        "Expected 2D (unbatched) or 3D (batched) input to conv1d, but got input of size: ", input.shape());
    const bool is_batched = input.ndim() == 3;
    Tensor output = convolution(
        is_batched ? input : input.unsqueeze(0), weight, bias, stride, padding, dilation, groups);
    return is_batched ? output : output.squeeze(0);
}

Tensor conv2d(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups
) {
    RAXIS_CHECK(input.ndim() == 3 || input.ndim() == 4,
        "Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: ", input.shape());
    const bool is_batched = input.ndim() == 4;
    Tensor output = convolution(
        is_batched ? input : input.unsqueeze(0), weight, bias, stride, padding, dilation, groups);
    return is_batched ? output : output.squeeze(0);
}

std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode
) {
    auto [result, indices] = ops::max_pool2d_with_indices(
        input, kernel_size, stride, padding, dilation, ceil_mode);
    if (autograd::requires_autograd(input)) {
        auto grad_fn = std::make_shared<autograd::MaxPool2DWithIndicesBackward0>(
            input, kernel_size, stride, padding, dilation, ceil_mode, indices);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, input);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return {result, indices};
}

Tensor max_pool2d(
    const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode
) {
    return std::get<0>(max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode));
}

Tensor avg_pool2d(
    const Tensor& input, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override
) {
    Tensor result = ops::avg_pool2d(
        input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
    if (autograd::requires_autograd(input)) {
        auto grad_fn = std::make_shared<autograd::AvgPool2DBackward0>(
            input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, input);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor batch_norm(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    bool training, double momentum, double eps
) {
    auto [result, save_mean, save_invstd] = ops::native_batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps);
    if (autograd::requires_autograd({input, weight, bias})) {
        auto grad_fn = std::make_shared<autograd::NativeBatchNormBackward0>(
            input, weight, bias, running_mean, running_var, save_mean, save_invstd, training, eps);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, {input, weight, bias});
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor layer_norm(
    const Tensor& input, IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    double eps
) {
    auto [result, mean, rstd] = ops::native_layer_norm(input, normalized_shape, weight, bias, eps);
    if (autograd::requires_autograd({input, weight, bias})) {
        auto grad_fn = std::make_shared<autograd::NativeLayerNormBackward0>(
            input, normalized_shape, weight, bias, mean, rstd);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, {input, weight, bias});
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor constant_pad_nd(const Tensor& self, IntArrayRef pad, Scalar value) {
    Tensor result = ops::constant_pad_nd(self, pad, value);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ConstantPadNdBackward0>(pad);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor padding(const Tensor& self, IntArrayRef pad, PaddingMode mode) {
    Tensor result = ops::padding(self, pad, mode);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::PaddingBackward0>(self, pad, mode);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

// From: aten/src/ATen/native/PadNd.cpp (pad_symint, _pad_enum_symint)
Tensor pad(const Tensor& self, IntArrayRef pad, std::string_view mode, std::optional<double> value) {
    PaddingMode padding_mode;
    if (mode == "constant") {
        padding_mode = PaddingMode::Constant;
    } else if (mode == "reflect") {
        padding_mode = PaddingMode::Reflect;
    } else if (mode == "replicate") {
        padding_mode = PaddingMode::Replicate;
    } else if (mode == "circular") {
        padding_mode = PaddingMode::Circular;
    } else {
        RAXIS_CHECK_NOT_IMPLEMENTED(false, "Unrecognised padding mode ", mode);
    }
    RAXIS_CHECK(pad.size() % 2 == 0, "Padding length must be divisible by 2");
    RAXIS_CHECK(static_cast<int64_t>(pad.size()) <= self.ndim() * 2,
        "Padding length should be less than or equal to two times the input dimension but got padding "
        "length ", pad.size(), " and input of dimension ", self.ndim());
    if (padding_mode == PaddingMode::Constant) {
        return constant_pad_nd(self, pad, value.value_or(0.0));
    }
    RAXIS_CHECK(!value.has_value() || *value == 0,
        "Padding mode \"", mode, "\" doesn't take in value argument");

    const int64_t npad = static_cast<int64_t>(pad.size()) / 2;
    const int64_t ndim = self.ndim();
    RAXIS_CHECK_NOT_IMPLEMENTED(npad >= 1 && npad <= 3 && (ndim == npad + 1 || ndim == npad + 2),
        "Padding size ", pad.size(), " is not supported for ", ndim, "D input tensor.\n"
        "Supported combinations for non-constant padding:\n"
        "  - 2D or 3D input: padding size = 2 (pads last dimension)\n"
        "  - 3D or 4D input: padding size = 4 (pads last 2 dimensions)\n"
        "  - 4D or 5D input: padding size = 6 (pads last 3 dimensions)");
    return padding(self, pad, padding_mode);
}

std::tuple<Tensor, Tensor> scaled_dot_product_flash_attention_for_cpu(
    const Tensor& query, const Tensor& key, const Tensor& value, double dropout_p, bool is_causal,
    const Tensor& attn_mask, std::optional<double> scale
) {
    auto [output, logsumexp] = ops::scaled_dot_product_flash_attention_for_cpu(
        query, key, value, dropout_p, is_causal, attn_mask, scale);
    if (autograd::requires_autograd({query, key, value})) {
        auto grad_fn = std::make_shared<autograd::ScaledDotProductFlashAttentionForCpuBackward0>(
            query, key, value, attn_mask, is_causal, scale, output, logsumexp);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, {query, key, value});
        autograd::set_autograd_meta(output, grad_fn);
    }
    return {output, logsumexp};
}

std::tuple<Tensor, Tensor, Tensor, Tensor> scaled_dot_product_efficient_attention(
    const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& attn_bias,
    bool compute_log_sumexp, double dropout_p, bool is_causal, std::optional<double> scale
) {
    auto [output, log_sumexp, philox_seed, philox_offset] = ops::scaled_dot_product_efficient_attention(
        query, key, value, attn_bias, compute_log_sumexp, dropout_p, is_causal, scale);
    if (autograd::requires_autograd({query, key, value})) {
        auto grad_fn = std::make_shared<autograd::ScaledDotProductEfficientAttentionBackward0>(
            query, key, value, attn_bias, dropout_p, is_causal, scale, output, log_sumexp, philox_seed,
            philox_offset);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, {query, key, value});
        autograd::set_autograd_meta(output, grad_fn);
    }
    return {output, log_sumexp, philox_seed, philox_offset};
}

// From: aten/src/ATen/native/Dropout.cpp (dropout, _dropout_impl)
Tensor dropout(const Tensor& input, double p, bool train) {
    RAXIS_CHECK(p >= 0.0 && p <= 1.0,
        "dropout probability has to be between 0 and 1, but got ", p);
    if (!train || p == 0.0 || input.numel() == 0) {
        return input;
    }
    auto [result, mask] = ops::native_dropout(input, p, true);
    if (autograd::requires_autograd(input)) {
        const double scale = p == 1.0 ? 0.0 : 1.0 / (1.0 - p);
        auto grad_fn = std::make_shared<autograd::NativeDropoutBackward0>(mask, scale);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, input);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}


// ============================================================
// Backward kernels (recorded with create_graph)
// ============================================================

Tensor threshold_backward(const Tensor& grad, const Tensor& self, Scalar threshold) {
    Tensor result = ops::threshold_backward(grad, self, threshold);
    if (autograd::requires_autograd(grad, self)) {
        auto grad_fn = std::make_shared<autograd::ThresholdBackwardBackward0>(grad, self, threshold);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor embedding_dense_backward(
    const Tensor& grad, const Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq
) {
    Tensor result = ops::embedding_dense_backward(grad, indices, num_weights, padding_idx, scale_grad_by_freq);
    if (autograd::requires_autograd(grad)) {
        auto grad_fn = std::make_shared<autograd::EmbeddingDenseBackwardBackward0>(indices, padding_idx);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor gelu_backward(const Tensor& grad, const Tensor& self, std::string_view approximate) {
    const GeluType type = gelu_type_from_string(approximate);
    Tensor result = ops::gelu_backward(grad, self, type);
    if (autograd::requires_autograd(grad, self)) {
        auto grad_fn = std::make_shared<autograd::GeluBackwardBackward0>(grad, self, type);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor softmax_backward_data(const Tensor& grad, const Tensor& output, int64_t dim) {
    Tensor result = ops::softmax_backward_data(grad, output, dim);
    if (autograd::requires_autograd(grad, output)) {
        auto grad_fn = std::make_shared<autograd::SoftmaxBackwardDataBackward0>(grad, output, dim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad, output);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor log_softmax_backward_data(const Tensor& grad, const Tensor& output, int64_t dim) {
    Tensor result = ops::log_softmax_backward_data(grad, output, dim);
    if (autograd::requires_autograd(grad, output)) {
        auto grad_fn = std::make_shared<autograd::LogSoftmaxBackwardDataBackward0>(grad, output, dim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad, output);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor native_dropout_backward(const Tensor& grad, const Tensor& mask, double scale) {
    Tensor result = ops::native_dropout_backward(grad, mask, scale);
    if (autograd::requires_autograd(grad)) {
        auto grad_fn = std::make_shared<autograd::NativeDropoutBackwardBackward0>(mask, scale);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor nll_loss_backward(
    const Tensor& grad, const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index, const Tensor& total_weight
) {
    Tensor result = ops::nll_loss_backward(grad, self, target, weight, reduction, ignore_index, total_weight);
    if (autograd::requires_autograd(grad, self)) {
        auto grad_fn = std::make_shared<autograd::NllLossBackwardBackward0>(
            self, target, weight, reduction, ignore_index);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor max_pool2d_with_indices_backward(
    const Tensor& grad_output, IntArrayRef input_shape, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& indices
) {
    Tensor result = ops::max_pool2d_with_indices_backward(
        grad_output, input_shape, kernel_size, stride, padding, dilation, ceil_mode, indices);
    if (autograd::requires_autograd(grad_output)) {
        auto grad_fn = std::make_shared<autograd::MaxPool2DWithIndicesBackwardBackward0>(indices);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad_output);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor avg_pool2d_backward(
    const Tensor& grad_output, IntArrayRef input_shape, IntArrayRef kernel_size,
    IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override
) {
    Tensor result = ops::avg_pool2d_backward(grad_output, input_shape, kernel_size, stride, padding,
                                             ceil_mode, count_include_pad, divisor_override);
    if (autograd::requires_autograd(grad_output)) {
        auto grad_fn = std::make_shared<autograd::AvgPool2DBackwardBackward0>(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad_output);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor padding_backward(
    const Tensor& grad_output, IntArrayRef input_shape, IntArrayRef pad, PaddingMode mode
) {
    Tensor result = ops::padding_backward(grad_output, input_shape, pad, mode);
    if (autograd::requires_autograd(grad_output)) {
        auto grad_fn = std::make_shared<autograd::PaddingBackwardBackward0>(pad, mode);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, grad_output);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

// The outputs of a backward kernel without a double backward: a second
// backward through them raises, instead of treating them as constants
static void set_not_implemented(std::initializer_list<Tensor*> outputs, ArrayRef<Tensor> inputs,
                                const char* op_name) {
    auto grad_fn = std::make_shared<autograd::NotImplementedBackward>(op_name);
    grad_fn->set_num_inputs(outputs.size());
    autograd::set_next_edges(grad_fn, inputs);
    uint32_t output_nr = 0;
    for (Tensor* output : outputs) {
        if (output->defined()) {
            autograd::set_autograd_meta(*output, grad_fn, output_nr);
        }
        output_nr++;
    }
}

std::tuple<Tensor, Tensor, Tensor> convolution_backward(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups,
    std::array<bool, 3> output_mask
) {
    auto [grad_input, grad_weight, grad_bias] = ops::convolution_backward(
        grad_output, input, weight, stride, padding, dilation, groups, output_mask);
    if (autograd::requires_autograd({grad_output, input, weight})) {
        set_not_implemented({&grad_input, &grad_weight, &grad_bias}, {grad_output, input, weight},
                            "convolution_backward");
    }
    return {grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd,
    bool training, double eps, std::array<bool, 3> output_mask
) {
    auto [grad_input, grad_weight, grad_bias] = ops::native_batch_norm_backward(
        grad_output, input, weight, running_mean, running_var, save_mean, save_invstd,
        training, eps, output_mask);
    if (autograd::requires_autograd({grad_output, input, weight})) {
        set_not_implemented({&grad_input, &grad_weight, &grad_bias}, {grad_output, input, weight},
                            "native_batch_norm_backward");
    }
    return {grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> native_layer_norm_backward(
    const Tensor& grad_output, const Tensor& input, IntArrayRef normalized_shape,
    const Tensor& mean, const Tensor& rstd, const Tensor& weight, const Tensor& bias,
    std::array<bool, 3> output_mask
) {
    auto [grad_input, grad_weight, grad_bias] = ops::native_layer_norm_backward(
        grad_output, input, normalized_shape, mean, rstd, weight, bias, output_mask);
    if (autograd::requires_autograd({grad_output, input, weight})) {
        set_not_implemented({&grad_input, &grad_weight, &grad_bias}, {grad_output, input, weight},
                            "native_layer_norm_backward");
    }
    return {grad_input, grad_weight, grad_bias};
}

std::tuple<Tensor, Tensor, Tensor> scaled_dot_product_flash_attention_for_cpu_backward(
    const Tensor& grad_output, const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output, const Tensor& logsumexp, double dropout_p, bool is_causal,
    const Tensor& attn_mask, std::optional<double> scale
) {
    auto [grad_query, grad_key, grad_value] = ops::scaled_dot_product_flash_attention_for_cpu_backward(
        grad_output, query, key, value, output, logsumexp, dropout_p, is_causal, attn_mask, scale);
    if (autograd::requires_autograd({grad_output, query, key, value})) {
        set_not_implemented({&grad_query, &grad_key, &grad_value}, {grad_output, query, key, value},
                            "_scaled_dot_product_flash_attention_for_cpu_backward");
    }
    return {grad_query, grad_key, grad_value};
}

std::tuple<Tensor, Tensor, Tensor> scaled_dot_product_efficient_attention_backward(
    const Tensor& grad_output, const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& attn_bias, const Tensor& output, const Tensor& logsumexp, const Tensor& philox_seed,
    const Tensor& philox_offset, double dropout_p, bool is_causal, std::optional<double> scale
) {
    auto [grad_query, grad_key, grad_value] = ops::scaled_dot_product_efficient_attention_backward(
        grad_output, query, key, value, attn_bias, output, logsumexp, philox_seed, philox_offset, dropout_p,
        is_causal, scale);
    if (autograd::requires_autograd({grad_output, query, key, value})) {
        set_not_implemented({&grad_query, &grad_key, &grad_value}, {grad_output, query, key, value},
                            "_scaled_dot_product_efficient_attention_backward");
    }
    return {grad_query, grad_key, grad_value};
}

} // namespace raxis
