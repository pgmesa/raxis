#include <raxis/autograd/functions/nn_backward.h>

#include <raxis/ops/activation.h>
#include <raxis/ops/attention.h>
#include <raxis/ops/convolution.h>
#include <raxis/ops/dropout.h>
#include <raxis/ops/embedding.h>
#include <raxis/ops/normalization.h>
#include <raxis/ops/pooling.h>
#include <raxis/ops/softmax.h>
#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>


namespace raxis::autograd {

ReluBackward0::ReluBackward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string ReluBackward0::name() const { return "ReluBackward0"; }

// relu'(x) is 1 where x > 0, which is the same set as result > 0
TensorList ReluBackward0::apply(const TensorList& grads) {
    return {threshold_backward(grads[0], result_.unpack(), 0)};
}


GeluBackward0::GeluBackward0(const Tensor& self, GeluType approximate) : approximate_(approximate) {
    self_.save(self);
}
std::string GeluBackward0::name() const { return "GeluBackward0"; }

TensorList GeluBackward0::apply(const TensorList& grads) {
    return {gelu_backward(grads[0], self_.unpack(), approximate_ == GeluType::Tanh ? "tanh" : "none")};
}


SoftmaxBackward0::SoftmaxBackward0(const Tensor& result, int64_t dim)
    : dim_(dim)
{
    result_.save(result, /*is_output=*/true);
}
std::string SoftmaxBackward0::name() const { return "SoftmaxBackward0"; }

TensorList SoftmaxBackward0::apply(const TensorList& grads) {
    return {softmax_backward_data(grads[0], result_.unpack(), dim_)};
}


LogSoftmaxBackward0::LogSoftmaxBackward0(const Tensor& result, int64_t dim)
    : dim_(dim)
{
    result_.save(result, /*is_output=*/true);
}
std::string LogSoftmaxBackward0::name() const { return "LogSoftmaxBackward0"; }

TensorList LogSoftmaxBackward0::apply(const TensorList& grads) {
    return {log_softmax_backward_data(grads[0], result_.unpack(), dim_)};
}


NllLossBackward0::NllLossBackward0(
    const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index, const Tensor& total_weight)
    : reduction_(reduction), ignore_index_(ignore_index)
{
    self_.save(self);
    target_.save(target);
    weight_.save(weight);
    total_weight_.save(total_weight);
}
std::string NllLossBackward0::name() const { return "NllLossBackward0"; }

// Only self gets a gradient: target holds class indices
TensorList NllLossBackward0::apply(const TensorList& grads) {
    return {nll_loss_backward(
        grads[0], self_.unpack(), target_.unpack(), weight_.unpack(),
        reduction_, ignore_index_, total_weight_.unpack())};
}


ConvolutionBackward0::ConvolutionBackward0(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups)
    : stride_(stride.begin(), stride.end()),
      padding_(padding.begin(), padding.end()),
      dilation_(dilation.begin(), dilation.end()),
      groups_(groups),
      input_requires_grad_(input.requires_grad()),
      weight_requires_grad_(weight.requires_grad()),
      bias_requires_grad_(bias.requires_grad())
{
    input_.save(input);
    weight_.save(weight);
}
std::string ConvolutionBackward0::name() const { return "ConvolutionBackward0"; }

TensorList ConvolutionBackward0::apply(const TensorList& grads) {
    auto [grad_input, grad_weight, grad_bias] = convolution_backward(
        grads[0], input_.unpack(), weight_.unpack(), stride_, padding_, dilation_, groups_,
        {input_requires_grad_, weight_requires_grad_, bias_requires_grad_});
    return {grad_input, grad_weight, grad_bias};
}


MaxPool2DWithIndicesBackward0::MaxPool2DWithIndicesBackward0(
    const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding,
    IntArrayRef dilation, bool ceil_mode, const Tensor& indices)
    : self_shape_(self.shape().begin(), self.shape().end()),
      kernel_size_(kernel_size.begin(), kernel_size.end()),
      stride_(stride.begin(), stride.end()),
      padding_(padding.begin(), padding.end()),
      dilation_(dilation.begin(), dilation.end()),
      ceil_mode_(ceil_mode)
{
    result1_.save(indices);
}
std::string MaxPool2DWithIndicesBackward0::name() const { return "MaxPool2DWithIndicesBackward0"; }

TensorList MaxPool2DWithIndicesBackward0::apply(const TensorList& grads) {
    return {max_pool2d_with_indices_backward(
        grads[0], self_shape_, kernel_size_, stride_, padding_, dilation_, ceil_mode_, result1_.unpack())};
}


AvgPool2DBackward0::AvgPool2DBackward0(
    const Tensor& self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding,
    bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override)
    : self_shape_(self.shape().begin(), self.shape().end()),
      kernel_size_(kernel_size.begin(), kernel_size.end()),
      stride_(stride.begin(), stride.end()),
      padding_(padding.begin(), padding.end()),
      ceil_mode_(ceil_mode),
      count_include_pad_(count_include_pad),
      divisor_override_(divisor_override)
{}
std::string AvgPool2DBackward0::name() const { return "AvgPool2DBackward0"; }

TensorList AvgPool2DBackward0::apply(const TensorList& grads) {
    return {avg_pool2d_backward(
        grads[0], self_shape_, kernel_size_, stride_, padding_,
        ceil_mode_, count_include_pad_, divisor_override_)};
}


NativeBatchNormBackward0::NativeBatchNormBackward0(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd, bool training, double eps)
    : training_(training),
      eps_(eps),
      input_requires_grad_(input.requires_grad()),
      weight_requires_grad_(weight.requires_grad()),
      bias_requires_grad_(bias.requires_grad())
{
    input_.save(input);
    weight_.save(weight);
    running_mean_.save(running_mean);
    running_var_.save(running_var);
    result1_.save(save_mean);
    result2_.save(save_invstd);
}
std::string NativeBatchNormBackward0::name() const { return "NativeBatchNormBackward0"; }

TensorList NativeBatchNormBackward0::apply(const TensorList& grads) {
    auto [grad_input, grad_weight, grad_bias] = native_batch_norm_backward(
        grads[0], input_.unpack(), weight_.unpack(), running_mean_.unpack(), running_var_.unpack(),
        result1_.unpack(), result2_.unpack(), training_, eps_,
        {input_requires_grad_, weight_requires_grad_, bias_requires_grad_});
    return {grad_input, grad_weight, grad_bias};
}


// bias is saved only for the shape check of the backward, as in PyTorch
NativeLayerNormBackward0::NativeLayerNormBackward0(
    const Tensor& input, IntArrayRef normalized_shape, const Tensor& weight, const Tensor& bias,
    const Tensor& mean, const Tensor& rstd)
    : normalized_shape_(normalized_shape.begin(), normalized_shape.end()),
      input_requires_grad_(input.requires_grad()),
      weight_requires_grad_(weight.requires_grad()),
      bias_requires_grad_(bias.requires_grad())
{
    input_.save(input);
    weight_.save(weight);
    bias_.save(bias);
    result1_.save(mean);
    result2_.save(rstd);
}
std::string NativeLayerNormBackward0::name() const { return "NativeLayerNormBackward0"; }

TensorList NativeLayerNormBackward0::apply(const TensorList& grads) {
    if (!grads[0].defined()) {
        return {Tensor(), Tensor(), Tensor()};
    }
    auto [grad_input, grad_weight, grad_bias] = native_layer_norm_backward(
        grads[0], input_.unpack(), normalized_shape_, result1_.unpack(), result2_.unpack(),
        weight_.unpack(), bias_.unpack(),
        {input_requires_grad_, weight_requires_grad_, bias_requires_grad_});
    return {grad_input, grad_weight, grad_bias};
}


EmbeddingBackward0::EmbeddingBackward0(
    const Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq)
    : num_weights_(num_weights), padding_idx_(padding_idx), scale_grad_by_freq_(scale_grad_by_freq)
{
    indices_.save(indices);
}
std::string EmbeddingBackward0::name() const { return "EmbeddingBackward0"; }

TensorList EmbeddingBackward0::apply(const TensorList& grads) {
    return {embedding_dense_backward(grads[0], indices_.unpack(), num_weights_, padding_idx_, scale_grad_by_freq_)};
}


ConstantPadNdBackward0::ConstantPadNdBackward0(IntArrayRef pad)
    : pad_(pad.begin(), pad.end()) {}
std::string ConstantPadNdBackward0::name() const { return "ConstantPadNdBackward0"; }

// From: torch/csrc/autograd/FunctionsManual.cpp (constant_pad_nd_backward)
TensorList ConstantPadNdBackward0::apply(const TensorList& grads) {
    DimVector negated(pad_.size());
    for (size_t i = 0; i < pad_.size(); ++i) {
        negated[i] = -pad_[i];
    }
    return {constant_pad_nd(grads[0], negated, 0)};
}


static std::string padding_node_name(PaddingMode mode, size_t pad_size, const char* suffix) {
    const char* kind = mode == PaddingMode::Reflect ? "ReflectionPad"
                     : mode == PaddingMode::Replicate ? "ReplicationPad" : "CircularPad";
    return std::string(kind) + std::to_string(pad_size / 2) + "D" + suffix;
}

PaddingBackward0::PaddingBackward0(const Tensor& self, IntArrayRef pad, PaddingMode mode)
    : self_shape_(self.shape().begin(), self.shape().end()),
      pad_(pad.begin(), pad.end()),
      mode_(mode) {}
std::string PaddingBackward0::name() const { return padding_node_name(mode_, pad_.size(), "Backward0"); }

TensorList PaddingBackward0::apply(const TensorList& grads) {
    return {padding_backward(grads[0], self_shape_, pad_, mode_)};
}


ScaledDotProductFlashAttentionForCpuBackward0::ScaledDotProductFlashAttentionForCpuBackward0(
    const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& attn_mask,
    bool is_causal, std::optional<double> scale, const Tensor& output, const Tensor& logsumexp)
    : is_causal_(is_causal), scale_(scale)
{
    query_.save(query);
    key_.save(key);
    value_.save(value);
    attn_mask_.save(attn_mask);
    output_.save(output, /*is_output=*/true);
    logsumexp_.save(logsumexp);
}
std::string ScaledDotProductFlashAttentionForCpuBackward0::name() const {
    return "ScaledDotProductFlashAttentionForCpuBackward0";
}

TensorList ScaledDotProductFlashAttentionForCpuBackward0::apply(const TensorList& grads) {
    auto [grad_query, grad_key, grad_value] = scaled_dot_product_flash_attention_for_cpu_backward(
        grads[0], query_.unpack(), key_.unpack(), value_.unpack(), output_.unpack(), logsumexp_.unpack(),
        /*dropout_p=*/0.0, is_causal_, attn_mask_.unpack(), scale_);
    return {grad_query, grad_key, grad_value};
}


ScaledDotProductEfficientAttentionBackward0::ScaledDotProductEfficientAttentionBackward0(
    const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& attn_bias,
    double dropout_p, bool is_causal, std::optional<double> scale, const Tensor& output,
    const Tensor& log_sumexp, const Tensor& philox_seed, const Tensor& philox_offset)
    : dropout_p_(dropout_p), is_causal_(is_causal), scale_(scale)
{
    query_.save(query);
    key_.save(key);
    value_.save(value);
    attn_bias_.save(attn_bias);
    output_.save(output, /*is_output=*/true);
    log_sumexp_.save(log_sumexp);
    philox_seed_.save(philox_seed);
    philox_offset_.save(philox_offset);
}
std::string ScaledDotProductEfficientAttentionBackward0::name() const {
    return "ScaledDotProductEfficientAttentionBackward0";
}

TensorList ScaledDotProductEfficientAttentionBackward0::apply(const TensorList& grads) {
    auto [grad_query, grad_key, grad_value] = scaled_dot_product_efficient_attention_backward(
        grads[0], query_.unpack(), key_.unpack(), value_.unpack(), attn_bias_.unpack(), output_.unpack(),
        log_sumexp_.unpack(), philox_seed_.unpack(), philox_offset_.unpack(), dropout_p_, is_causal_, scale_);
    return {grad_query, grad_key, grad_value};
}


NativeDropoutBackward0::NativeDropoutBackward0(const Tensor& mask, double scale)
    : scale_(scale)
{
    result1_.save(mask);
}
std::string NativeDropoutBackward0::name() const { return "NativeDropoutBackward0"; }

TensorList NativeDropoutBackward0::apply(const TensorList& grads) {
    return {native_dropout_backward(grads[0], result1_.unpack(), scale_)};
}


// ============================================================
// Double backward
// ============================================================

// From: tools/autograd/derivatives.yaml (threshold_backward)
// Linear in grad_output; piecewise constant in self, so its derivative there
// is zero
ThresholdBackwardBackward0::ThresholdBackwardBackward0(
    const Tensor& grad_output, const Tensor& self, Scalar threshold)
    : threshold_(threshold), needs_grad_{grad_output.requires_grad(), self.requires_grad()}
{
    self_.save(self);
}
std::string ThresholdBackwardBackward0::name() const { return "ThresholdBackwardBackward0"; }

TensorList ThresholdBackwardBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_grad_output, grad_self;
    if (needs_grad_[0]) grad_grad_output = threshold_backward(grad, self_.unpack(), threshold_);
    if (needs_grad_[1]) grad_self = zeros_like(grad);
    return {grad_grad_output, grad_self};
}


// From: tools/autograd/derivatives.yaml (gelu_backward)
//       torch/csrc/autograd/FunctionsManual.cpp (gelu_double_backward)
// gelu_backward(g, x) = g * gelu'(x): linear in g, so d/dg is gelu_backward
// again, and d/dx = gg * g * gelu''(x)
GeluBackwardBackward0::GeluBackwardBackward0(const Tensor& grad_output, const Tensor& self, GeluType approximate)
    : approximate_(approximate), needs_grad_{grad_output.requires_grad(), self.requires_grad()}
{
    grad_output_.save(grad_output);
    self_.save(self);
}
std::string GeluBackwardBackward0::name() const { return "GeluBackwardBackward0"; }

TensorList GeluBackwardBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    const Tensor self = self_.unpack();
    const char* approximate = approximate_ == GeluType::Tanh ? "tanh" : "none";
    Tensor grad_grad_output, grad_self;
    if (needs_grad_[0]) {
        grad_grad_output = gelu_backward(grad, self, approximate);
    }
    if (needs_grad_[1]) {
        Tensor second_derivative;
        if (approximate_ == GeluType::Tanh) {
            // gelu' = f'gh + f g h with f = 0.5 x, g = 1 - tanh^2(inner),
            // h = inner'; differentiating each factor once more
            const Tensor inner = (self + self.pow(3) * kGeluKappa) * kGeluBeta;
            const Tensor tanh_inner = inner.tanh();
            const Tensor sech_inner = inner.cosh().reciprocal();
            const Tensor f = self * 0.5;
            const Tensor g = -(tanh_inner * tanh_inner) + 1;
            const Tensor h = (self * self * (3 * kGeluKappa) + 1) * kGeluBeta;
            const Tensor f_prime_gh = g * h * 0.5;
            const Tensor g_prime = sech_inner * 2 * (-sech_inner * tanh_inner) * h;
            const Tensor g_prime_fh = f * h * g_prime;
            const Tensor h_prime_fg = f * g * (self * (6 * kGeluKappa * kGeluBeta));
            second_derivative = f_prime_gh * 2 + g_prime_fh + h_prime_fg;
        } else {
            // gelu'' = 2 pdf(x) - x^2 pdf(x)
            const Tensor self_sq = self * self;
            const Tensor pdf = (self_sq * -0.5).exp() * kGeluPdfScale;
            second_derivative = pdf * 2 - self_sq * pdf;
        }
        grad_self = grad * grad_output_.unpack() * second_derivative;
    }
    return {grad_grad_output, grad_self};
}


// From: torch/csrc/autograd/FunctionsManual.cpp (embedding_dense_double_backward_symint)
// The frequency scaling is not applied again: the first backward did it
EmbeddingDenseBackwardBackward0::EmbeddingDenseBackwardBackward0(const Tensor& indices, int64_t padding_idx)
    : padding_idx_(padding_idx)
{
    indices_.save(indices);
}
std::string EmbeddingDenseBackwardBackward0::name() const { return "EmbeddingDenseBackwardBackward0"; }

TensorList EmbeddingDenseBackwardBackward0::apply(const TensorList& grads) {
    const Tensor indices = indices_.unpack();
    Tensor gg_weight = embedding(grads[0], indices);
    if (padding_idx_ >= 0) {
        gg_weight = masked_fill(gg_weight, (indices == padding_idx_).unsqueeze(-1), 0);
    }
    return {gg_weight};
}


// From: tools/autograd/derivatives.yaml (_softmax_backward_data)
//       torch/csrc/autograd/FunctionsManual.cpp (softmax_double_backward)
// softmax_backward_data(g, y) = y * (g - sum(g * y)), so with gg the
// incoming gradient: d/dg = softmax_backward_data(gg, y) and
// d/dy = g * gg - sum(y * g) * gg - g * sum(y * gg)
SoftmaxBackwardDataBackward0::SoftmaxBackwardDataBackward0(
    const Tensor& grad_output, const Tensor& output, int64_t dim)
    : dim_(dim), needs_grad_{grad_output.requires_grad(), output.requires_grad()}
{
    grad_output_.save(grad_output);
    output_.save(output);
}
std::string SoftmaxBackwardDataBackward0::name() const { return "SoftmaxBackwardDataBackward0"; }

TensorList SoftmaxBackwardDataBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    const Tensor output = output_.unpack();
    Tensor grad_grad_output, grad_output_out;
    if (needs_grad_[0]) {
        grad_grad_output = softmax_backward_data(grad, output, dim_);
    }
    if (needs_grad_[1]) {
        const Tensor grad_output = grad_output_.unpack();
        grad_output_out = grad_output * grad - (output * grad_output).sum({dim_}, true) * grad
            - grad_output * (output * grad).sum({dim_}, true);
    }
    return {grad_grad_output, grad_output_out};
}


// From: tools/autograd/derivatives.yaml (_log_softmax_backward_data)
// log_softmax_backward_data(g, y) = g - exp(y) * sum(g)
LogSoftmaxBackwardDataBackward0::LogSoftmaxBackwardDataBackward0(
    const Tensor& grad_output, const Tensor& output, int64_t dim)
    : dim_(dim), needs_grad_{grad_output.requires_grad(), output.requires_grad()}
{
    grad_output_.save(grad_output);
    output_.save(output);
}
std::string LogSoftmaxBackwardDataBackward0::name() const { return "LogSoftmaxBackwardDataBackward0"; }

TensorList LogSoftmaxBackwardDataBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    const Tensor output = output_.unpack();
    Tensor grad_grad_output, grad_output_out;
    if (needs_grad_[0]) {
        grad_grad_output = grad - (grad * output.exp()).sum({dim_}, true);
    }
    if (needs_grad_[1]) {
        grad_output_out = -grad_output_.unpack().sum({dim_}, true) * output.exp() * grad;
    }
    return {grad_grad_output, grad_output_out};
}


// From: torch/csrc/autograd/FunctionsManual.cpp (native_dropout_double_backward)
NativeDropoutBackwardBackward0::NativeDropoutBackwardBackward0(const Tensor& mask, double scale)
    : scale_(scale)
{
    mask_.save(mask);
}
std::string NativeDropoutBackwardBackward0::name() const { return "NativeDropoutBackwardBackward0"; }

TensorList NativeDropoutBackwardBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    return {grad * (mask_.unpack().to(grad.dtype()) * scale_)};
}


// From: tools/autograd/derivatives.yaml (nll_loss_backward)
// Linear in grad_output: its derivative applies nll_loss to the incoming
// gradient. self only gives the shape.
NllLossBackwardBackward0::NllLossBackwardBackward0(
    const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index)
    : reduction_(reduction), ignore_index_(ignore_index), self_requires_grad_(self.requires_grad())
{
    target_.save(target);
    weight_.save(weight);
}
std::string NllLossBackwardBackward0::name() const { return "NllLossBackwardBackward0"; }

TensorList NllLossBackwardBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_self = self_requires_grad_ ? zeros_like(grad) : Tensor();
    return {nll_loss(grad, target_.unpack(), weight_.unpack(), reduction_, ignore_index_), grad_self};
}


// From: torch/csrc/autograd/FunctionsManual.cpp (max_pool_double_backward)
// Each output gradient came from one input position: read it back there
MaxPool2DWithIndicesBackwardBackward0::MaxPool2DWithIndicesBackwardBackward0(const Tensor& indices) {
    indices_.save(indices);
}
std::string MaxPool2DWithIndicesBackwardBackward0::name() const { return "MaxPool2DWithIndicesBackwardBackward0"; }

TensorList MaxPool2DWithIndicesBackwardBackward0::apply(const TensorList& grads) {
    const Tensor indices = indices_.unpack();
    return {gather(grads[0].flatten(-2), -1, indices.flatten(-2)).reshape(indices.shape())};
}


// From: tools/autograd/derivatives.yaml (avg_pool2d_backward)
AvgPool2DBackwardBackward0::AvgPool2DBackwardBackward0(
    IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding,
    bool ceil_mode, bool count_include_pad, std::optional<int64_t> divisor_override)
    : kernel_size_(kernel_size.begin(), kernel_size.end()),
      stride_(stride.begin(), stride.end()),
      padding_(padding.begin(), padding.end()),
      ceil_mode_(ceil_mode),
      count_include_pad_(count_include_pad),
      divisor_override_(divisor_override) {}
std::string AvgPool2DBackwardBackward0::name() const { return "AvgPool2DBackwardBackward0"; }

TensorList AvgPool2DBackwardBackward0::apply(const TensorList& grads) {
    return {avg_pool2d(grads[0], kernel_size_, stride_, padding_, ceil_mode_,
                       count_include_pad_, divisor_override_)};
}


PaddingBackwardBackward0::PaddingBackwardBackward0(IntArrayRef pad, PaddingMode mode)
    : pad_(pad.begin(), pad.end()), mode_(mode) {}
std::string PaddingBackwardBackward0::name() const {
    return padding_node_name(mode_, pad_.size(), "BackwardBackward0");
}

TensorList PaddingBackwardBackward0::apply(const TensorList& grads) {
    return {padding(grads[0], pad_, mode_)};
}

} // namespace raxis::autograd
