#include <raxis/autograd/functions/nn_backward.h>

#include <raxis/ops/activation.h>
#include <raxis/ops/convolution.h>
#include <raxis/ops/normalization.h>
#include <raxis/ops/pooling.h>
#include <raxis/ops/softmax.h>
#include <raxis/tensor/shape_utils.h>


namespace raxis::autograd {

ReluBackward0::ReluBackward0(const Tensor& result) {
    result_.save(result, /*is_output=*/true);
}
std::string ReluBackward0::name() const { return "ReluBackward0"; }

// relu'(x) is 1 where x > 0, which is the same set as result > 0
TensorList ReluBackward0::apply(const TensorList& grads) {
    return {ops::threshold_backward(grads[0], result_.unpack(), 0)};
}


SoftmaxBackward0::SoftmaxBackward0(const Tensor& result, int64_t dim)
    : dim_(dim)
{
    result_.save(result, /*is_output=*/true);
}
std::string SoftmaxBackward0::name() const { return "SoftmaxBackward0"; }

TensorList SoftmaxBackward0::apply(const TensorList& grads) {
    return {ops::softmax_backward_data(grads[0], result_.unpack(), dim_)};
}


LogSoftmaxBackward0::LogSoftmaxBackward0(const Tensor& result, int64_t dim)
    : dim_(dim)
{
    result_.save(result, /*is_output=*/true);
}
std::string LogSoftmaxBackward0::name() const { return "LogSoftmaxBackward0"; }

TensorList LogSoftmaxBackward0::apply(const TensorList& grads) {
    return {ops::log_softmax_backward_data(grads[0], result_.unpack(), dim_)};
}


NllLossBackward0::NllLossBackward0(
    const Tensor& self, const Tensor& target,
    Reduction reduction, int64_t ignore_index, double total_weight)
    : reduction_(reduction), ignore_index_(ignore_index), total_weight_(total_weight)
{
    self_.save(self);
    target_.save(target);
}
std::string NllLossBackward0::name() const { return "NllLossBackward0"; }

// Only self gets a gradient: target holds class indices
TensorList NllLossBackward0::apply(const TensorList& grads) {
    return {ops::nll_loss_backward(
        grads[0], self_.unpack(), target_.unpack(), reduction_, ignore_index_, total_weight_)};
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
    auto [grad_input, grad_weight, grad_bias] = ops::convolution_backward(
        grads[0], input_.unpack(), weight_.unpack(), stride_, padding_, dilation_, groups_,
        {input_requires_grad_, weight_requires_grad_, bias_requires_grad_});
    return {grad_input, grad_weight, grad_bias};
}


MaxPool2DWithIndicesBackward0::MaxPool2DWithIndicesBackward0(const Tensor& self, const Tensor& indices)
    : self_shape_(self.shape().begin(), self.shape().end())
{
    result1_.save(indices);
}
std::string MaxPool2DWithIndicesBackward0::name() const { return "MaxPool2DWithIndicesBackward0"; }

TensorList MaxPool2DWithIndicesBackward0::apply(const TensorList& grads) {
    return {ops::max_pool2d_with_indices_backward(grads[0], self_shape_, result1_.unpack())};
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
    return {ops::avg_pool2d_backward(
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
    auto [grad_input, grad_weight, grad_bias] = ops::native_batch_norm_backward(
        grads[0], input_.unpack(), weight_.unpack(), running_mean_.unpack(), running_var_.unpack(),
        result1_.unpack(), result2_.unpack(), training_, eps_,
        {input_requires_grad_, weight_requires_grad_, bias_requires_grad_});
    return {grad_input, grad_weight, grad_bias};
}

} // namespace raxis::autograd
