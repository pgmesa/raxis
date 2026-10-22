#include <raxis/autograd/functions/nn_backward.h>

#include <raxis/ops/activation.h>
#include <raxis/ops/convolution.h>
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

} // namespace raxis::autograd
