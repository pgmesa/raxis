#include <raxis/autograd/functions/nn_backward.h>

#include <raxis/ops/activation.h>
#include <raxis/ops/softmax.h>
#include <raxis/tensor/shape_utils.h>


namespace raxis::autograd {

ReluBackward0::ReluBackward0(const Tensor& result) : result_(result, /*is_output=*/true) {}
std::string ReluBackward0::name() const { return "ReluBackward0"; }

// relu'(x) is 1 where x > 0, which is the same set as result > 0
TensorList ReluBackward0::apply(const TensorList& grads) {
    return {ops::threshold_backward(grads[0], result_.unpack(), 0)};
}


SoftmaxBackward0::SoftmaxBackward0(const Tensor& result, int64_t dim)
    : result_(result, /*is_output=*/true), dim_(dim) {}
std::string SoftmaxBackward0::name() const { return "SoftmaxBackward0"; }

TensorList SoftmaxBackward0::apply(const TensorList& grads) {
    return {ops::softmax_backward_data(grads[0], result_.unpack(), dim_)};
}


LogSoftmaxBackward0::LogSoftmaxBackward0(const Tensor& result, int64_t dim)
    : result_(result, /*is_output=*/true), dim_(dim) {}
std::string LogSoftmaxBackward0::name() const { return "LogSoftmaxBackward0"; }

TensorList LogSoftmaxBackward0::apply(const TensorList& grads) {
    return {ops::log_softmax_backward_data(grads[0], result_.unpack(), dim_)};
}


NllLossBackward0::NllLossBackward0(
    const Tensor& self, const Tensor& target,
    Reduction reduction, int64_t ignore_index, double total_weight)
    : self_(self), target_(target), reduction_(reduction),
      ignore_index_(ignore_index), total_weight_(total_weight) {}
std::string NllLossBackward0::name() const { return "NllLossBackward0"; }

// Only self gets a gradient: target holds class indices
TensorList NllLossBackward0::apply(const TensorList& grads) {
    return {ops::nll_loss_backward(
        grads[0], self_, target_, reduction_, ignore_index_, total_weight_)};
}

} // namespace raxis::autograd
