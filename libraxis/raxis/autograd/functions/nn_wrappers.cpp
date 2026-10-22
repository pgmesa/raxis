#include <raxis/ops/activation.h>
#include <raxis/ops/convolution.h>
#include <raxis/ops/loss.h>
#include <raxis/ops/pooling.h>
#include <raxis/ops/softmax.h>

#include <raxis/tensor/shape_utils.h>

#include <raxis/autograd/autograd_macros.h>
#include <raxis/autograd/functions/nn_backward.h>


namespace raxis {

RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(relu, ReluBackward0)

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

Tensor nll_loss(const Tensor& self, const Tensor& target, Reduction reduction, int64_t ignore_index) {
    auto [result, total_weight] = ops::nll_loss_forward(self, target, reduction, ignore_index);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::NllLossBackward0>(
            self, target, reduction, ignore_index, total_weight);
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
        auto grad_fn = std::make_shared<autograd::MaxPool2DWithIndicesBackward0>(input, indices);
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

} // namespace raxis
