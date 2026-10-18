#include <raxis/ops/activation.h>
#include <raxis/ops/loss.h>
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

} // namespace raxis
