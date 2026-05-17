
#include <raxis/ops/reduce_ops.h>

#include <raxis/tensor/tensor.h>

#include <raxis/autograd/variable.h>
#include <raxis/autograd/functions/reduce_backward.h>


namespace raxis {

Tensor sum(const Tensor& self, std::optional<Dtype> dtype) {
    Tensor result = ops::sum(self, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::SumBackward0>(self);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor sum(const Tensor& self, IntArrayRef dim, bool keepdim, std::optional<Dtype> dtype) {
    Tensor result = ops::sum(self, dim, keepdim, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::SumBackward1>(self, dim, keepdim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}


Tensor mean(const Tensor& self, std::optional<Dtype> dtype) {
    Tensor result = ops::mean(self, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::MeanBackward0>(self);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor mean(const Tensor& self, IntArrayRef dim, bool keepdim, std::optional<Dtype> dtype) {
    Tensor result = ops::mean(self, dim, keepdim, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::MeanBackward1>(self, dim, keepdim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}


Tensor prod(const Tensor& self, std::optional<Dtype> dtype) {
    Tensor result = ops::prod(self, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ProdBackward0>(self, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor prod(const Tensor& self, IntArrayRef dim, bool keepdim, std::optional<Dtype> dtype) {
    Tensor result = ops::prod(self, dim, keepdim, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ProdBackward1>(
            self, dim, keepdim, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}


Tensor nansum(const Tensor& self, std::optional<Dtype> dtype) {
    Tensor result = ops::nansum(self, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::NansumBackward0>(self);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor nansum(const Tensor& self, IntArrayRef dim, bool keepdim, std::optional<Dtype> dtype) {
    Tensor result = ops::nansum(self, dim, keepdim, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::NansumBackward1>(self, dim, keepdim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}


Tensor nanmean(const Tensor& self, std::optional<Dtype> dtype) {
    Tensor result = ops::nanmean(self, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::NanmeanBackward0>(self);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor nanmean(const Tensor& self, IntArrayRef dim, bool keepdim, std::optional<Dtype> dtype) {
    Tensor result = ops::nanmean(self, dim, keepdim, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::NanmeanBackward1>(self, dim, keepdim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}


Tensor max(const Tensor& self) {
    Tensor result = ops::max(self);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::MaxBackward0>(self, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor min(const Tensor& self) {
    Tensor result = ops::min(self);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::MinBackward0>(self, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}


Tensor amax(const Tensor& self, IntArrayRef dim, bool keepdim) {
    Tensor result = ops::amax(self, dim, keepdim);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::AmaxBackward0>(
            self, dim, keepdim, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor amin(const Tensor& self, IntArrayRef dim, bool keepdim) {
    Tensor result = ops::amin(self, dim, keepdim);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::AminBackward0>(
            self, dim, keepdim, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

} // namespace raxis