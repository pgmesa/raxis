
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/tensor_compare.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/shape_utils.h>

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
        auto grad_fn = std::make_shared<autograd::MaxBackward1>(self, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

template <typename Node>
static std::tuple<Tensor, Tensor> minmax_dim(
    const Tensor& self, int64_t dim, bool keepdim,
    std::tuple<Tensor, Tensor> (*op)(const Tensor&, int64_t, bool)
) {
    auto [values, indices] = op(self, dim, keepdim);
    if (autograd::requires_autograd(self)) {
        int64_t wrapped = wrap_dim(dim, std::max<int64_t>(self.ndim(), 1));
        auto grad_fn = std::make_shared<Node>(self, wrapped, indices, keepdim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(values, grad_fn);
    }
    return {values, indices};
}

std::tuple<Tensor, Tensor> max(const Tensor& self, int64_t dim, bool keepdim) {
    return minmax_dim<autograd::MaxBackward0>(self, dim, keepdim, &ops::max);
}

std::tuple<Tensor, Tensor> min(const Tensor& self, int64_t dim, bool keepdim) {
    return minmax_dim<autograd::MinBackward0>(self, dim, keepdim, &ops::min);
}

Tensor min(const Tensor& self) {
    Tensor result = ops::min(self);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::MinBackward1>(self, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}


Tensor var(const Tensor& self, IntArrayRef dim, std::optional<double> correction, bool keepdim) {
    Tensor result = ops::var(self, dim, correction, keepdim);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::VarBackward0>(
            self, dim, correction.value_or(1.0), keepdim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor std(const Tensor& self, IntArrayRef dim, std::optional<double> correction, bool keepdim) {
    Tensor result = ops::std(self, dim, correction, keepdim);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::StdBackward0>(
            self, result, dim, correction.value_or(1.0), keepdim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor argmax(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    return ops::argmax(self, dim, keepdim);
}

Tensor argmin(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    return ops::argmin(self, dim, keepdim);
}

Tensor cumsum(const Tensor& self, int64_t dim, std::optional<Dtype> dtype) {
    Tensor result = ops::cumsum(self, dim, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::CumsumBackward0>(self, dim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor cumprod(const Tensor& self, int64_t dim, std::optional<Dtype> dtype) {
    Tensor result = ops::cumprod(self, dim, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::CumprodBackward0>(self, dim, result);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor reversed_cumsum(const Tensor& self, int64_t dim) {
    Tensor result = ops::reversed_cumsum(self, dim);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ReversedCumsumBackward0>(dim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor all(const Tensor& self, IntArrayRef dim, bool keepdim) {
    return ops::all(self, dim, keepdim);
}

Tensor any(const Tensor& self, IntArrayRef dim, bool keepdim) {
    return ops::any(self, dim, keepdim);
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