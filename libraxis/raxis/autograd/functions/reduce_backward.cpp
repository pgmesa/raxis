
#include <raxis/autograd/functions/reduce_backward.h>

#include <raxis/autograd/variable.h>


namespace raxis::autograd {

namespace {

// Helper: restore reduced dimensions by unsqueezing
Tensor unsqueeze_reduced_dims(const Tensor& grad, IntArrayRef self_shape, IntArrayRef dim) {
    Tensor result = grad;
    for (int64_t d : dim) {
        int64_t nd = d < 0 ? d + static_cast<int64_t>(self_shape.size()) : d;
        result = result.unsqueeze(nd);
    }
    return result;
}

// Helper: count non-NaN elements along dims
Tensor count_nonnan(const Tensor& self, IntArrayRef dim, bool keepdim) {
    Tensor mask = self.isnan().logical_not().to(self.dtype());
    if (dim.empty()) {
        return mask.sum();
    }
    return mask.sum(dim, keepdim);
}

} // namespace


SumBackward0::SumBackward0(const Tensor& self)
    : self_shape_(self.shape()) {}

std::string SumBackward0::name() const { return "SumBackward0"; }

TensorList SumBackward0::apply(const TensorList& grads) {
    return {grads[0].broadcast_to(self_shape_)};
}

SumBackward1::SumBackward1(const Tensor& self, IntArrayRef dim, bool keepdim)
    : self_shape_(self.shape()), dim_(dim), keepdim_(keepdim) {}

std::string SumBackward1::name() const { return "SumBackward1"; }

TensorList SumBackward1::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, self_shape_, dim_);
    }
    return {grad.broadcast_to(self_shape_)};
}

MeanBackward0::MeanBackward0(const Tensor& self)
    : self_shape_(self.shape()), numel_(self.numel()) {}

std::string MeanBackward0::name() const { return "MeanBackward0"; }

TensorList MeanBackward0::apply(const TensorList& grads) {
    return {grads[0].broadcast_to(self_shape_) / static_cast<double>(numel_)};
}

MeanBackward1::MeanBackward1(const Tensor& self, IntArrayRef dim, bool keepdim)
    : self_shape_(self.shape()), dim_(dim), keepdim_(keepdim)
{
    numel_reduced_ = 1;
    if (dim.empty()) {
        numel_reduced_ = self.numel();
    } else {
        int64_t ndim = self.ndim();
        for (int64_t d : dim) {
            int64_t nd = d < 0 ? d + ndim : d;
            numel_reduced_ *= self_shape_[nd];
        }
    }
}

std::string MeanBackward1::name() const { return "MeanBackward1"; }

TensorList MeanBackward1::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, self_shape_, dim_);
    }
    return {grad.broadcast_to(self_shape_) / static_cast<double>(numel_reduced_)};
}


ProdBackward0::ProdBackward0(const Tensor& self, const Tensor& result)
    : self_(self), result_(result, /*is_output=*/true) {}

std::string ProdBackward0::name() const { return "ProdBackward0"; }

TensorList ProdBackward0::apply(const TensorList& grads) {
    return {(grads[0] * result_.unpack()).broadcast_to(self_.shape()) / self_};
}

ProdBackward1::ProdBackward1(const Tensor& self, IntArrayRef dim, bool keepdim,
                             const Tensor& result)
    : self_(self), result_(result, /*is_output=*/true), dim_(dim), keepdim_(keepdim) {}

std::string ProdBackward1::name() const { return "ProdBackward1"; }

TensorList ProdBackward1::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    Tensor res = result_.unpack();

    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, DimVector(self_.shape()), dim_);
        res = unsqueeze_reduced_dims(res, DimVector(self_.shape()), dim_);
    }

    // grad * result / self, broadcast back
    return {(grad * res).broadcast_to(self_.shape()) / self_};
}


AmaxBackward0::AmaxBackward0(const Tensor& self, IntArrayRef dim, bool keepdim,
                             const Tensor& result)
    : self_(self), result_(result, /*is_output=*/true), dim_(dim), keepdim_(keepdim) {}

std::string AmaxBackward0::name() const { return "AmaxBackward0"; }

TensorList AmaxBackward0::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    Tensor res = result_.unpack();

    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, DimVector(self_.shape()), dim_);
        res = unsqueeze_reduced_dims(res, DimVector(self_.shape()), dim_);
    }

    Tensor mask = (self_ == res.broadcast_to(self_.shape())).to(grad.dtype());
    Tensor num_matches = mask.sum(dim_, /*keepdim=*/true);
    return {grad.broadcast_to(self_.shape()) * mask / num_matches.broadcast_to(self_.shape())};
}


AminBackward0::AminBackward0(const Tensor& self, IntArrayRef dim, bool keepdim,
                             const Tensor& result)
    : self_(self), result_(result, /*is_output=*/true), dim_(dim), keepdim_(keepdim) {}

std::string AminBackward0::name() const { return "AminBackward0"; }

TensorList AminBackward0::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    Tensor res = result_.unpack();

    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, DimVector(self_.shape()), dim_);
        res = unsqueeze_reduced_dims(res, DimVector(self_.shape()), dim_);
    }

    Tensor mask = (self_ == res.broadcast_to(self_.shape())).to(grad.dtype());
    Tensor num_matches = mask.sum(dim_, /*keepdim=*/true);
    return {grad.broadcast_to(self_.shape()) * mask / num_matches.broadcast_to(self_.shape())};
}


MaxBackward0::MaxBackward0(const Tensor& self, const Tensor& result)
    : self_(self), result_(result, /*is_output=*/true) {}

std::string MaxBackward0::name() const { return "MaxBackward0"; }

TensorList MaxBackward0::apply(const TensorList& grads) {
    Tensor mask = (self_ == result_.unpack().broadcast_to(self_.shape())).to(grads[0].dtype());
    Tensor num_matches = mask.sum();
    return {grads[0].broadcast_to(self_.shape()) * mask / num_matches.broadcast_to(self_.shape())};
}


MinBackward0::MinBackward0(const Tensor& self, const Tensor& result)
    : self_(self), result_(result, /*is_output=*/true) {}

std::string MinBackward0::name() const { return "MinBackward0"; }

TensorList MinBackward0::apply(const TensorList& grads) {
    Tensor mask = (self_ == result_.unpack().broadcast_to(self_.shape())).to(grads[0].dtype());
    Tensor num_matches = mask.sum();
    return {grads[0].broadcast_to(self_.shape()) * mask / num_matches.broadcast_to(self_.shape())};
}


NansumBackward0::NansumBackward0(const Tensor& self)
    : self_(self), self_shape_(self.shape()) {}

std::string NansumBackward0::name() const { return "NansumBackward0"; }

TensorList NansumBackward0::apply(const TensorList& grads) {
    Tensor not_nan = self_.isnan().logical_not().to(grads[0].dtype());
    return {grads[0].broadcast_to(self_shape_) * not_nan};
}

NansumBackward1::NansumBackward1(const Tensor& self, IntArrayRef dim, bool keepdim)
    : self_(self), self_shape_(self.shape()), dim_(dim), keepdim_(keepdim) {}

std::string NansumBackward1::name() const { return "NansumBackward1"; }

TensorList NansumBackward1::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, self_shape_, dim_);
    }
    Tensor not_nan = self_.isnan().logical_not().to(grad.dtype());
    return {grad.broadcast_to(self_shape_) * not_nan};
}


NanmeanBackward0::NanmeanBackward0(const Tensor& self)
    : self_(self), self_shape_(self.shape()) {}

std::string NanmeanBackward0::name() const { return "NanmeanBackward0"; }

TensorList NanmeanBackward0::apply(const TensorList& grads) {
    Tensor not_nan = self_.isnan().logical_not().to(grads[0].dtype());
    Tensor count = not_nan.sum();
    return {grads[0].broadcast_to(self_shape_) * not_nan / count.broadcast_to(self_shape_)};
}

NanmeanBackward1::NanmeanBackward1(const Tensor& self, IntArrayRef dim, bool keepdim)
    : self_(self), self_shape_(self.shape()), dim_(dim), keepdim_(keepdim) {}

std::string NanmeanBackward1::name() const { return "NanmeanBackward1"; }

TensorList NanmeanBackward1::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    Tensor count = count_nonnan(self_, dim_, /*keepdim=*/true);

    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, self_shape_, dim_);
    }

    Tensor not_nan = self_.isnan().logical_not().to(grad.dtype());
    return {grad.broadcast_to(self_shape_) * not_nan / count.broadcast_to(self_shape_)};
}

} // namespace raxis::autograd