
#include <raxis/autograd/functions/reduce_backward.h>

#include <bitset>
#include <limits>

#include <raxis/autograd/variable.h>
#include <raxis/autograd/grad_mode.h>
#include <raxis/tensor/factories.h>
#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/tensor_compare.h>


namespace raxis::autograd {

namespace {

// From: torch/csrc/autograd/FunctionsManual.cpp (unsqueeze_multiple)
// Restores the reduced dims (keepdim=false) as size 1 dims. They must be
// inserted in increasing order: each unsqueeze shifts the dims after it, so
// sum(dim=(2, 0)) of a 3-d tensor unsqueezes 0, then 2.
Tensor unsqueeze_reduced_dims(const Tensor& grad, IntArrayRef self_shape, IntArrayRef dim) {
    int64_t ndim = static_cast<int64_t>(self_shape.size());
    std::bitset<64> reduced;
    for (int64_t d : dim) {
        reduced.set(d < 0 ? d + ndim : d);
    }
    Tensor result = grad;
    for (int64_t d = 0; d < ndim; d++) {
        if (reduced.test(d)) {
            result = result.unsqueeze(d);
        }
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


ProdBackward0::ProdBackward0(const Tensor& self, const Tensor& result) {
    self_.save(self);
    result_.save(result, /*is_output=*/true);
}

std::string ProdBackward0::name() const { return "ProdBackward0"; }

TensorList ProdBackward0::apply(const TensorList& grads) {
    const Tensor& self = self_.unpack();
    return {(grads[0] * result_.unpack()).broadcast_to(self.shape()) / self};
}

ProdBackward1::ProdBackward1(const Tensor& self, IntArrayRef dim, bool keepdim,
                             const Tensor& result)
    : dim_(dim), keepdim_(keepdim)
{
    self_.save(self);
    result_.save(result, /*is_output=*/true);
}

std::string ProdBackward1::name() const { return "ProdBackward1"; }

TensorList ProdBackward1::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    Tensor res = result_.unpack();
    const Tensor& self = self_.unpack();

    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, DimVector(self.shape()), dim_);
        res = unsqueeze_reduced_dims(res, DimVector(self.shape()), dim_);
    }

    // grad * result / self, broadcast back
    return {(grad * res).broadcast_to(self.shape()) / self};
}


AmaxBackward0::AmaxBackward0(const Tensor& self, IntArrayRef dim, bool keepdim,
                             const Tensor& result)
    : dim_(dim), keepdim_(keepdim)
{
    self_.save(self);
    result_.save(result, /*is_output=*/true);
}

std::string AmaxBackward0::name() const { return "AmaxBackward0"; }

TensorList AmaxBackward0::apply(const TensorList& grads) {
    const Tensor& self = self_.unpack();
    Tensor grad = grads[0];
    Tensor res = result_.unpack();

    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, DimVector(self.shape()), dim_);
        res = unsqueeze_reduced_dims(res, DimVector(self.shape()), dim_);
    }

    Tensor mask = (self == res.broadcast_to(self.shape())).to(grad.dtype());
    Tensor num_matches = mask.sum(dim_, /*keepdim=*/true);
    return {grad.broadcast_to(self.shape()) * mask / num_matches.broadcast_to(self.shape())};
}


AminBackward0::AminBackward0(const Tensor& self, IntArrayRef dim, bool keepdim,
                             const Tensor& result)
    : dim_(dim), keepdim_(keepdim)
{
    self_.save(self);
    result_.save(result, /*is_output=*/true);
}

std::string AminBackward0::name() const { return "AminBackward0"; }

TensorList AminBackward0::apply(const TensorList& grads) {
    const Tensor& self = self_.unpack();
    Tensor grad = grads[0];
    Tensor res = result_.unpack();

    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, DimVector(self.shape()), dim_);
        res = unsqueeze_reduced_dims(res, DimVector(self.shape()), dim_);
    }

    Tensor mask = (self == res.broadcast_to(self.shape())).to(grad.dtype());
    Tensor num_matches = mask.sum(dim_, /*keepdim=*/true);
    return {grad.broadcast_to(self.shape()) * mask / num_matches.broadcast_to(self.shape())};
}


// From: torch/csrc/autograd/FunctionsManual.cpp (var_backward)
// d var / dx_i = 2 (x_i - mean) / (n - correction)
static Tensor var_backward(
    Tensor grad, const Tensor& self, IntArrayRef dim, double correction, bool keepdim
) {
    if (self.ndim() == 0 || dim.empty()) {
        double dof = static_cast<double>(self.numel()) - correction;
        if (dof <= 0) {
            // 2 / dof is infinite: inf * (x - mean), NaN where x == mean
            return grad * ::raxis::where(self.eq(self.mean()),
                std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity());
        }
        return grad * (2.0 / dof) * (self - self.mean());
    }
    if (!keepdim && self.ndim() > 1) {
        grad = unsqueeze_reduced_dims(grad, self.shape(), dim);
    }
    int64_t count = 1;
    for (int64_t d : dim) {
        count *= self.shape()[d < 0 ? d + self.ndim() : d];
    }
    return grad * (2.0 / (static_cast<double>(count) - correction)) * (self - self.mean(dim, /*keepdim=*/true));
}

VarBackward0::VarBackward0(const Tensor& self, IntArrayRef dim, double correction, bool keepdim)
    : dim_(dim), correction_(correction), keepdim_(keepdim)
{
    self_.save(self);
}

std::string VarBackward0::name() const { return "VarBackward0"; }

TensorList VarBackward0::apply(const TensorList& grads) {
    return {var_backward(grads[0], self_.unpack(), dim_, correction_, keepdim_)};
}

StdBackward0::StdBackward0(
    const Tensor& self, const Tensor& result, IntArrayRef dim, double correction, bool keepdim)
    : dim_(dim), correction_(correction), keepdim_(keepdim)
{
    self_.save(self);
    result_.save(result, /*is_output=*/true);
}

std::string StdBackward0::name() const { return "StdBackward0"; }

// std = sqrt(var): d std = d var / (2 std), and 0 where std is 0
TensorList StdBackward0::apply(const TensorList& grads) {
    Tensor result = result_.unpack();
    Tensor grad_var = where(result.eq(0.0), 0, grads[0] / (result * 2.0));
    return {var_backward(grad_var, self_.unpack(), dim_, correction_, keepdim_)};
}

MaxBackward0::MaxBackward0(const Tensor& self, int64_t dim, const Tensor& indices, bool keepdim)
    : self_shape_(self.shape()), dim_(dim), keepdim_(keepdim)
{
    indices_.save(indices);
}

std::string MaxBackward0::name() const { return "MaxBackward0"; }
std::string MinBackward0::name() const { return "MinBackward0"; }

// From: aten/src/ATen/native/ReduceOps.cpp (value_selecting_reduction_backward)
// zeros with the incoming gradient scattered to the selected positions
TensorList MaxBackward0::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    Tensor indices = indices_.unpack();
    if (!keepdim_ && !self_shape_.empty()) {
        grad = grad.unsqueeze(dim_);
        indices = indices.unsqueeze(dim_);
    }
    Tensor grad_self = zeros(self_shape_, grad.options());
    return {scatter_(grad_self, dim_, indices, grad)};
}

MaxBackward1::MaxBackward1(const Tensor& self, const Tensor& result) {
    self_.save(self);
    result_.save(result, /*is_output=*/true);
}

std::string MaxBackward1::name() const { return "MaxBackward1"; }

TensorList MaxBackward1::apply(const TensorList& grads) {
    const Tensor& self = self_.unpack();
    Tensor mask = (self == result_.unpack().broadcast_to(self.shape())).to(grads[0].dtype());
    Tensor num_matches = mask.sum();
    return {grads[0].broadcast_to(self.shape()) * mask / num_matches.broadcast_to(self.shape())};
}


MinBackward1::MinBackward1(const Tensor& self, const Tensor& result) {
    self_.save(self);
    result_.save(result, /*is_output=*/true);
}

std::string MinBackward1::name() const { return "MinBackward1"; }

TensorList MinBackward1::apply(const TensorList& grads) {
    const Tensor& self = self_.unpack();
    Tensor mask = (self == result_.unpack().broadcast_to(self.shape())).to(grads[0].dtype());
    Tensor num_matches = mask.sum();
    return {grads[0].broadcast_to(self.shape()) * mask / num_matches.broadcast_to(self.shape())};
}


NansumBackward0::NansumBackward0(const Tensor& self)
    : self_shape_(self.shape())
{
    self_.save(self);
}

std::string NansumBackward0::name() const { return "NansumBackward0"; }

TensorList NansumBackward0::apply(const TensorList& grads) {
    Tensor not_nan = self_.unpack().isnan().logical_not().to(grads[0].dtype());
    return {grads[0].broadcast_to(self_shape_) * not_nan};
}

NansumBackward1::NansumBackward1(const Tensor& self, IntArrayRef dim, bool keepdim)
    : self_shape_(self.shape()), dim_(dim), keepdim_(keepdim)
{
    self_.save(self);
}

std::string NansumBackward1::name() const { return "NansumBackward1"; }

TensorList NansumBackward1::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, self_shape_, dim_);
    }
    Tensor not_nan = self_.unpack().isnan().logical_not().to(grad.dtype());
    return {grad.broadcast_to(self_shape_) * not_nan};
}


NanmeanBackward0::NanmeanBackward0(const Tensor& self)
    : self_shape_(self.shape())
{
    self_.save(self);
}

std::string NanmeanBackward0::name() const { return "NanmeanBackward0"; }

TensorList NanmeanBackward0::apply(const TensorList& grads) {
    Tensor not_nan = self_.unpack().isnan().logical_not().to(grads[0].dtype());
    Tensor count = not_nan.sum();
    return {grads[0].broadcast_to(self_shape_) * not_nan / count.broadcast_to(self_shape_)};
}

NanmeanBackward1::NanmeanBackward1(const Tensor& self, IntArrayRef dim, bool keepdim)
    : self_shape_(self.shape()), dim_(dim), keepdim_(keepdim)
{
    self_.save(self);
}

std::string NanmeanBackward1::name() const { return "NanmeanBackward1"; }

TensorList NanmeanBackward1::apply(const TensorList& grads) {
    const Tensor& self = self_.unpack();
    Tensor grad = grads[0];
    Tensor count = count_nonnan(self, dim_, /*keepdim=*/true);

    if (!keepdim_ && !dim_.empty()) {
        grad = unsqueeze_reduced_dims(grad, self_shape_, dim_);
    }

    Tensor not_nan = self.isnan().logical_not().to(grad.dtype());
    return {grad.broadcast_to(self_shape_) * not_nan / count.broadcast_to(self_shape_)};
}

CumsumBackward0::CumsumBackward0(const Tensor& self, int64_t dim)
    : self_dtype_(self.dtype()), dim_(dim) {}

std::string CumsumBackward0::name() const { return "CumsumBackward0"; }

// From: torch/csrc/autograd/FunctionsManual.cpp (cumsum_backward)
// out[i] = sum of x[j] for j <= i, so dL/dx[j] = sum of grad[i] for i >= j:
// a cumulative sum from the end. PyTorch writes it flip + cumsum + flip.
TensorList CumsumBackward0::apply(const TensorList& grads) {
    Tensor grad = grads[0].to(self_dtype_);
    if (grad.numel() <= 1 || grad.ndim() == 0 || grad.shape()[wrap_dim(dim_, grad.ndim())] == 1) {
        return {grad};
    }
    return {reversed_cumsum(grad, dim_)};
}


CumprodBackward0::CumprodBackward0(const Tensor& self, int64_t dim, const Tensor& result) : dim_(dim) {
    self_.save(self);
    result_.save(result, /*is_output=*/true);
}

std::string CumprodBackward0::name() const { return "CumprodBackward0"; }

// From: aten/src/ATen/native/ReduceOps.cpp (cumprod_backward)
// y_j = x_0 * ... * x_j, so dy_j/dx_k = y_j / x_k for j >= k and
// dL/dx_k = (sum over j >= k of grad_j * y_j) / x_k: a reversed cumsum, then
// a division. A zero in x breaks the division. Per slice, with z1 the first
// zero and z2 the second (or the end):
//   k < z1: the same formula, summing only j < z1 (later y_j are 0)
//   k > z1: 0, every product through x_k also goes through x_z1 = 0
//   k = z1: y_{z1-1} * sum over z1 <= j < z2 of grad_j * x_{z1+1} * ... * x_j
// PyTorch builds these masks with whole-tensor ops; so does this. Its
// second-derivative path (an O(n^2) loop) is not ported: with zeros and
// create_graph this raises instead.
TensorList CumprodBackward0::apply(const TensorList& grads) {
    const Tensor input = self_.unpack();
    const Tensor grad = grads[0].to(input.dtype());
    if (input.numel() <= 1) {
        return {grad};
    }
    const int64_t dim = wrap_dim(dim_, input.ndim());
    if (input.shape()[dim] == 1) {
        return {grad};
    }

    const Tensor output = result_.unpack();
    const Tensor w = output * grad;
    const Tensor is_zero = input == 0;
    if (!is_zero.any().item<bool>()) {
        return {reversed_cumsum(w, dim).div(input).to(input.dtype())};
    }
    RAXIS_CHECK(!GradMode::is_enabled(),
        "cumprod: the second derivative with zeros in the input is not implemented");

    // Number of zeros up to each position: 0 before z1, 1 in [z1, z2)
    const Tensor zeros_seen = is_zero.cumsum(dim);

    Tensor mask = zeros_seen == 0;
    const Tensor before_first_zero = reversed_cumsum(masked_fill(w, mask.logical_not(), 0), dim).div(input);
    Tensor grad_input = where(mask, before_first_zero, 0);

    mask = zeros_seen == 1;
    // max over a bool slice gives the first true: z1 (0 when there is no zero,
    // which first_zero_mask filters out)
    const Tensor first_zero_index = std::get<1>(max(mask, dim, /*keepdim=*/true));
    const Tensor first_zero_mask = mask.logical_and(is_zero);
    const Tensor after_first_zero = mask.logical_and(first_zero_mask.logical_not());

    const Tensor prods_after_zero = masked_fill(input, after_first_zero.logical_not(), 1).cumprod(dim);
    const Tensor grad_until_second_zero = masked_fill(grad, zeros_seen != 1, 0);
    const Tensor output_before_zero = masked_fill(
        gather(output, dim, (first_zero_index - 1).relu()), first_zero_index == 0, 1);
    const Tensor at_first_zero = (prods_after_zero * grad_until_second_zero).sum({dim}, /*keepdim=*/true)
        * output_before_zero;
    grad_input = where(first_zero_mask, at_first_zero, grad_input);
    return {grad_input.to(input.dtype())};
}


ReversedCumsumBackward0::ReversedCumsumBackward0(int64_t dim) : dim_(dim) {}

std::string ReversedCumsumBackward0::name() const { return "ReversedCumsumBackward0"; }

// The transpose of a sum from the end is a sum from the start
TensorList ReversedCumsumBackward0::apply(const TensorList& grads) {
    return {cumsum(grads[0], dim_)};
}

} // namespace raxis::autograd