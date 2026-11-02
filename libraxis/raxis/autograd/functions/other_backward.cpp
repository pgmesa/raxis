
#include <raxis/autograd/functions/other_backward.h>

#include <cmath>

#include <raxis/ops/cat_ops.h>
#include <raxis/ops/tensor_compare.h>
#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/ops/tensor_transformations.h>
#include <raxis/ops/triangular_ops.h>
#include <raxis/autograd/variable.h>
#include <raxis/tensor/factories.h>


namespace raxis::autograd {

// ============================================================
// Cat / Stack / Unbind
// ============================================================

CatBackward0::CatBackward0(ArrayRef<Tensor> tensors, int64_t dim) : dim_(dim) {
    sizes_.reserve(tensors.size());
    requires_grad_.reserve(tensors.size());
    for (const auto& t : tensors) {
        sizes_.push_back(t.shape()[dim]);
        requires_grad_.push_back(t.requires_grad());
    }
}

std::string CatBackward0::name() const { return "CatBackward0"; }

TensorList CatBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    TensorList result;
    result.reserve(sizes_.size());

    int64_t offset = 0;
    for (size_t i = 0; i < sizes_.size(); ++i) {
        int64_t size = sizes_[i];
        if (requires_grad_[i]) {
            result.push_back(grad.slice(dim_, offset, offset + size));
        } else {
            result.emplace_back();
        }
        offset += size;
    }
    return result;
}


StackBackward0::StackBackward0(int64_t num_tensors, int64_t dim)
    : num_tensors_(num_tensors), dim_(dim) {}

std::string StackBackward0::name() const { return "StackBackward0"; }

TensorList StackBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    TensorList result;
    result.reserve(num_tensors_);
    for (int64_t i = 0; i < num_tensors_; ++i) {
        result.push_back(grad.select(dim_, i));
    }
    return result;
}

UnbindBackward0::UnbindBackward0(const Tensor& self, int64_t dim, int64_t num_outputs)
    : self_shape_(self.shape()), dim_(dim) {}

std::string UnbindBackward0::name() const { return "UnbindBackward0"; }

// From: torch/csrc/autograd/FunctionsManual.cpp (unbind_backward)
// Outputs that were not used have no gradient: they count as zeros, made
// with a 0-d zero expanded to the right shape (no allocation)
TensorList UnbindBackward0::apply(const TensorList& grads) {
    const Tensor* defined = nullptr;
    for (const Tensor& g : grads) {
        if (g.defined()) {
            defined = &g;
            break;
        }
    }
    RAXIS_INTERNAL_ASSERT(defined, "UnbindBackward0 called without any gradient");

    TensorList all_grads;
    all_grads.reserve(grads.size());
    for (const Tensor& g : grads) {
        all_grads.push_back(g.defined() ? g : zeros({}, defined->options()).expand(defined->shape()));
    }
    return {stack(all_grads, dim_)};
}


SplitWithSizesBackward0::SplitWithSizesBackward0(
    const Tensor& self, IntArrayRef split_sizes, int64_t dim)
    : self_shape_(self.shape()), split_sizes_(split_sizes), dim_(dim) {}

std::string SplitWithSizesBackward0::name() const { return "SplitWithSizesBackward0"; }

// From: torch/csrc/autograd/FunctionsManual.cpp (split_with_sizes_backward)
// The input grad is the output grads glued back together; unused pieces
// count as zeros
TensorList SplitWithSizesBackward0::apply(const TensorList& grads) {
    TensorOptions options;
    for (const Tensor& g : grads) {
        if (g.defined()) {
            options = g.options();
            break;
        }
    }

    TensorList all_grads;
    all_grads.reserve(grads.size());
    for (size_t i = 0; i < grads.size(); i++) {
        if (grads[i].defined()) {
            all_grads.push_back(grads[i]);
        } else {
            DimVector shape(self_shape_);
            shape[dim_] = split_sizes_[i];
            all_grads.push_back(zeros(shape, options));
        }
    }
    return {cat(all_grads, dim_)};
}

// ============================================================
// where / clamp
// ============================================================

WhereBackward0::WhereBackward0(const Tensor& condition, const Tensor& self, const Tensor& other)
    : self_shape_(self.shape()),
      other_shape_(other.shape()),
      self_requires_grad_(self.requires_grad()),
      other_requires_grad_(other.requires_grad())
{
    condition_.save(condition);
}

std::string WhereBackward0::name() const { return "WhereBackward0"; }

// Each element of the result came from self or from other: the gradient goes
// back to the one that was selected and the other gets 0
TensorList WhereBackward0::apply(const TensorList& grads) {
    const Tensor& condition = condition_.unpack();
    const Tensor& grad = grads[0];
    Tensor grad_self, grad_other;
    if (self_requires_grad_)
        grad_self = unbroadcast(where(condition, grad, 0), self_shape_);
    if (other_requires_grad_)
        grad_other = unbroadcast(where(condition, 0, grad), other_shape_);
    return {grad_self, grad_other};
}


ClampBackward0::ClampBackward0(const Tensor& self, const Tensor& min, const Tensor& max)
    : needs_grad_{self.requires_grad(), min.requires_grad(), max.requires_grad()}
{
    self_.save(self);
    min_.save(min);
    max_.save(max);
    shapes_[0] = DimVector(self.shape());
    if (min.defined()) shapes_[1] = DimVector(min.shape());
    if (max.defined()) shapes_[2] = DimVector(max.shape());
}

std::string ClampBackward0::name() const { return "ClampBackward0"; }

// From: torch/csrc/autograd/FunctionsManual.cpp (clamp_backward and
// clamp_backward_min_max for tensor bounds)
// The output is the input, min or max element by element, and the gradient
// goes to whichever was selected. Where two of them tie (x == min < max),
// each gets half. With reversed bounds (min > max) the output is max, so max
// gets everything. masked_fill(g, m, 0) is written where(m, 0, g).
TensorList ClampBackward0::apply(const TensorList& grads) {
    const Tensor& self = self_.unpack();
    const Tensor& min = min_.unpack();
    const Tensor& max = max_.unpack();
    const Tensor& grad = grads[0];
    const Tensor half = grad / 2;
    Tensor grad_self, grad_min, grad_max;

    if (min.defined() && max.defined()) {
        const Tensor min_lt_max = min.lt(max);
        const Tensor min_eq_max = min.eq(max);
        if (needs_grad_[0]) {
            const Tensor tie = self.eq(min).logical_or(self.eq(max)).logical_and(min_lt_max);
            const Tensor inactive = self.lt(min).logical_or(self.gt(max));
            grad_self = where(inactive, 0, where(tie, half, grad));
        }
        if (needs_grad_[1]) {
            const Tensor active = min_lt_max.logical_and(self.le(min))
                .logical_or(min_eq_max.logical_and(self.lt(min)));
            grad_min = where(active.logical_not(), 0, where(self.eq(min), half, grad));
        }
        if (needs_grad_[2]) {
            const Tensor active = max.lt(min)
                .logical_or(min_lt_max.logical_and(self.ge(max)))
                .logical_or(min_eq_max.logical_and(self.gt(max)));
            const Tensor tie = self.eq(max).logical_and(min_lt_max);
            grad_max = where(active.logical_not(), 0, where(tie, half, grad));
        }
    } else if (min.defined()) {
        const Tensor split = where(self.eq(min), half, grad);
        if (needs_grad_[0]) grad_self = where(self.lt(min), 0, split);
        if (needs_grad_[1]) grad_min = where(self.gt(min), 0, split);
    } else {
        const Tensor split = where(self.eq(max), half, grad);
        if (needs_grad_[0]) grad_self = where(self.gt(max), 0, split);
        if (needs_grad_[2]) grad_max = where(self.lt(max), 0, split);
    }

    if (grad_self.defined()) grad_self = unbroadcast(grad_self, shapes_[0]);
    if (grad_min.defined()) grad_min = unbroadcast(grad_min, shapes_[1]);
    if (grad_max.defined()) grad_max = unbroadcast(grad_max, shapes_[2]);
    return {grad_self, grad_min, grad_max};
}

ClampBackward1::ClampBackward1(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max)
    : min_(min), max_(max)
{
    self_.save(self);
}

std::string ClampBackward1::name() const { return "ClampBackward1"; }

// From: torch/csrc/autograd/FunctionsManual.cpp (clamp_backward)
// The gradient passes where the input was strictly inside (min, max). At the
// bounds the derivative is not defined: torch 2.14 uses 0 there (up to 2.13
// it used the subgradient 1, with >= and <=).
TensorList ClampBackward1::apply(const TensorList& grads) {
    const Tensor& self = self_.unpack();
    const Tensor& grad = grads[0];
    if (min_ && max_) {
        return {where(self.gt(*min_).logical_and(self.lt(*max_)), grad, 0)};
    } else if (min_) {
        return {where(self.gt(*min_), grad, 0)};
    } else if (max_) {
        return {where(self.lt(*max_), grad, 0)};
    }
    return {grad};
}

// ============================================================
// Gather / scatter / index_select
// ============================================================
// From: tools/autograd/derivatives.yaml. Each op's gradient is another op
// of the family: what gather reads, its backward adds back (scatter_add).

GatherBackward0::GatherBackward0(const Tensor& self, int64_t dim, const Tensor& index)
    : self_shape_(self.shape()), dim_(dim)
{
    index_.save(index);
}

std::string GatherBackward0::name() const { return "GatherBackward0"; }

// Elements read more than once accumulate their gradients
TensorList GatherBackward0::apply(const TensorList& grads) {
    Tensor grad_self = zeros(self_shape_, grads[0].options());
    return {scatter_add_(grad_self, dim_, index_.unpack(), grads[0])};
}


ScatterBackward0::ScatterBackward0(int64_t dim, const Tensor& index, bool has_src)
    : dim_(dim), has_src_(has_src)
{
    index_.save(index);
}

std::string ScatterBackward0::name() const { return "ScatterBackward0"; }

// Overwritten elements of self get no gradient; src gets the gradient of
// the positions it was written to
TensorList ScatterBackward0::apply(const TensorList& grads) {
    const Tensor& index = index_.unpack();
    const Tensor& grad = grads[0];
    Tensor grad_self = scatter(grad, dim_, index, 0);
    if (!has_src_) {
        return {grad_self};
    }
    return {grad_self, gather(grad, dim_, index)};
}


ScatterAddBackward0::ScatterAddBackward0(int64_t dim, const Tensor& index)
    : dim_(dim)
{
    index_.save(index);
}

std::string ScatterAddBackward0::name() const { return "ScatterAddBackward0"; }

TensorList ScatterAddBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    return {grad, gather(grad, dim_, index_.unpack())};
}


IndexSelectBackward0::IndexSelectBackward0(const Tensor& self, int64_t dim, const Tensor& index)
    : self_shape_(self.shape()), dim_(dim)
{
    index_.save(index);
}

std::string IndexSelectBackward0::name() const { return "IndexSelectBackward0"; }

// Rows selected more than once accumulate their gradients (an embedding
// lookup of a repeated token)
TensorList IndexSelectBackward0::apply(const TensorList& grads) {
    Tensor grad_self = zeros(self_shape_, grads[0].options());
    return {index_add_(grad_self, dim_, index_.unpack(), grads[0])};
}


IndexAddBackward0::IndexAddBackward0(int64_t dim, const Tensor& index, const Tensor& source)
    : dim_(dim), source_shape_(source.shape())
{
    index_.save(index);
}

std::string IndexAddBackward0::name() const { return "IndexAddBackward0"; }

TensorList IndexAddBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_source = index_select(grad, dim_, index_.unpack());
    // A 0-d source was added as a single element
    if (source_shape_.empty()) {
        grad_source = grad_source.reshape({});
    }
    return {grad, grad_source};
}


IndexBackward0::IndexBackward0(const Tensor& self, ArrayRef<Tensor> indices)
    : self_shape_(self.shape())
{
    indices_.save(indices);
}

std::string IndexBackward0::name() const { return "IndexBackward0"; }

// From: torch/csrc/autograd/FunctionsManual.cpp (index_backward)
TensorList IndexBackward0::apply(const TensorList& grads) {
    Tensor grad_self = zeros(self_shape_, grads[0].options());
    index_put_(grad_self, indices_.unpack(), grads[0], /*accumulate=*/true);
    return {grad_self};
}


IndexPutBackward0::IndexPutBackward0(
    const Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate
)
    : values_shape_(values.shape()),
      accumulate_(accumulate),
      needs_grad_{self.requires_grad(), values.requires_grad()}
{
    indices_.save(indices);
}

std::string IndexPutBackward0::name() const { return "IndexPutBackward0"; }

// From: tools/autograd/derivatives.yaml (index_put)
// Without accumulate, the positions written by values took nothing from
// self, so their gradient for self is zero
TensorList IndexPutBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    const std::vector<Tensor> indices = indices_.unpack();
    Tensor grad_self, grad_values;
    if (needs_grad_[0]) {
        grad_self = accumulate_
            ? grad
            : index_put(grad, indices, zeros(values_shape_, grad.options()));
    }
    if (needs_grad_[1]) {
        grad_values = unbroadcast(index(grad, indices), values_shape_);
    }
    return {grad_self, grad_values};
}


MaskedFillBackward0::MaskedFillBackward0(const Tensor& self, const Tensor& mask)
    : self_shape_(self.shape())
{
    mask_.save(mask);
}

std::string MaskedFillBackward0::name() const { return "MaskedFillBackward0"; }

TensorList MaskedFillBackward0::apply(const TensorList& grads) {
    return {unbroadcast(masked_fill(grads[0], mask_.unpack(), 0), self_shape_)};
}

// ============================================================
// Shape ops
// ============================================================

ViewBackward0::ViewBackward0(const Tensor& self)
    : self_shape_(self.shape()) {}

std::string ViewBackward0::name() const { return "ViewBackward0"; }

TensorList ViewBackward0::apply(const TensorList& grads) {
    return {grads[0].reshape(self_shape_)};
}


TransposeBackward0::TransposeBackward0(int64_t dim0, int64_t dim1)
    : dim0_(dim0), dim1_(dim1) {}

std::string TransposeBackward0::name() const { return "TransposeBackward0"; }

TensorList TransposeBackward0::apply(const TensorList& grads) {
    return {grads[0].transpose(dim0_, dim1_)};
}


PermuteBackward0::PermuteBackward0(IntArrayRef dims) {
    // Build inverse permutation
    inverse_dims_.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        inverse_dims_[dims[i]] = static_cast<int64_t>(i);
    }
}

std::string PermuteBackward0::name() const { return "PermuteBackward0"; }

TensorList PermuteBackward0::apply(const TensorList& grads) {
    return {grads[0].permute(inverse_dims_)};
}


FlipBackward0::FlipBackward0(IntArrayRef dims) : dims_(dims.begin(), dims.end()) {}

std::string FlipBackward0::name() const { return "FlipBackward0"; }

TensorList FlipBackward0::apply(const TensorList& grads) {
    return {flip(grads[0], dims_)};
}


TrilBackward0::TrilBackward0(int64_t diagonal) : diagonal_(diagonal) {}

std::string TrilBackward0::name() const { return "TrilBackward0"; }

TensorList TrilBackward0::apply(const TensorList& grads) {
    return {tril(grads[0], diagonal_)};
}


TriuBackward0::TriuBackward0(int64_t diagonal) : diagonal_(diagonal) {}

std::string TriuBackward0::name() const { return "TriuBackward0"; }

TensorList TriuBackward0::apply(const TensorList& grads) {
    return {triu(grads[0], diagonal_)};
}


UnsqueezeBackward0::UnsqueezeBackward0(int64_t dim) : dim_(dim) {}

std::string UnsqueezeBackward0::name() const { return "UnsqueezeBackward0"; }

TensorList UnsqueezeBackward0::apply(const TensorList& grads) {
    return {grads[0].squeeze(dim_)};
}


SqueezeBackward0::SqueezeBackward0(const Tensor& self, std::optional<int64_t> dim)
    : self_shape_(self.shape()), has_dim_(dim.has_value()) {}

// torch has one node per overload: squeeze() and squeeze(dim)
std::string SqueezeBackward0::name() const { return has_dim_ ? "SqueezeBackward1" : "SqueezeBackward0"; }

TensorList SqueezeBackward0::apply(const TensorList& grads) {
    return {grads[0].reshape(self_shape_)};
}


ExpandBackward0::ExpandBackward0(const Tensor& self)
    : self_shape_(self.shape()) {}

std::string ExpandBackward0::name() const { return "ExpandBackward0"; }

TensorList ExpandBackward0::apply(const TensorList& grads) {
    return {unbroadcast(grads[0], self_shape_)};
}


RepeatBackward0::RepeatBackward0(const Tensor& self, IntArrayRef repeats)
    : self_shape_(self.shape()), repeats_(repeats.begin(), repeats.end()) {}

std::string RepeatBackward0::name() const { return "RepeatBackward0"; }

// From: torch/csrc/autograd/FunctionsManual.cpp (repeat_backward)
// The gradient [r0 * a0, r1 * a1] seen as [r0, a0, r1, a1]: summing the r
// dims adds the gradients of every copy of each element (with a repeat of 0
// the grad is empty and the sum gives zeros)
TensorList RepeatBackward0::apply(const TensorList& grads) {
    const int64_t ndim = static_cast<int64_t>(repeats_.size());
    const int64_t new_dims = ndim - static_cast<int64_t>(self_shape_.size());
    DimVector split_shape;
    DimVector repeat_dims;
    for (int64_t d = 0; d < ndim; ++d) {
        repeat_dims.push_back(2 * d);
        split_shape.push_back(repeats_[d]);
        split_shape.push_back(d < new_dims ? 1 : self_shape_[d - new_dims]);
    }
    return {grads[0].reshape(split_shape).sum(repeat_dims).reshape(self_shape_)};
}


SelectBackward0::SelectBackward0(const Tensor& self, int64_t dim, int64_t index)
    : self_shape_(self.shape()), dim_(dim), index_(index) {}

std::string SelectBackward0::name() const { return "SelectBackward0"; }

TensorList SelectBackward0::apply(const TensorList& grads) {
    Tensor result = zeros(self_shape_, grads[0].options());
    result.select(dim_, index_).copy_(grads[0]);
    return {result};
}


SliceBackward0::SliceBackward0(
    const Tensor& self, int64_t dim, int64_t start, int64_t stop, int64_t step)
    : self_shape_(self.shape()), dim_(dim), start_(start), stop_(stop), step_(step) {}

std::string SliceBackward0::name() const { return "SliceBackward0"; }

TensorList SliceBackward0::apply(const TensorList& grads) {
    Tensor result = zeros(self_shape_, grads[0].options());
    result.slice(dim_, start_, stop_, step_).copy_(grads[0]);
    return {result};
}


// ============================================================
// Copies
// ============================================================

std::string CloneBackward0::name() const { return "CloneBackward0"; }

TensorList CloneBackward0::apply(const TensorList& grads) {
    return {grads[0]};
}

std::string PinMemoryBackward0::name() const { return "PinMemoryBackward0"; }

TensorList PinMemoryBackward0::apply(const TensorList& grads) {
    return {grads[0]};
}

ToCopyBackward0::ToCopyBackward0(const Tensor& self)
    : self_dtype_(self.dtype()), self_device_(self.device()) {}

std::string ToCopyBackward0::name() const { return "ToCopyBackward0"; }

TensorList ToCopyBackward0::apply(const TensorList& grads) {
    return {grads[0].to(self_device_).to(self_dtype_)};
}

} // namespace raxis::autograd