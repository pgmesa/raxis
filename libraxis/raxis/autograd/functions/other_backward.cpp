
#include <raxis/autograd/functions/other_backward.h>

#include <cmath>

#include <raxis/ops/cat_ops.h>
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


UnsqueezeBackward0::UnsqueezeBackward0(int64_t dim) : dim_(dim) {}

std::string UnsqueezeBackward0::name() const { return "UnsqueezeBackward0"; }

TensorList UnsqueezeBackward0::apply(const TensorList& grads) {
    return {grads[0].squeeze(dim_)};
}


SqueezeBackward0::SqueezeBackward0(const Tensor& self, std::optional<int64_t> dim)
    : self_shape_(self.shape()) {}

std::string SqueezeBackward0::name() const { return "SqueezeBackward0"; }

TensorList SqueezeBackward0::apply(const TensorList& grads) {
    return {grads[0].reshape(self_shape_)};
}


ExpandBackward0::ExpandBackward0(const Tensor& self)
    : self_shape_(self.shape()) {}

std::string ExpandBackward0::name() const { return "ExpandBackward0"; }

TensorList ExpandBackward0::apply(const TensorList& grads) {
    return {unbroadcast(grads[0], self_shape_)};
}


SelectBackward0::SelectBackward0(const Tensor& self, int64_t dim, int64_t index)
    : self_shape_(self.shape()), dim_(dim), index_(index) {}

std::string SelectBackward0::name() const { return "SelectBackward0"; }

TensorList SelectBackward0::apply(const TensorList& grads) {
    Tensor result = zeros(self_shape_, grads[0].dtype());
    result.select(dim_, index_).copy_(grads[0]);
    return {result};
}


SliceBackward0::SliceBackward0(
    const Tensor& self, int64_t dim, int64_t start, int64_t stop, int64_t step)
    : self_shape_(self.shape()), dim_(dim), start_(start), stop_(stop), step_(step) {}

std::string SliceBackward0::name() const { return "SliceBackward0"; }

TensorList SliceBackward0::apply(const TensorList& grads) {
    Tensor result = zeros(self_shape_, grads[0].dtype());
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

ToCopyBackward0::ToCopyBackward0(const Tensor& self) : self_dtype_(self.dtype()) {}

std::string ToCopyBackward0::name() const { return "ToCopyBackward0"; }

TensorList ToCopyBackward0::apply(const TensorList& grads) {
    return {grads[0].to(self_dtype_)};
}

} // namespace raxis::autograd