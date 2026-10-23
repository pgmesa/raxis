
#include <raxis/ops/copy.h>
#include <raxis/ops/fill.h>
#include <raxis/ops/distribution_ops.h>
#include <raxis/ops/cat_ops.h>
#include <raxis/ops/scalar_ops.h>
#include <raxis/ops/tensor_compare.h>
#include <raxis/ops/tensor_advanced_indexing.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_shape.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/type_promotion.h>

#include <raxis/autograd/grad_mode.h>
#include <raxis/autograd/variable.h>
#include <raxis/autograd/functions/common.h>
#include <raxis/autograd/functions/other_backward.h>
#include <raxis/autograd/functions/tensor.h>


namespace raxis {

// ============================================================
// Cat / Stack / Unbind
// ============================================================

Tensor cat(ArrayRef<Tensor> tensors, int64_t dim) {
    Tensor result = ops::cat(tensors, dim);
    if (autograd::requires_autograd(tensors)) {
        auto grad_fn = std::make_shared<autograd::CatBackward0>(tensors, dim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, tensors);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor stack(ArrayRef<Tensor> tensors, int64_t dim) {
    Tensor result = ops::stack(tensors, dim);
    if (autograd::requires_autograd(tensors)) {
        auto grad_fn = std::make_shared<autograd::StackBackward0>(
            static_cast<int64_t>(tensors.size()), dim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, tensors);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

std::vector<Tensor> unbind(const Tensor& self, int64_t dim) {
    std::vector<Tensor> results = ops::unbind(self, dim);
    autograd::mark_multi_output_views(results);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::UnbindBackward0>(
            self, dim, static_cast<int64_t>(results.size()));
        grad_fn->set_num_inputs(static_cast<int64_t>(results.size()));
        autograd::set_next_edges(grad_fn, self);

        for (uint32_t i = 0; i < results.size(); ++i) {
            autograd::set_autograd_meta(results[i], grad_fn, /*output_nr=*/i);
        }
    }
    return results;
}

// One multi-output node for split, split_with_sizes and chunk
std::vector<Tensor> split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
    std::vector<Tensor> results = ops::split_with_sizes(self, split_sizes, dim);
    autograd::mark_multi_output_views(results);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::SplitWithSizesBackward0>(
            self, split_sizes, wrap_dim(dim, self.ndim()));
        grad_fn->set_num_inputs(static_cast<int64_t>(results.size()));
        autograd::set_next_edges(grad_fn, self);

        for (uint32_t i = 0; i < results.size(); ++i) {
            autograd::set_autograd_meta(results[i], grad_fn, /*output_nr=*/i);
        }
    }
    return results;
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
    RAXIS_CHECK(self.ndim() != 0, "split expects at least a 1-dimensional tensor");
    int64_t dim_size = self.shape()[wrap_dim(dim, self.ndim())];
    return split_with_sizes(self, ops::split_sizes_for(dim_size, split_size), dim);
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
    RAXIS_CHECK(self.ndim() > 0, "chunk expects at least a 1-dimensional tensor");
    RAXIS_CHECK(chunks > 0, "chunk expects `chunks` to be greater than 0, got: ", chunks);
    int64_t dim_size = self.shape()[wrap_dim(dim, self.ndim())];
    int64_t split_size = (dim_size + chunks - 1) / chunks;
    if (split_size == 0 && dim_size == 0) {
        DimVector sizes(chunks, 0);
        return split_with_sizes(self, sizes, dim);
    }
    return split(self, split_size, dim);
}

// ============================================================
// Shape ops
// ============================================================

Tensor reshape(const Tensor& self, IntArrayRef shape) {
    Tensor result = ops::reshape(self, shape);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ViewBackward0>(self);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor view(const Tensor& self, IntArrayRef shape) {
    Tensor result = ops::view(self, shape);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ViewBackward0>(self);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
    Tensor result = ops::transpose(self, dim0, dim1);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::TransposeBackward0>(dim0, dim1);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor permute(const Tensor& self, IntArrayRef dims) {
    Tensor result = ops::permute(self, dims);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::PermuteBackward0>(dims);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
    Tensor result = ops::unsqueeze(self, dim);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::UnsqueezeBackward0>(dim);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor squeeze(const Tensor& self, std::optional<int64_t> dim_opt) {
    Tensor result = ops::squeeze(self, dim_opt);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::SqueezeBackward0>(self, dim_opt);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
    Tensor result = ops::select(self, dim, index);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::SelectBackward0>(self, dim, index);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor slice(const Tensor& self, int64_t dim, int64_t start, int64_t stop, int64_t step) {
    Tensor result = ops::slice(self, dim, start, stop, step);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::SliceBackward0>(self, dim, start, stop, step);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor expand(const Tensor& self, IntArrayRef shape) {
    Tensor result = ops::expand(self, shape);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ExpandBackward0>(self);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor broadcast_to(const Tensor& self, IntArrayRef shape) {
    return expand(self, shape);
}


// ============================================================
// where / clamp
// ============================================================

Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
    Tensor result = ops::where(condition, self, other);
    if (autograd::requires_autograd(self, other)) {
        auto grad_fn = std::make_shared<autograd::WhereBackward0>(condition, self, other);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self, other);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor where(const Tensor& condition, const Tensor& self, Scalar other) {
    return where(condition, self, wrapped_scalar_tensor(other, self.device()));
}

Tensor where(const Tensor& condition, Scalar self, const Tensor& other) {
    return where(condition, wrapped_scalar_tensor(self, other.device()), other);
}

// From: aten/src/ATen/native/TensorCompare.cpp
// Two scalars give a tensor of their promoted dtype (float scalars give the
// default float dtype) with the shape of the condition
Tensor where(const Tensor& condition, Scalar self, Scalar other) {
    Tensor self_t = wrapped_scalar_tensor(self, condition.device());
    Tensor other_t = wrapped_scalar_tensor(other, condition.device());
    Dtype dtype = result_type({self_t, other_t});
    return ops::where(condition, self_t.to(dtype), other_t.to(dtype));
}

Tensor clamp(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    Tensor result = ops::clamp(self, min, max);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ClampBackward1>(self, min, max);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor clamp(const Tensor& self, const std::optional<Tensor>& min, const std::optional<Tensor>& max) {
    Tensor result = ops::clamp(self, min, max);
    const Tensor min_t = min.value_or(Tensor());
    const Tensor max_t = max.value_or(Tensor());
    if (autograd::requires_autograd({self, min_t, max_t})) {
        auto grad_fn = std::make_shared<autograd::ClampBackward0>(self, min_t, max_t);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, {self, min_t, max_t});
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor& clamp_(Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    return autograd::inplace_op<autograd::ClampBackward1>(self, {self},
        [&] { ops::clamp_(self, min, max); }, self, min, max);
}

Tensor clamp_min(const Tensor& self, Scalar min) {
    return clamp(self, min, std::nullopt);
}

Tensor clamp_max(const Tensor& self, Scalar max) {
    return clamp(self, std::nullopt, max);
}

// ============================================================
// Gather / scatter / index_select
// ============================================================

static int64_t wrap_index_dim(int64_t dim, const Tensor& self) {
    return wrap_dim(dim, std::max<int64_t>(self.ndim(), 1));
}

Tensor gather(const Tensor& self, int64_t dim, const Tensor& index) {
    Tensor result = ops::gather(self, dim, index);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::GatherBackward0>(
            self, wrap_index_dim(dim, self), index);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    Tensor result = ops::scatter(self, dim, index, src);
    if (autograd::requires_autograd(self, src)) {
        auto grad_fn = std::make_shared<autograd::ScatterBackward0>(
            wrap_index_dim(dim, self), index, /*has_src=*/true);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self, src);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    Tensor result = ops::scatter(self, dim, index, value);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ScatterBackward0>(
            wrap_index_dim(dim, self), index, /*has_src=*/false);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor scatter_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    Tensor result = ops::scatter_add(self, dim, index, src);
    if (autograd::requires_autograd(self, src)) {
        auto grad_fn = std::make_shared<autograd::ScatterAddBackward0>(wrap_index_dim(dim, self), index);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self, src);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor& scatter_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return autograd::inplace_op<autograd::ScatterBackward0>(self, {self, src},
        [&] { ops::scatter_(self, dim, index, src); },
        wrap_index_dim(dim, self), index, /*has_src=*/true);
}

Tensor& scatter_(Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    return autograd::inplace_op<autograd::ScatterBackward0>(self, {self},
        [&] { ops::scatter_(self, dim, index, value); },
        wrap_index_dim(dim, self), index, /*has_src=*/false);
}

Tensor& scatter_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    return autograd::inplace_op<autograd::ScatterAddBackward0>(self, {self, src},
        [&] { ops::scatter_add_(self, dim, index, src); },
        wrap_index_dim(dim, self), index);
}

Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
    Tensor result = ops::index_select(self, dim, index);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::IndexSelectBackward0>(
            self, wrap_index_dim(dim, self), index);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor index_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    Tensor result = ops::index_add(self, dim, index, source);
    if (autograd::requires_autograd(self, source)) {
        auto grad_fn = std::make_shared<autograd::IndexAddBackward0>(
            wrap_index_dim(dim, self), index, source);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self, source);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor& index_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    return autograd::inplace_op<autograd::IndexAddBackward0>(self, {self, source},
        [&] { ops::index_add_(self, dim, index, source); },
        wrap_index_dim(dim, self), index, source);
}

// ============================================================
// Advanced indexing
// ============================================================

Tensor index(const Tensor& self, ArrayRef<Tensor> indices) {
    Tensor result = ops::index(self, indices);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::IndexBackward0>(self, indices);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor index_put(const Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate) {
    Tensor result = ops::index_put(self, indices, values, accumulate);
    if (autograd::requires_autograd(self, values)) {
        auto grad_fn = std::make_shared<autograd::IndexPutBackward0>(self, indices, values, accumulate);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self, values);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor& index_put_(Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate) {
    return autograd::inplace_op<autograd::IndexPutBackward0>(self, {self, values},
        [&] { ops::index_put_(self, indices, values, accumulate); },
        self, indices, values, accumulate);
}

// Integer output: never differentiable
Tensor nonzero(const Tensor& self) {
    return ops::nonzero(self);
}

Tensor masked_fill(const Tensor& self, const Tensor& mask, Scalar value) {
    Tensor result = ops::masked_fill(self, mask, value);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::MaskedFillBackward0>(self, mask);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor& masked_fill_(Tensor& self, const Tensor& mask, Scalar value) {
    return autograd::inplace_op<autograd::MaskedFillBackward0>(self, {self},
        [&] { ops::masked_fill_(self, mask, value); }, self, mask);
}

// ============================================================
// Copies
// ============================================================

Tensor& copy_(Tensor& self, const Tensor& src) {
    return autograd::inplace_op<autograd::CopyBackwards>(self, {self, src},
        [&] { copy(src, self); }, self, src);
}

// In-place fills: the old values of self are gone, so its gradient is zero
// (PyTorch names the nodes after the op)
Tensor& fill_(Tensor& self, Scalar value) {
    return autograd::inplace_op<autograd::ZeroBackward0>(self, {self},
        [&] { ops::fill_(self, value); }, "FillBackward2");
}

Tensor& zero_(Tensor& self) {
    return autograd::inplace_op<autograd::ZeroBackward0>(self, {self},
        [&] { ops::zero_(self); }, "ZeroBackward0");
}

Tensor& uniform_(Tensor& self, double low, double high) {
    return autograd::inplace_op<autograd::ZeroBackward0>(self, {self},
        [&] { ops::uniform_(self, low, high); }, "UniformBackward0");
}

Tensor& normal_(Tensor& self, double mean, double std) {
    return autograd::inplace_op<autograd::ZeroBackward0>(self, {self},
        [&] { ops::normal_(self, mean, std); }, "NormalBackward0");
}

Tensor& randint_(Tensor& self, int64_t low, int64_t high) {
    return autograd::inplace_op<autograd::ZeroBackward0>(self, {self},
        [&] { ops::randint_(self, low, high); }, "RandomBackward0");
}

Tensor clone(const Tensor& self, MemoryFormat memory_format) {
    Tensor result = ops::clone(self, memory_format);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::CloneBackward0>();
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor to_copy(const Tensor& self, Dtype dtype) {
    Tensor result = ops::to_copy(self, dtype);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::ToCopyBackward0>(self);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

} // namespace raxis