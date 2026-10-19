
#include <raxis/ops/copy.h>
#include <raxis/ops/cat_ops.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_shape.h>

#include <raxis/autograd/grad_mode.h>
#include <raxis/autograd/variable.h>
#include <raxis/autograd/functions/other_backward.h>


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

Tensor broadcast_to(const Tensor& self, IntArrayRef shape) {
    Tensor result = ops::broadcast_to(self, shape);
    if (autograd::requires_autograd(self)) {
        auto grad_fn = std::make_shared<autograd::BroadcastToBackward0>(self);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}


// ============================================================
// Copies
// ============================================================

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