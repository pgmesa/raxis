#include <raxis/autograd/functions/tensor.h>

#include <utility>

#include <raxis/ops/copy.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_shape.h>
#include <raxis/autograd/grad_mode.h>
#include <raxis/autograd/variable.h>


namespace raxis::autograd {

namespace {

// The elements of `geometry` within a buffer laid out as `base`, as
// PyTorch's as_strided. With create_graph it is recorded: writing it goes
// to the history of buffer (CopySlices), and reading it takes its slice of
// the gradient of buffer (AsStridedBackward0).
Tensor view_in(const Tensor& buffer, const TensorGeometry& base, const TensorGeometry& geometry,
               IntArrayRef shape) {
    Tensor view = make_view(buffer, buffer.storage_offset() + geometry.storage_offset - base.storage_offset,
                            shape, geometry.strides);
    if (GradMode::is_enabled() && buffer.requires_grad()) {
        refresh_view_grad_fn(view);
    }
    return view;
}

} // namespace


// From: torch/csrc/autograd/functions/tensor.cpp (CopyBackwards)
CopyBackwards::CopyBackwards(const Tensor& self, const Tensor& src)
    : needs_grad_{self.requires_grad(), src.requires_grad()},
      src_dtype_(src.dtype()),
      src_shape_(src.shape()) {}

std::string CopyBackwards::name() const { return "CopyBackwards"; }

TensorList CopyBackwards::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_self, grad_src;
    if (!grad.defined()) {
        return {grad_self, grad_src};
    }
    if (needs_grad_[0]) {
        grad_self = zeros_like(grad, {}, MemoryFormat::Contiguous);
    }
    if (needs_grad_[1]) {
        grad_src = unbroadcast(grad.to(src_dtype_), src_shape_);
    }
    return {grad_self, grad_src};
}


CopySlices::CopySlices(const Tensor& base, const Tensor& view, NodePtr fn)
    : base_(base), view_(view), fn_(std::move(fn)) {
    set_num_inputs(1);
    // Before base takes this node as its grad_fn
    add_next_edge(gradient_edge(base));
    for (size_t i = 1; i < fn_->num_outputs(); i++) {
        add_next_edge(fn_->next_edges()[i]);
    }
}

std::string CopySlices::name() const { return "CopySlices"; }

// From: torch/csrc/autograd/functions/tensor.cpp (CopySlices::apply_impl)
TensorList CopySlices::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    TensorList grad_inputs(num_outputs());
    if (!grad.defined()) {
        return grad_inputs;
    }

    // The base's gradient, laid out as the base so the view's geometry
    // finds its slice in it
    Tensor result = empty_strided(base_.shape, base_.strides, grad.options());
    result.copy_(grad);
    Tensor grad_slice = view_in(result, base_, view_, view_.shape);

    // fn may save its gradient input, and grad_slice is written below
    TensorList res = fn_->apply({grad_slice.clone(MemoryFormat::Contiguous)});
    for (size_t i = 0; i < res.size(); i++) {
        if (!res[i].defined()) continue;
        if (i == 0) {
            // The slice now holds the gradient through the op
            grad_slice.copy_(res[0]);
            grad_inputs[0] = std::move(result);
        } else {
            grad_inputs[i] = std::move(res[i]);
        }
    }
    return grad_inputs;
}

void CopySlices::release_variables() {
    Node::release_variables();
    fn_->release_variables();
}


AsStridedBackward0::AsStridedBackward0(const Tensor& base, const Tensor& view)
    : base_(base), view_(view) {}

std::string AsStridedBackward0::name() const { return "AsStridedBackward0"; }

// From: torch/csrc/autograd/FunctionsManual.cpp (as_strided_backward)
// PyTorch handles any geometry, including views whose elements overlap in
// arbitrary ways. Views here only overlap through expanded dims (stride 0),
// where every element is the same element of the base: their gradients add
// up, so those dims are summed first.
TensorList AsStridedBackward0::apply(const TensorList& grads) {
    Tensor grad = grads[0];
    if (!grad.defined()) {
        return {Tensor()};
    }
    DimVector reduce_dims;
    DimVector slot_shape = view_.shape;
    for (size_t d = 0; d < view_.shape.size(); d++) {
        if (view_.strides[d] == 0 && view_.shape[d] > 1) {
            reduce_dims.push_back(static_cast<int64_t>(d));
            slot_shape[d] = 1;
        }
    }
    if (!reduce_dims.empty()) {
        grad = grad.sum(reduce_dims, /*keepdim=*/true);
    }

    Tensor result = empty_strided(base_.shape, base_.strides, grad.options());
    result.zero_();
    Tensor slot = view_in(result, base_, view_, slot_shape);
    slot.copy_(grad);
    return {result};
}

} // namespace raxis::autograd
