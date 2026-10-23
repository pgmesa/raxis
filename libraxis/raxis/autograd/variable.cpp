
#include <raxis/autograd/variable.h>

#include <cstdint>
#include <memory>
#include <sstream>

#include <raxis/util/exception.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/autograd/graph.h>
#include <raxis/autograd/engine.h>
#include <raxis/autograd/autograd_meta.h>
#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/detail/storage_impl.h>
#include <raxis/autograd/saved_variable.h>
#include <raxis/autograd/functions/tensor.h>

#include <raxis/ops/copy.h>
#include <raxis/ops/binary_ops.h>
#include <raxis/ops/reduce_ops.h>


namespace raxis::autograd {

// From: torch/csrc/autograd/functions/accumulate_grad.cpp
// The highest sequence_nr: once ready, a leaf runs before any other node, so
// its gradient leaves the engine's buffers as early as possible.
AccumulateGrad::AccumulateGrad(const Tensor& variable)
    : Node(/*sequence_nr=*/UINT64_MAX), variable_(variable) {
    RAXIS_INTERNAL_ASSERT(variable.requires_grad(),
        "AccumulateGrad: variable does not require grad");
    set_num_inputs(1);
}

std::string AccumulateGrad::name() const {
    return "AccumulateGrad";
}

// From: torch/csrc/autograd/utils/grad_layout_contract.h
// Gradient layout contract: .grad has the strides of the variable when the
// variable is dense (a transposed weight gets a transposed grad), else it is
// contiguous. Optimizers then update param and grad elementwise in the same
// memory order, and the in-place accumulation of later backwards keeps the
// layout.
static bool obeys_layout_contract(const Tensor& grad, const Tensor& variable) {
    if (!variable.is_non_overlapping_and_dense()) {
        return grad.is_contiguous();
    }
    IntArrayRef grad_shape = grad.shape();
    IntArrayRef grad_strides = grad.strides();
    IntArrayRef variable_strides = variable.strides();
    for (size_t i = 0; i < grad_shape.size(); i++) {
        if (grad_shape[i] != 1) {
            if (grad_strides[i] != variable_strides[i]) return false;
        } else if (grad_strides[i] == 0) {
            // A size 1 dim does not affect the layout, but stride 0 means the
            // grad is a broadcast view of something else
            return false;
        }
    }
    return true;
}

static Tensor clone_obey_contract(const Tensor& grad, const Tensor& variable) {
    if (!variable.is_non_overlapping_and_dense()) {
        return grad.clone(MemoryFormat::Contiguous);
    }
    Tensor out = empty_strided(variable.shape(), variable.strides(), grad.dtype());
    out.copy_(grad);
    return out;
}

// From: torch/csrc/autograd/functions/accumulate_grad.h (accumulateGrad)
TensorList AccumulateGrad::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    RAXIS_CHECK(grad.defined(), "AccumulateGrad received undefined gradient");

    if (Tensor variable_grad = variable_.grad(); variable_grad.defined()) {
        // In place, as PyTorch: .grad keeps its memory (and its layout)
        // across backward passes, and references to it see the sum
        variable_grad.add_(grad);
    } else {
        // .grad must own its memory: optimizers update it in place, and the
        // incoming grad can be a broadcast view or be shared with other nodes
        // or with the gradient passed by the user to backward(). PyTorch only
        // checks the impl: a view of the user's gradient (from reshape) is
        // stolen there and the next backward adds into the user's tensor.
        bool can_steal = grad.use_count() == 1
            && grad.unsafe_get_tensor_impl()->storage().use_count() == 1
            && obeys_layout_contract(grad, variable_);
        variable_.set_grad(can_steal ? grad : clone_obey_contract(grad, variable_));
    }

    return {};
}


// From: torch/csrc/autograd/variable.cpp (impl::grad_accumulator)
// A leaf used by k ops gets a single AccumulateGrad: the engine sums the k
// gradients in its input buffer and .grad is updated once, instead of k
// times by k nodes (one allocation each). The node lives as long as some
// graph uses it; the leaf only keeps a weak reference.
NodePtr grad_accumulator(const Tensor& leaf) {
    AutogradMeta* meta = leaf.unsafe_get_tensor_impl()->autograd_meta();
    RAXIS_INTERNAL_ASSERT(meta && meta->requires_grad && !meta->grad_fn,
        "grad_accumulator() should be only called on leaves that require grad");
    if (NodePtr acc = meta->grad_accumulator.lock()) {
        return acc;
    }
    auto acc = std::make_shared<AccumulateGrad>(leaf);
    meta->grad_accumulator = acc;
    return acc;
}


void set_autograd_meta(Tensor& result, NodePtr grad_fn, uint32_t output_nr) {
    result.set_requires_grad(true);
    result.set_grad_fn(grad_fn);
    result.set_output_nr(output_nr);
}


Edge gradient_edge(const Tensor& input) {
    if (!input.defined()) {
        return Edge{};
    }
    if (NodePtr grad_fn = input.grad_fn()) {
        return Edge{std::move(grad_fn), input.output_nr()};
    }
    // A leaf of its own: a view made under no_grad requires grad through its
    // base but has no AccumulateGrad (PyTorch returns no edge for it too)
    AutogradMeta* meta = input.unsafe_get_tensor_impl()->autograd_meta();
    if (meta && meta->requires_grad) {
        return Edge{grad_accumulator(input), 0};
    }
    return Edge{};
}


void set_next_edges(const NodePtr& grad_fn, const Tensor& input) {
    grad_fn->add_next_edge(gradient_edge(input));
}

void set_next_edges(const NodePtr& grad_fn, const Tensor& input0, const Tensor& input1) {
    grad_fn->add_next_edge(gradient_edge(input0));
    grad_fn->add_next_edge(gradient_edge(input1));
}

void set_next_edges(const NodePtr& grad_fn, ArrayRef<Tensor> inputs) {
    for (const auto& input : inputs) {
        grad_fn->add_next_edge(gradient_edge(input));
    }
}


// From: torch/csrc/autograd/variable.cpp (handle_view_on_rebase)
[[noreturn]] static void view_cannot_be_modified(const Tensor& view, bool indirect) {
    detail::TensorImpl* impl = view.unsafe_get_tensor_impl();
    AutogradMeta* meta = impl->autograd_meta();
    Node* grad_fn = meta ? meta->grad_fn.get() : nullptr;
    const char* modified = indirect
        ? "its base or another view of its base has been" : "is being";
    std::ostringstream msg;
    if (impl->creation_meta() == CreationMeta::NoGradMode || !grad_fn) {
        if (grad_fn) {
            msg << "Output " << meta->output_nr << " of " << grad_fn->forward_op_name()
                << " is a view of a view which was created in";
        } else {
            msg << "A view was created in";
        }
        msg << " no_grad mode and " << modified << " modified inplace with grad mode enabled."
            << " Given that this use case is ambiguous and error-prone, it is forbidden."
            << " You can clarify your code by moving both the view and the inplace either both"
            << " inside the no_grad block (if you don't want the inplace to be tracked) or both"
            << " outside (if you want the inplace to be tracked).";
    } else {
        msg << "Output " << meta->output_nr << " of " << grad_fn->forward_op_name() << " is a view and "
            << modified << " modified inplace. This view is the output of a function that"
            << " returns multiple views. Such functions do not allow the output views to be"
            << " modified inplace. You should replace the inplace operation by an out-of-place one.";
    }
    RAXIS_CHECK(false, msg.str());
}

void check_inplace(const Tensor& self, bool requires_grad) {
    if (!requires_grad) return;
    detail::TensorImpl* impl = self.unsafe_get_tensor_impl();
    if (impl->is_view()) {
        if (impl->creation_meta() != CreationMeta::Default) {
            view_cannot_be_modified(self, /*indirect=*/false);
        }
        RAXIS_CHECK(!(self.requires_grad() && self.base().is_leaf()),
            "a view of a leaf Variable that requires grad is being used in an in-place operation.");
    }
    RAXIS_CHECK(!(self.requires_grad() && self.is_leaf()),
        "a leaf Variable that requires grad is being used in an in-place operation.");
}

void save_original_self(Node& grad_fn, const Tensor& self) {
    Tensor original;
    for (SavedVariable* saved : grad_fn.saved_variables()) {
        if (saved->holds(self)) {
            if (!original.defined()) {
                original = ops::clone(self);
            }
            saved->save(original);
        }
    }
}

void bump_version(const Tensor& self) {
    self.unsafe_get_tensor_impl()->storage()->bump_version();
}

void rebase_history(const Tensor& self, NodePtr grad_fn) {
    detail::TensorImpl* impl = self.unsafe_get_tensor_impl();
    if (impl->is_view()) {
        Tensor base = self.base();
        auto copy_slices = std::make_shared<CopySlices>(base, self, std::move(grad_fn));
        set_autograd_meta(base, copy_slices);
        refresh_view_grad_fn(self);
        return;
    }
    Tensor result = self;
    set_autograd_meta(result, std::move(grad_fn));
}

void finish_inplace(const Tensor& self, NodePtr grad_fn) {
    bump_version(self);
    if (grad_fn) {
        rebase_history(self, std::move(grad_fn));
    }
}

void refresh_view_grad_fn(const Tensor& view) {
    detail::TensorImpl* impl = view.unsafe_get_tensor_impl();
    AutogradMeta* meta = impl->autograd_meta();
    Tensor base = view.base();
    // Nothing to describe: no history before, and none to take from the base
    if ((!meta || !meta->grad_fn) && !base.requires_grad()) {
        return;
    }
    if (impl->creation_meta() != CreationMeta::Default) {
        view_cannot_be_modified(view, /*indirect=*/true);
    }
    auto grad_fn = std::make_shared<AsStridedBackward0>(base, view);
    grad_fn->set_num_inputs(1);
    set_next_edges(grad_fn, base);
    // Through the meta: Tensor::set_grad_fn would look at the version again
    if (!meta) {
        impl->set_autograd_meta(std::make_unique<AutogradMeta>());
        meta = impl->autograd_meta();
    }
    meta->grad_fn = std::move(grad_fn);
    meta->output_nr = 0;
    meta->requires_grad = true;
    impl->set_view_version(impl->storage()->version());
}

void mark_multi_output_views(ArrayRef<Tensor> views) {
    if (!GradMode::is_enabled()) return;
    for (const Tensor& view : views) {
        if (view.is_view()) {
            view.unsafe_get_tensor_impl()->set_creation_meta(CreationMeta::MultiOutputNode);
        }
    }
}


Tensor unbroadcast(const Tensor& grad, IntArrayRef original_shape) {
    int64_t grad_dim = grad.ndim();
    int64_t orig_dim = static_cast<int64_t>(original_shape.size());

    if (grad_dim == orig_dim && grad.shape() == original_shape) {
        return grad;
    }

    // Collect all dims that need reduction
    DimVector reduce_dims;

    // Extra leading dims added by broadcasting
    int64_t leading = grad_dim - orig_dim;
    for (int64_t i = 0; i < leading; ++i) {
        reduce_dims.push_back(i);
    }

    // Dims that were broadcast from 1 -> N
    for (int64_t i = 0; i < orig_dim; ++i) {
        if (original_shape[i] == 1 && grad.shape()[leading + i] != 1) {
            reduce_dims.push_back(leading + i);
        }
    }

    if (reduce_dims.empty()) {
        return grad;
    }

    // Single sum call for all broadcast dims, then reshape
    Tensor result = grad.sum(reduce_dims, /*keepdim=*/true);
    return result.reshape(original_shape);
}

} // namespace raxis::autograd