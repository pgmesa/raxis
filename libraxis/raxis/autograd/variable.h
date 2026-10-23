#pragma once

#include <memory>
#include <utility>

#include <raxis/export.h>
#include <raxis/tensor/tensor.h>
#include <raxis/autograd/graph.h>
#include <raxis/autograd/engine.h>
#include <raxis/autograd/grad_mode.h>


namespace raxis::autograd {

// Check if any tensor in the pack requires autograd tracking
inline bool requires_autograd(const Tensor& a) {
    return GradMode::is_enabled() && a.requires_grad();
}

inline bool requires_autograd(const Tensor& a, const Tensor& b) {
    return GradMode::is_enabled() && (a.requires_grad() || b.requires_grad());
}

inline bool requires_autograd(ArrayRef<Tensor> tensors) {
    if (!GradMode::is_enabled()) return false;
    for (const auto& t : tensors) {
        if (t.requires_grad()) return true;
    }
    return false;
}


// Leaf accumulation node: stores gradient into the variable's .grad field
class RAXIS_API AccumulateGrad : public Node {
public:
    explicit AccumulateGrad(const Tensor& variable);
    std::string name() const override;
    TensorList apply(const TensorList& grads) override;
    // The hooks registered on the leaf
    void call_tensor_pre_hooks(TensorList& grads) override;

    const Tensor& variable() const { return variable_; }

private:
    Tensor variable_;
};


// The AccumulateGrad node of a leaf that requires grad, created on first use
RAXIS_API NodePtr grad_accumulator(const Tensor& leaf);


// From: torch/_tensor.py (Tensor.register_hook)
// hook(grad) runs when the gradient of self is computed, and may return a
// new gradient (an undefined tensor keeps it)
RAXIS_API RemovableHandle register_hook(const Tensor& self, TensorHook hook);

// From: torch/_tensor.py (Tensor.register_post_accumulate_grad_hook)
// hook(self) runs on a leaf after its .grad was updated
RAXIS_API RemovableHandle register_post_accumulate_grad_hook(const Tensor& self, PostAccumulateGradHook hook);


// Wire the autograd graph for an output tensor.
// Creates edges from grad_fn back to each input's grad_fn (or AccumulateGrad for leaves).
RAXIS_API void set_autograd_meta(
    Tensor& result,
    NodePtr grad_fn,
    uint32_t output_nr = 0);

// Convenience: wire edges for a single input
RAXIS_API void set_next_edges(
    const NodePtr& grad_fn,
    const Tensor& input);

// Convenience: wire edges for two inputs (binary ops)
RAXIS_API void set_next_edges(
    const NodePtr& grad_fn,
    const Tensor& input0,
    const Tensor& input1);

// Convenience: wire edges for a list of inputs (cat, stack, etc.)
RAXIS_API void set_next_edges(
    const NodePtr& grad_fn,
    ArrayRef<Tensor> inputs);


// Reduce a gradient to match an original shape that was broadcast.
// Sums along dimensions that were broadcast or added.
RAXIS_API Tensor unbroadcast(const Tensor& grad, IntArrayRef original_shape);


// ============================================================
// In-place ops and views
// ============================================================
//
// An in-place op with autograd, as PyTorch's generated code does it:
//
//   check_inplace(self, requires_grad);
//   grad_fn = <backward node of the op, from the inputs>;
//   save_original_self(*grad_fn, self);   // before the kernel writes self
//   <kernel>;
//   finish_inplace(self, grad_fn);        // new version, new history

// From: torch/csrc/autograd/variable.cpp (gradient_edge)
// Where the gradient of `t` goes: its grad_fn, or the AccumulateGrad of a
// leaf that requires grad (an invalid edge otherwise)
RAXIS_API Edge gradient_edge(const Tensor& t);

// From: torch/csrc/autograd/VariableTypeUtils.h (check_inplace)
// A leaf that requires grad cannot be modified in place while autograd
// records (its .grad would no longer match its values), nor can a view of
// one, nor a view whose CreationMeta forbids it
RAXIS_API void check_inplace(const Tensor& self, bool requires_grad);

// From: the original_self of the in-place ops of PyTorch's VariableType
// If grad_fn saved self, which the kernel is about to overwrite, it keeps a
// copy of the current values instead. PyTorch's codegen clones self for the
// formulas that use it; here the node is checked, so any node works.
RAXIS_API void save_original_self(Node& grad_fn, const Tensor& self);

// From: torch/csrc/autograd/variable.cpp (bump_version)
RAXIS_API void bump_version(const Tensor& self);

// From: torch/csrc/autograd/variable.cpp (rebase_history)
// grad_fn becomes the history of self. If self is a view, the op wrote a
// part of its base: the base gets a CopySlices node, and the view reads its
// gradient from the base's.
RAXIS_API void rebase_history(const Tensor& self, NodePtr grad_fn);

// Bumps the version of self and, if grad_fn is set, rebases its history
RAXIS_API void finish_inplace(const Tensor& self, NodePtr grad_fn);

// From: torch/csrc/autograd/variable.cpp (VariableHooks::grad_fn)
// The grad_fn of a view whose memory was written in place through its base
// or another view: it takes its slice of the base's gradient
// (AsStridedBackward0), since the base's history now includes the write.
RAXIS_API void refresh_view_grad_fn(const Tensor& view);

// The views returned by unbind / split / chunk (CreationMeta::MultiOutputNode)
RAXIS_API void mark_multi_output_views(ArrayRef<Tensor> views);

// The steps above for an op that writes self. `inputs` are its
// differentiable inputs in the order of the node's edges, self first;
// node_args are the arguments of the node's constructor, which only runs
// if autograd records the op.
template <typename BackwardNode, typename Kernel, typename... NodeArgs>
Tensor& inplace_op(Tensor& self, ArrayRef<Tensor> inputs, Kernel&& kernel, NodeArgs&&... node_args) {
    const bool requires_grad = requires_autograd(inputs);
    check_inplace(self, requires_grad);
    std::shared_ptr<BackwardNode> grad_fn;
    if (requires_grad) {
        grad_fn = std::make_shared<BackwardNode>(std::forward<NodeArgs>(node_args)...);
        grad_fn->set_num_inputs(1);
        set_next_edges(grad_fn, inputs);
        save_original_self(*grad_fn, self);
    }
    kernel();
    finish_inplace(self, grad_fn);
    return self;
}

} // namespace raxis::autograd