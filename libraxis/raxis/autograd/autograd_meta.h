#pragma once

#include <memory>

#include <raxis/tensor/tensor.h>
#include <raxis/autograd/hooks.h>


namespace raxis::autograd {
    
class Node; 

// From: torch/csrc/autograd/variable.h (CreationMeta)
// How a view was made, for the in-place ops it does not allow
enum class CreationMeta : uint8_t {
    // In-place allowed: autograd records it on the base (CopySlices)
    Default,
    // Made under no_grad: modifying it with grad mode on is ambiguous (should
    // the write be tracked or not?), so it is an error
    NoGradMode,
    // One of the views of unbind / split / chunk: CopySlices can only
    // replace the backward of a single view
    MultiOutputNode,
    // Returned by a Python autograd.Function: CopySlices would replace its
    // custom backward
    InCustomFunction,
};

struct AutogradMeta {
    bool requires_grad = false;
    bool retains_grad = false;
    Tensor grad;
    std::shared_ptr<autograd::Node> grad_fn;
    // Leaves only: the AccumulateGrad shared by every op that uses the leaf.
    // Weak because the node owns the leaf (and so this meta).
    std::weak_ptr<autograd::Node> grad_accumulator;
    uint32_t output_nr = 0;
    // Tensor::register_hook. Shared with the grad_fn (or the AccumulateGrad
    // of a leaf) that runs them; a new list after an in-place op, so new
    // hooks see the gradient of the new values.
    std::shared_ptr<HookList<TensorHook>> tensor_hooks;
    std::shared_ptr<HookList<PostAccumulateGradHook>> post_accumulate_grad_hooks;

    bool is_leaf() const { return !grad_fn; }
};

} // namespace raxis::autograd