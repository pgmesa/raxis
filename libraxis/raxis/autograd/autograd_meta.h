#pragma once

#include <memory>

#include <raxis/tensor/tensor.h>


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

    bool is_leaf() const { return !grad_fn; }
};

} // namespace raxis::autograd