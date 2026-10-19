#pragma once

#include <memory>

#include <raxis/tensor/tensor.h>


namespace raxis::autograd {
    
class Node; 

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