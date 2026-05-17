#pragma once

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

private:
    Tensor variable_;
};


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

} // namespace raxis::autograd