
#include <raxis/autograd/variable.h>

#include <raxis/util/exception.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/autograd/graph.h>
#include <raxis/autograd/engine.h>

#include <raxis/ops/binary_ops.h>
#include <raxis/ops/reduce_ops.h>


namespace raxis::autograd {

AccumulateGrad::AccumulateGrad(const Tensor& variable) : variable_(variable) {
    RAXIS_INTERNAL_ASSERT(variable.requires_grad(),
        "AccumulateGrad: variable does not require grad");
    set_num_inputs(1);
}

std::string AccumulateGrad::name() const {
    return "AccumulateGrad";
}

TensorList AccumulateGrad::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    RAXIS_CHECK(grad.defined(), "AccumulateGrad received undefined gradient");

    if (variable_.grad().defined()) {
        variable_.set_grad(variable_.grad() + grad);
    } else {
        variable_.set_grad(grad);
    }

    return {};
}


void set_autograd_meta(Tensor& result, NodePtr grad_fn, uint32_t output_nr) {
    result.set_requires_grad(true);
    result.set_grad_fn(grad_fn);
    result.set_output_nr(output_nr);
}


static Edge make_edge_for(const Tensor& input) {
    if (!input.requires_grad()) {
        return Edge{};
    }
    if (input.grad_fn()) {
        return Edge{input.grad_fn(), input.output_nr()};
    }
    // Leaf tensor: create or reuse AccumulateGrad
    auto acc = std::make_shared<AccumulateGrad>(input);
    return Edge{std::move(acc), 0};
}


void set_next_edges(const NodePtr& grad_fn, const Tensor& input) {
    grad_fn->add_next_edge(make_edge_for(input));
}

void set_next_edges(const NodePtr& grad_fn, const Tensor& input0, const Tensor& input1) {
    grad_fn->add_next_edge(make_edge_for(input0));
    grad_fn->add_next_edge(make_edge_for(input1));
}

void set_next_edges(const NodePtr& grad_fn, ArrayRef<Tensor> inputs) {
    for (const auto& input : inputs) {
        grad_fn->add_next_edge(make_edge_for(input));
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