#include <raxis/autograd/engine.h>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>

#include <raxis/util/exception.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/autograd/graph.h>
#include <raxis/autograd/grad_mode.h>
#include <raxis/autograd/variable.h>


namespace raxis::autograd {


void run_backward(const Tensor& tensor, const Tensor& grad, bool retain_graph) {
    RAXIS_CHECK(tensor.defined(),
        "Tensor passed to backward is not defined");
    RAXIS_CHECK(tensor.requires_grad(),
        "Tensor does not require grad and does not have a grad_fn");

    Tensor grad_output = grad;
    if (!grad.defined()) {
        RAXIS_CHECK(tensor.numel() == 1,
            "grad can be implicitly created only for scalar outputs");
        grad_output = ones_like(tensor);
    }

    RAXIS_CHECK(grad_output.shape() == tensor.shape(),
        "Shape mismatch between grad and tensor");

    // Backward on a leaf: the graph is just its AccumulateGrad
    if (!tensor.grad_fn()) {
        NoGradGuard guard;
        AccumulateGrad(tensor).apply({grad_output});
        return;
    }

    // 1) Build topological order via DFS
    std::vector<NodePtr> topo;
    std::unordered_set<Node*> visited;
    std::unordered_map<Node*, TensorList> grad_buffer;

    std::function<void(const NodePtr&)> dfs = [&](const NodePtr& node) {
        if (!node || visited.count(node.get())) return;
        visited.insert(node.get());

        grad_buffer[node.get()].resize(node->num_inputs());

        for (const Edge& edge : node->next_edges()) {
            dfs(edge.node);
        }
        topo.push_back(node);
    };

    NodePtr root = tensor.grad_fn();
    dfs(root);

    // Seed the root node
    grad_buffer[root.get()][tensor.output_nr()] = grad_output;

    // 2) Walk in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        NodePtr node = *it;

        TensorList& input_grads = grad_buffer[node.get()];

        RAXIS_CHECK(!node->is_released(),
            "Trying to backward through the graph a second time (", node->name(),
            "). Specify retain_graph=True if you need to backward through the "
            "graph a second time.");

        TensorList output_grads;
        {
            NoGradGuard guard;
            output_grads = node->apply(input_grads);
        }

        // Retain grad: save the incoming gradient on the tensor if it is still alive
        if (Tensor t = node->retained_tensor(); t.defined()) {
            const Tensor& incoming = input_grads[0];
            if (incoming.defined()) {
                if (t.grad().defined()) {
                    t.set_grad(t.grad() + incoming);
                } else {
                    t.set_grad(incoming);
                }
            }
        }

        const EdgeList& edges = node->next_edges();
        RAXIS_INTERNAL_ASSERT(edges.size() == output_grads.size(),
            node->name(), " returned ", output_grads.size(),
            " gradients, but has ", edges.size(), " edges");

        for (size_t i = 0; i < edges.size(); ++i) {
            const Edge& edge = edges[i];
            if (!edge.is_valid()) continue;

            Tensor& out_grad = output_grads[i];
            if (!out_grad.defined()) continue;

            TensorList& next_grads = grad_buffer.at(edge.node.get());
            Tensor& slot = next_grads[edge.input_nr];

            // Accumulate out of place: the slot may be a broadcast view (stride 0,
            // e.g. from SumBackward0) or share memory with another slot or a
            // retained .grad, so writing into it would corrupt them.
            if (slot.defined()) {
                slot = slot + out_grad;
            } else {
                slot = out_grad;
            }
        }

        if (!retain_graph) {
            node->mark_released();
            grad_buffer.erase(node.get());
        }
    }
}

} // namespace raxis::autograd