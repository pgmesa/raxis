#include <raxis/autograd/engine.h>

#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include <raxis/util/exception.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/autograd/graph.h>
#include <raxis/autograd/grad_mode.h>
#include <raxis/autograd/variable.h>


namespace raxis::autograd {

namespace {

// From: torch/csrc/autograd/input_buffer.cpp (can_accumulate_inplace)
// The buffer slot can be overwritten only if nothing else sees it: no other
// Tensor holds the impl (another slot, a retained .grad, the gradient the
// user passed to backward) and no view shares its storage. It must also be
// dense: a broadcast view (stride 0, e.g. from SumBackward0) has several
// elements on the same address.
bool can_accumulate_inplace(const Tensor& t) {
    return t.use_count() == 1
        && t.unsafe_get_tensor_impl()->storage().use_count() == 1
        && t.is_non_overlapping_and_dense();
}

// From: torch/csrc/autograd/input_buffer.cpp (accumulate)
// Sums the gradients that reach the same input of a node. In place when
// possible: a parameter used by k ops gets k - 1 additions without
// allocating k - 1 temporaries.
void accumulate(TensorList& buffer, size_t pos, Tensor&& grad) {
    Tensor& old = buffer[pos];
    if (!old.defined()) {
        old = std::move(grad);
    } else if (can_accumulate_inplace(old)) {
        old.add_(grad);
    } else {
        old = old + grad;
    }
}

struct NodeTask {
    NodePtr node;
    TensorList inputs;
};

// From: torch/csrc/autograd/engine.h (ReadyQueue)
// Max-heap on sequence_nr. A heap over a vector instead of a
// std::priority_queue, whose top() is const and would force copying the
// task (and its gradients) out instead of moving it.
class ReadyQueue {
public:
    void push(NodeTask task) {
        heap_.push_back(std::move(task));
        std::push_heap(heap_.begin(), heap_.end(), compare);
    }

    NodeTask pop() {
        std::pop_heap(heap_.begin(), heap_.end(), compare);
        NodeTask task = std::move(heap_.back());
        heap_.pop_back();
        return task;
    }

    bool empty() const { return heap_.empty(); }

private:
    static bool compare(const NodeTask& a, const NodeTask& b) {
        return a.node->sequence_nr() < b.node->sequence_nr();
    }

    std::vector<NodeTask> heap_;
};

// From: torch/csrc/autograd/engine.cpp (Engine::compute_dependencies)
// Number of edges that point to each node reachable from root: a node can
// run once that many gradients have arrived. Iterative, with an explicit
// stack, so the graph depth is not limited by the call stack.
std::unordered_map<Node*, int> compute_dependencies(Node* root) {
    std::unordered_map<Node*, int> dependencies;
    std::vector<Node*> stack{root};
    while (!stack.empty()) {
        Node* node = stack.back();
        stack.pop_back();
        for (const Edge& edge : node->next_edges()) {
            if (Node* next = edge.node.get()) {
                // First time seen: expand it
                if (dependencies[next]++ == 0) {
                    stack.push_back(next);
                }
            }
        }
    }
    return dependencies;
}

} // namespace


// From: torch/csrc/autograd/engine.cpp (Engine::execute, evaluate_function)
// Single-threaded version of PyTorch's engine. Instead of building a full
// topological order first, each node waits in `not_ready` with its input
// buffer until its dependency count drops to 0, and then moves to the ready
// queue. Gradients are only kept for the nodes between the "frontier" and
// the root, and buffers are freed as soon as their node runs.
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

    NoGradGuard guard;

    // Backward on a leaf: the graph is just its AccumulateGrad
    if (!tensor.grad_fn()) {
        // Moved in, so that an implicit ones_like can be stolen as .grad
        TensorList inputs(1);
        inputs[0] = std::move(grad_output);
        grad_accumulator(tensor)->apply(inputs);
        return;
    }

    NodePtr root = tensor.grad_fn();
    std::unordered_map<Node*, int> dependencies = compute_dependencies(root.get());
    std::unordered_map<Node*, TensorList> not_ready;
    ReadyQueue ready;

    TensorList root_inputs(root->num_inputs());
    root_inputs[tensor.output_nr()] = std::move(grad_output);
    ready.push({root, std::move(root_inputs)});

    while (!ready.empty()) {
        NodeTask task = ready.pop();
        Node& node = *task.node;

        TensorList outputs = node.apply(task.inputs);

        // Retain grad: save the incoming gradient on the tensor if it is still alive
        if (Tensor t = node.retained_tensor(); t.defined()) {
            const Tensor& incoming = task.inputs[0];
            if (incoming.defined()) {
                if (t.grad().defined()) {
                    t.set_grad(t.grad() + incoming);
                } else {
                    t.set_grad(incoming);
                }
            }
        }
        // Free the inputs before passing the outputs on: a node that returns
        // its incoming gradient as is (add, a view) must not see it shared
        task.inputs.clear();

        if (!retain_graph) {
            node.release_variables();
        }

        const EdgeList& edges = node.next_edges();
        RAXIS_INTERNAL_ASSERT(edges.size() == outputs.size(),
            node.name(), " returned ", outputs.size(),
            " gradients, but has ", edges.size(), " edges");

        for (size_t i = 0; i < edges.size(); ++i) {
            const Edge& edge = edges[i];
            if (!edge.is_valid()) continue;
            Node* next = edge.node.get();

            // Undefined gradients still count as arrived: the node runs with
            // an undefined slot (an output of a multi-output op not used)
            auto [buffer_it, inserted] = not_ready.try_emplace(next);
            TensorList& buffer = buffer_it->second;
            if (inserted) {
                buffer.resize(next->num_inputs());
            }
            if (outputs[i].defined()) {
                accumulate(buffer, edge.input_nr, std::move(outputs[i]));
            }

            if (--dependencies.at(next) == 0) {
                ready.push({edge.node, std::move(buffer)});
                not_ready.erase(buffer_it);
            }
        }
    }
}

} // namespace raxis::autograd
