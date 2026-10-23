#include <raxis/autograd/engine.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
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
// With create_graph the sum must be recorded: never in place.
void accumulate(TensorList& buffer, size_t pos, Tensor&& grad) {
    Tensor& old = buffer[pos];
    if (!old.defined()) {
        old = std::move(grad);
    } else if (!GradMode::is_enabled() && can_accumulate_inplace(old)) {
        old.add_(grad);
    } else {
        old = old + grad;
    }
}

struct NodeTask {
    NodePtr node;
    TensorList inputs;
};

// From: torch/csrc/autograd/engine.cpp (call_function)
// The node with its hooks. The inputs are cleared before returning: a node
// that returns its incoming gradient as is (add, a view) must not see it
// shared when the next node accumulates into it.
TensorList call_function(Node& node, TensorList& inputs) {
    node.call_tensor_pre_hooks(inputs);
    node.call_retain_grad(inputs);
    node.call_pre_hooks(inputs);
    TensorList outputs = node.apply(inputs);
    node.call_post_hooks(outputs, inputs);
    inputs.clear();
    return outputs;
}

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

// From: torch/csrc/autograd/engine.cpp (GraphTask::init_to_execute)
// For autograd.grad: the nodes that must run are those with a path to a
// node whose gradient is captured. Post-order DFS with an explicit stack:
// a node is needed if one of its children is needed or captured.
std::unordered_set<Node*> nodes_to_execute(Node* root, const std::unordered_set<Node*>& captured) {
    std::unordered_set<Node*> needed;
    std::unordered_set<Node*> visited{root};
    // (node, index of the next edge to look at)
    std::vector<std::pair<Node*, size_t>> stack{{root, 0}};
    while (!stack.empty()) {
        auto& [node, next_edge] = stack.back();
        const EdgeList& edges = node->next_edges();
        if (next_edge < edges.size()) {
            Node* child = edges[next_edge++].node.get();
            if (child && visited.insert(child).second) {
                stack.emplace_back(child, 0);
            }
            continue;
        }
        for (const Edge& edge : edges) {
            Node* child = edge.node.get();
            if (child && (needed.count(child) || captured.count(child))) {
                needed.insert(node);
                break;
            }
        }
        stack.pop_back();
    }
    return needed;
}

// From: torch/csrc/autograd/functions/basic_ops.h (GraphRoot)
// The root of a backward from several tensors: it hands each its gradient
class GraphRoot : public Node {
public:
    GraphRoot(EdgeList roots, TensorList grads) : grads_(std::move(grads)) {
        for (Edge& edge : roots) {
            add_next_edge(std::move(edge));
        }
    }
    std::string name() const override { return "GraphRoot"; }
    TensorList apply(const TensorList&) override { return grads_; }

private:
    TensorList grads_;
};

// From: torch/csrc/autograd/autograd.cpp (_make_grads)
TensorList make_grads(ArrayRef<Tensor> outputs, ArrayRef<Tensor> grad_outputs) {
    RAXIS_CHECK(grad_outputs.empty() || grad_outputs.size() == outputs.size(),
        "got ", outputs.size(), " tensors and ", grad_outputs.size(), " gradients");
    TensorList grads;
    for (size_t i = 0; i < outputs.size(); i++) {
        const Tensor& output = outputs[i];
        const Tensor grad = grad_outputs.empty() ? Tensor() : grad_outputs[i];
        if (grad.defined()) {
            RAXIS_CHECK(grad.shape() == output.shape(),
                "Mismatch in shape: grad_output[", i, "] has a shape of ", grad.shape(),
                " and output[", i, "] has a shape of ", output.shape(), ".");
            grads.push_back(grad);
            continue;
        }
        RAXIS_CHECK(output.numel() == 1, "grad can be implicitly created only for scalar outputs");
        RAXIS_CHECK(output.dtype().is_floating(),
            "grad can be computed only for real scalar outputs but got ", output.dtype());
        grads.push_back(ones_like(output, {}, MemoryFormat::Contiguous));
    }
    return grads;
}

// From: torch/csrc/autograd/engine.cpp (Engine::execute, evaluate_function)
//
// Single-threaded version of PyTorch's engine. Instead of building a full
// topological order first, each node waits in `not_ready` with its input
// buffer until its dependency count drops to 0, and then moves to the ready
// queue. Gradients are only kept for the nodes between the "frontier" and
// the root, and buffers are freed as soon as their node runs.
//
// With captures (autograd.grad), the gradient that reaches each captured
// edge is returned, and only the nodes on a path to a captured edge run: a
// leaf's AccumulateGrad is captured, not run, so .grad is left alone.
TensorList execute(EdgeList roots, TensorList grads, bool keep_graph, bool create_graph,
                   const EdgeList& captures) {
    // The backward of the backward is recorded only with create_graph
    GradModeGuard grad_mode(create_graph);

    // A single root gets its gradient directly in its input buffer
    NodePtr root;
    TensorList root_inputs;
    if (roots.size() == 1) {
        root = roots[0].node;
        root_inputs.resize(root->num_inputs());
        root_inputs[roots[0].input_nr] = std::move(grads[0]);
    } else {
        root = std::make_shared<GraphRoot>(std::move(roots), std::move(grads));
    }

    std::unordered_map<Node*, std::vector<std::pair<uint32_t, size_t>>> capture_slots;
    std::unordered_set<Node*> captured_nodes;
    for (size_t i = 0; i < captures.size(); i++) {
        if (captures[i].node) {
            capture_slots[captures[i].node.get()].emplace_back(captures[i].input_nr, i);
            captured_nodes.insert(captures[i].node.get());
        }
    }
    const bool has_captures = !captures.empty();
    const std::unordered_set<Node*> needed = has_captures
        ? nodes_to_execute(root.get(), captured_nodes) : std::unordered_set<Node*>();
    TensorList captured(captures.size());

    std::unordered_map<Node*, int> dependencies = compute_dependencies(root.get());
    std::unordered_map<Node*, TensorList> not_ready;
    ReadyQueue ready;
    ready.push({root, std::move(root_inputs)});

    while (!ready.empty()) {
        NodeTask task = ready.pop();
        Node& node = *task.node;

        if (has_captures) {
            const bool is_needed = needed.count(&node) > 0;
            // The hooks of the captured tensor still run (call_function
            // runs them for a node that is needed)
            if (!is_needed) {
                node.call_tensor_pre_hooks(task.inputs);
            }
            if (auto it = capture_slots.find(&node); it != capture_slots.end()) {
                for (auto [input_nr, index] : it->second) {
                    captured[index] = task.inputs[input_nr];
                }
            }
            if (!is_needed) continue;
        }

        TensorList outputs = call_function(node, task.inputs);

        if (!keep_graph) {
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
            if (has_captures && !needed.count(next) && !captured_nodes.count(next)) continue;

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
    return captured;
}

EdgeList root_edges(ArrayRef<Tensor> tensors) {
    EdgeList roots;
    for (size_t i = 0; i < tensors.size(); i++) {
        RAXIS_CHECK(tensors[i].defined(), "element ", i, " of tensors is not defined");
        Edge edge = gradient_edge(tensors[i]);
        RAXIS_CHECK(edge.is_valid(),
            "element ", i, " of tensors does not require grad and does not have a grad_fn");
        roots.push_back(std::move(edge));
    }
    return roots;
}

} // namespace


void backward(ArrayRef<Tensor> tensors, ArrayRef<Tensor> grad_tensors,
              std::optional<bool> retain_graph, bool create_graph) {
    TensorList grads = make_grads(tensors, grad_tensors);
    execute(root_edges(tensors), std::move(grads), retain_graph.value_or(create_graph), create_graph, {});
}

TensorList grad(ArrayRef<Tensor> outputs, ArrayRef<Tensor> inputs, ArrayRef<Tensor> grad_outputs,
                std::optional<bool> retain_graph, bool create_graph, bool allow_unused) {
    TensorList grads = make_grads(outputs, grad_outputs);
    EdgeList captures;
    for (size_t i = 0; i < inputs.size(); i++) {
        RAXIS_CHECK(inputs[i].requires_grad(),
            "One of the differentiated Tensors does not require grad");
        captures.push_back(gradient_edge(inputs[i]));
    }
    TensorList result = execute(root_edges(outputs), std::move(grads),
                                retain_graph.value_or(create_graph), create_graph, captures);
    if (!allow_unused) {
        for (const Tensor& g : result) {
            RAXIS_CHECK(g.defined(),
                "One of the differentiated Tensors appears to not have been used in the graph. "
                "Set allow_unused=True if this is the desired behavior.");
        }
    }
    return result;
}

void run_backward(const Tensor& tensor, const Tensor& grad, bool retain_graph, bool create_graph) {
    RAXIS_CHECK(tensor.defined(), "Tensor passed to backward is not defined");
    RAXIS_CHECK(tensor.requires_grad(), "element 0 of tensors does not require grad and does not have a grad_fn");
    backward({tensor}, {grad}, retain_graph || create_graph, create_graph);
}

} // namespace raxis::autograd
