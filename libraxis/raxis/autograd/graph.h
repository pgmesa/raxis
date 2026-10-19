#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <raxis/export.h>
#include <raxis/core/types.h>
#include <raxis/util/array_ref.h>
#include <raxis/util/small_vector.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

struct Edge;
class Node;
class SavedVariable;

using NodePtr = std::shared_ptr<Node>;
using EdgeList = std::vector<Edge>;
using TensorList = std::vector<Tensor>;

struct RAXIS_API Edge {
    Edge() : node(nullptr), input_nr(0) {}
    Edge(NodePtr node, uint32_t input_nr)
        : node(std::move(node)), input_nr(input_nr) {}

    bool is_valid() const { return node != nullptr; }

    NodePtr node;
    uint32_t input_nr;
};

// Thread-local counter that orders nodes by creation time
RAXIS_API uint64_t next_sequence_nr();

class RAXIS_API Node {
public:
    Node() : sequence_nr_(next_sequence_nr()) {}
    explicit Node(uint64_t sequence_nr) : sequence_nr_(sequence_nr) {}
    virtual ~Node();

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&) = delete;
    Node& operator=(Node&&) = delete;

    virtual std::string name() const = 0;
    virtual TensorList apply(const TensorList& grads) = 0;

    void set_num_inputs(size_t n) { num_inputs_ = n; }
    size_t num_inputs() const { return num_inputs_; }

    void add_next_edge(Edge edge) {
        next_edges_.emplace_back(std::move(edge));
    }

    const EdgeList& next_edges() const { return next_edges_; }
    size_t num_outputs() const { return next_edges_.size(); }

    // The engine runs ready nodes with the highest sequence_nr first: the
    // most recently created op, so the backward follows the forward in
    // reverse as closely as the dependencies allow.
    uint64_t sequence_nr() const { return sequence_nr_; }

    // The retained tensor is an output of this node and owns it through its
    // grad_fn, so it is held weakly to avoid a reference cycle. Returns an
    // undefined tensor if it was already freed.
    void set_retained_tensor(const Tensor& t) { retained_tensor_ = t.weak_impl(); }
    Tensor retained_tensor() const { return Tensor(retained_tensor_.lock()); }

    // From: torch/csrc/autograd/function.h (release_variables)
    // Frees the saved tensors once the node ran in a backward without
    // retain_graph: memory goes down as the backward advances instead of
    // staying at its peak until the graph is dropped.
    void release_variables();

    ArrayRef<SavedVariable*> saved_variables() const { return saved_variables_; }

private:
    friend class SavedVariable;
    void register_saved_variable(SavedVariable* saved) { saved_variables_.push_back(saved); }

    uint64_t sequence_nr_;
    EdgeList next_edges_;
    size_t num_inputs_ = 0;
    std::weak_ptr<detail::TensorImpl> retained_tensor_;
    // Members of the derived node: valid while the node is alive
    SmallVector<SavedVariable*, 2> saved_variables_;
};

} // namespace raxis::autograd