#pragma once

#include <memory>
#include <string>
#include <vector>

#include <raxis/export.h>
#include <raxis/core/types.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

struct Edge;
class Node;

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

class RAXIS_API Node {
public:
    Node() = default;
    virtual ~Node() = default;

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

    // The retained tensor is an output of this node and owns it through its
    // grad_fn, so it is held weakly to avoid a reference cycle. Returns an
    // undefined tensor if it was already freed.
    void set_retained_tensor(const Tensor& t) { retained_tensor_ = t.weak_impl(); }
    Tensor retained_tensor() const { return Tensor(retained_tensor_.lock()); }

private:
    EdgeList next_edges_;
    size_t num_inputs_ = 0;
    std::weak_ptr<detail::TensorImpl> retained_tensor_;
};

} // namespace raxis::autograd