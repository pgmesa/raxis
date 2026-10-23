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
#include <raxis/autograd/hooks.h>


namespace raxis::autograd {

struct Edge;
class Node;
class SavedVariable;

using NodePtr = std::shared_ptr<Node>;
using EdgeList = std::vector<Edge>;

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

class RAXIS_API Node : public std::enable_shared_from_this<Node> {
public:
    Node() : sequence_nr_(next_sequence_nr()) {}
    explicit Node(uint64_t sequence_nr) : sequence_nr_(sequence_nr) {}
    virtual ~Node();

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    Node(Node&&) = delete;
    Node& operator=(Node&&) = delete;

    virtual std::string name() const = 0;

    // The op whose backward this is, for error messages: MulBackward0 -> Mul
    std::string forward_op_name() const;
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

    // -- Hooks --
    // From: torch/csrc/autograd/function.h
    // The engine runs them around apply, in this order: the hooks of the
    // tensors whose gradients arrive here, retain_grad, pre hooks, apply,
    // post hooks (call_function in engine.cpp).
    RemovableHandle register_prehook(FunctionPreHook hook);
    RemovableHandle register_hook(FunctionPostHook hook);

    // The hooks of the tensor that is output `output_nr` of the op. The list
    // is shared with the tensor: a hook removed there is removed here.
    void add_tensor_hooks(uint32_t output_nr, std::shared_ptr<HookList<TensorHook>> hooks);

    // retain_grad on output `output_nr`. The tensor owns this node through
    // its grad_fn, so it is held weakly to avoid a reference cycle.
    void set_retained_tensor(uint32_t output_nr, const Tensor& t);
    void clear_retained_tensor(uint32_t output_nr);

    // Each takes the gradients the node receives (grads) and may replace them
    virtual void call_tensor_pre_hooks(TensorList& grads);
    void call_retain_grad(const TensorList& grads);
    void call_pre_hooks(TensorList& grads);
    void call_post_hooks(TensorList& grad_inputs, const TensorList& grads);

    // From: torch/csrc/autograd/function.h (release_variables)
    // Frees the saved tensors once the node ran in a backward without
    // retain_graph: memory goes down as the backward advances instead of
    // staying at its peak until the graph is dropped.
    virtual void release_variables();

    ArrayRef<SavedVariable*> saved_variables() const { return saved_variables_; }

private:
    friend class SavedVariable;
    void register_saved_variable(SavedVariable* saved) { saved_variables_.push_back(saved); }

    uint64_t sequence_nr_;
    EdgeList next_edges_;
    size_t num_inputs_ = 0;
    // Allocated on first use: most nodes never get a hook
    std::vector<std::pair<uint32_t, std::shared_ptr<HookList<TensorHook>>>> tensor_hooks_;
    std::vector<std::pair<uint32_t, std::weak_ptr<detail::TensorImpl>>> retained_tensors_;
    std::shared_ptr<HookList<FunctionPreHook>> pre_hooks_;
    std::shared_ptr<HookList<FunctionPostHook>> post_hooks_;
    // Members of the derived node: valid while the node is alive
    SmallVector<SavedVariable*, 2> saved_variables_;
};

} // namespace raxis::autograd