#pragma once

#include <memory>
#include <vector>

#include <raxis/export.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/array_ref.h>


namespace raxis::autograd {

class Node;

// From: torch/csrc/autograd/saved_variable.h
// A tensor saved by a backward node to be used later in apply().
//
// Inputs can be stored as they are, but an output owns the node that saves
// it through its grad_fn. Storing the output itself would close the cycle
// output -> grad_fn -> node -> output and the graph would never be freed, so
// outputs are stored as a detached alias that shares the same storage.
//
// Each SavedVariable registers itself with the node that owns it, declared as
// a member with `SavedVariable self_{this, "self"};`. The node can then free
// all its saved tensors after backward (Node::release_variables) and list
// them, without code per node. PyTorch generates that code per node instead.
class RAXIS_API SavedVariable {
public:
    // name: the argument of the forward op it holds, as PyTorch names it
    SavedVariable(Node* owner, const char* name);

    // Registered by address in its node
    SavedVariable(const SavedVariable&) = delete;
    SavedVariable& operator=(const SavedVariable&) = delete;

    void save(const Tensor& tensor, bool is_output = false);

    // Undefined if nothing was saved. Raises if it was saved and released,
    // or if an in-place op changed it after it was saved.
    Tensor unpack() const;

    // Whether this holds `tensor` itself (not a copy or another view of it)
    bool holds(const Tensor& tensor) const { return !released_ && data_.defined() && data_.is_same(tensor); }

    void release();

    const char* name() const { return name_; }
    bool was_saved() const { return was_saved_; }
    bool is_released() const { return released_; }

private:
    Tensor data_;
    Node* owner_;
    const char* name_;
    // Version of the memory of data_ when it was saved
    uint32_t saved_version_ = 0;
    uint32_t output_nr_ = 0;
    bool is_output_ = false;
    bool was_saved_ = false;
    bool released_ = false;
};

// A list of saved inputs, such as the index tensors of x[i, j]: one
// SavedVariable per tensor, all registered with the node under the same
// name (PyTorch: std::vector<SavedVariable> indices_). Undefined tensors
// (dims without an index) are kept as such.
class RAXIS_API SavedVariableList {
public:
    SavedVariableList(Node* owner, const char* name) : owner_(owner), name_(name) {}

    SavedVariableList(const SavedVariableList&) = delete;
    SavedVariableList& operator=(const SavedVariableList&) = delete;

    void save(ArrayRef<Tensor> tensors);
    void add(const Tensor& tensor, bool is_output);

    std::vector<Tensor> unpack() const;

private:
    Node* owner_;
    const char* name_;
    // unique_ptr: the node holds their addresses
    std::vector<std::unique_ptr<SavedVariable>> items_;
};

} // namespace raxis::autograd
