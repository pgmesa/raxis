#pragma once

#include <raxis/export.h>
#include <raxis/tensor/tensor.h>


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

    // Undefined if nothing was saved. Raises if it was saved and released.
    const Tensor& unpack() const;

    void release();

    const char* name() const { return name_; }
    bool was_saved() const { return was_saved_; }
    bool is_released() const { return released_; }

private:
    Tensor data_;
    const char* name_;
    bool was_saved_ = false;
    bool released_ = false;
};

} // namespace raxis::autograd
