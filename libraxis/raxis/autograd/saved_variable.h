#pragma once

#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

// From: torch/csrc/autograd/saved_variable.h
// A tensor saved by a backward node to be used later in apply().
//
// Inputs can be stored as they are, but an output owns the node that saves
// it through its grad_fn. Storing the output itself would close the cycle
// output -> grad_fn -> node -> output and the graph would never be freed, so
// outputs are stored as a detached alias that shares the same storage.
class SavedVariable {
public:
    SavedVariable() = default;
    SavedVariable(const Tensor& tensor, bool is_output)
        : data_(is_output ? tensor.detach() : tensor) {}

    const Tensor& unpack() const { return data_; }

private:
    Tensor data_;
};

} // namespace raxis::autograd
