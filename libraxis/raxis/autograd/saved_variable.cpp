#include <raxis/autograd/saved_variable.h>

#include <memory>
#include <vector>

#include <raxis/util/exception.h>
#include <raxis/autograd/graph.h>


namespace raxis::autograd {

// From: torch/csrc/autograd/saved_variable.cpp (ERR_BACKWARD_TWICE)
static const char* ERR_BACKWARD_TWICE =
    "Trying to backward through the graph a second time (or directly access saved "
    "tensors after they have already been freed). Saved intermediate values "
    "of the graph are freed when you call .backward(). Specify "
    "retain_graph=True if you need to backward through the graph a second time or "
    "if you need to access saved tensors after calling backward.";

SavedVariable::SavedVariable(Node* owner, const char* name) : name_(name) {
    owner->register_saved_variable(this);
}

void SavedVariable::save(const Tensor& tensor, bool is_output) {
    data_ = is_output && tensor.defined() ? tensor.detach() : tensor;
    was_saved_ = true;
}

const Tensor& SavedVariable::unpack() const {
    RAXIS_CHECK(!released_, ERR_BACKWARD_TWICE);
    return data_;
}

void SavedVariable::release() {
    if (was_saved_) {
        data_ = Tensor();
        released_ = true;
    }
}


void SavedVariableList::save(ArrayRef<Tensor> tensors) {
    items_.reserve(tensors.size());
    for (const Tensor& tensor : tensors) {
        items_.push_back(std::make_unique<SavedVariable>(owner_, name_));
        items_.back()->save(tensor);
    }
}

std::vector<Tensor> SavedVariableList::unpack() const {
    std::vector<Tensor> tensors;
    tensors.reserve(items_.size());
    for (const auto& item : items_) {
        tensors.push_back(item->unpack());
    }
    return tensors;
}

} // namespace raxis::autograd
