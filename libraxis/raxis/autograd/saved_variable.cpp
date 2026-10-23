#include <raxis/autograd/saved_variable.h>

#include <memory>
#include <sstream>
#include <string>
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

SavedVariable::SavedVariable(Node* owner, const char* name) : owner_(owner), name_(name) {
    owner->register_saved_variable(this);
}

void SavedVariable::save(const Tensor& tensor, bool is_output) {
    data_ = is_output && tensor.defined() ? tensor.detach() : tensor;
    is_output_ = is_output;
    was_saved_ = true;
    if (tensor.defined()) {
        saved_version_ = tensor.version();
        output_nr_ = tensor.output_nr();
    }
}

// From: torch/csrc/autograd/saved_variable.cpp (SavedVariable::unpack)
// The backward formula needs the values the tensor had when it was saved:
// if an in-place op wrote its memory since, the gradient would be wrong.
const Tensor& SavedVariable::unpack() const {
    RAXIS_CHECK(!released_, ERR_BACKWARD_TWICE);
    if (data_.defined() && data_.version() != saved_version_) {
        // An output is saved without its grad_fn (it would own this node):
        // the node that saved it is its grad_fn
        NodePtr grad_fn = data_.grad_fn();
        std::string producer = is_output_ ? owner_->forward_op_name()
                                          : (grad_fn ? grad_fn->forward_op_name() : "");
        std::ostringstream message;
        message << "one of the variables needed for gradient computation has been "
                   "modified by an inplace operation: [" << data_.dtype() << " " << data_.shape() << "]";
        if (!producer.empty()) {
            message << ", which is output " << output_nr_ << " of " << producer << ",";
        }
        message << " is at version " << data_.version() << "; expected version "
                << saved_version_ << " instead.";
        RAXIS_CHECK(false, message.str());
    }
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
