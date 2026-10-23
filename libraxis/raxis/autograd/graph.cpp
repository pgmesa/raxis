#include <raxis/autograd/graph.h>

#include <atomic>
#include <cctype>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <raxis/util/exception.h>
#include <raxis/autograd/saved_variable.h>


namespace raxis::autograd {

uint64_t next_sequence_nr() {
    static thread_local uint64_t sequence_nr = 0;
    return sequence_nr++;
}

// From: torch/csrc/autograd/function.cpp (Node::forward_op_name)
// Strips a Backward<N> suffix, keeping N unless it is 0 (AddBackward1 -> Add1)
std::string Node::forward_op_name() const {
    std::string n = name();
    const size_t pos = n.rfind("Backward");
    if (pos == std::string::npos) {
        return n;
    }
    const std::string suffix = n.substr(pos + 8);
    for (char c : suffix) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return n;
    }
    return n.substr(0, pos) + (suffix == "0" ? "" : suffix);
}

uint64_t next_hook_id() {
    static std::atomic<uint64_t> id{0};
    return id++;
}

RemovableHandle Node::register_prehook(FunctionPreHook hook) {
    if (!pre_hooks_) pre_hooks_ = std::make_shared<HookList<FunctionPreHook>>();
    return RemovableHandle(pre_hooks_, pre_hooks_->add(std::move(hook)));
}

RemovableHandle Node::register_hook(FunctionPostHook hook) {
    if (!post_hooks_) post_hooks_ = std::make_shared<HookList<FunctionPostHook>>();
    return RemovableHandle(post_hooks_, post_hooks_->add(std::move(hook)));
}

void Node::add_tensor_hooks(uint32_t output_nr, std::shared_ptr<HookList<TensorHook>> hooks) {
    tensor_hooks_.emplace_back(output_nr, std::move(hooks));
}

void Node::set_retained_tensor(uint32_t output_nr, const Tensor& t) {
    clear_retained_tensor(output_nr);
    retained_tensors_.emplace_back(output_nr, t.weak_impl());
}

void Node::clear_retained_tensor(uint32_t output_nr) {
    std::erase_if(retained_tensors_, [output_nr](const auto& entry) { return entry.first == output_nr; });
}

// From: torch/csrc/autograd/python_hook.cpp (PyFunctionTensorPreHook)
void Node::call_tensor_pre_hooks(TensorList& grads) {
    for (const auto& [output_nr, hooks] : tensor_hooks_) {
        for (const TensorHook& hook : hooks->hooks()) {
            if (!grads[output_nr].defined()) break;
            if (Tensor result = hook(grads[output_nr]); result.defined()) {
                grads[output_nr] = std::move(result);
            }
        }
    }
}

// From: torch/csrc/autograd/variable.cpp (VariableHooks::retain_grad)
// The gradient after the tensor hooks; added to .grad over several backwards
void Node::call_retain_grad(const TensorList& grads) {
    for (const auto& [output_nr, weak] : retained_tensors_) {
        Tensor t(weak.lock());
        const Tensor& grad = grads[output_nr];
        if (!t.defined() || !grad.defined()) continue;
        t.set_grad(t.grad().defined() ? t.grad() + grad : grad);
    }
}

static void check_hook_result(const Node& node, const char* kind, const TensorList& result, size_t expected) {
    RAXIS_CHECK(result.size() == expected,
        "hook '", kind, "' on ", node.name(), " has returned an incorrect number of values (got ",
        result.size(), ", but expected ", expected, ")");
}

void Node::call_pre_hooks(TensorList& grads) {
    if (!pre_hooks_) return;
    for (const FunctionPreHook& hook : pre_hooks_->hooks()) {
        if (std::optional<TensorList> result = hook(grads)) {
            check_hook_result(*this, "prehook", *result, grads.size());
            grads = std::move(*result);
        }
    }
}

void Node::call_post_hooks(TensorList& grad_inputs, const TensorList& grads) {
    if (!post_hooks_) return;
    for (const FunctionPostHook& hook : post_hooks_->hooks()) {
        if (std::optional<TensorList> result = hook(grad_inputs, grads)) {
            check_hook_result(*this, "hook", *result, grad_inputs.size());
            grad_inputs = std::move(*result);
        }
    }
}

void Node::release_variables() {
    for (SavedVariable* saved : saved_variables_) {
        saved->release();
    }
}

// From: torch/csrc/autograd/function.cpp (gatherFunctions)
// Takes the children this node owns alone out of its edges, so that they are
// destroyed by the caller's loop instead of by this node's destructor
static void gather_functions(EdgeList& edges, std::vector<NodePtr>& stack) {
    for (Edge& edge : edges) {
        if (edge.node.use_count() == 1) {
            stack.push_back(std::move(edge.node));
        } else {
            edge.node.reset();
        }
    }
}

// From: torch/csrc/autograd/function.cpp (releaseGraphIteratively)
// Dropping the last reference to the root of a chain of n nodes would run n
// nested destructors and overflow the stack for deep graphs (a long RNN, or
// 100k ops in a loop). Each node instead hands the children it owns to an
// explicit stack, and they are freed one at a time with constant depth.
Node::~Node() {
    std::vector<NodePtr> stack;
    gather_functions(next_edges_, stack);
    while (!stack.empty()) {
        NodePtr node = std::move(stack.back());
        stack.pop_back();
        // A saved input holds its grad_fn too, which would then be freed
        // recursively when the node dies: drop it first
        node->release_variables();
        gather_functions(node->next_edges_, stack);
    }
}

} // namespace raxis::autograd
