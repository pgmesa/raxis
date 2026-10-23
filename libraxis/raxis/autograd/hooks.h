#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <raxis/export.h>
#include <raxis/tensor/tensor.h>


namespace raxis::autograd {

using TensorList = std::vector<Tensor>;

// From: torch/csrc/autograd/function_hook.h
//
// Tensor hook: sees the gradient of a tensor before the node that receives
//   it runs, and may return a replacement (undefined keeps it).
// Pre hook of a node: sees the gradients the node receives (its grad_outputs)
//   and may return replacements for all of them.
// Post hook of a node: sees what the node returned (its grad_inputs) and what
//   it received, and may return replacements for its results.
// Post-accumulate-grad hook: runs on a leaf once its .grad was updated.
using TensorHook = std::function<Tensor(const Tensor& grad)>;
using FunctionPreHook = std::function<std::optional<TensorList>(const TensorList& grad_outputs)>;
using FunctionPostHook = std::function<std::optional<TensorList>(
    const TensorList& grad_inputs, const TensorList& grad_outputs)>;
using PostAccumulateGradHook = std::function<void(const Tensor& tensor)>;

RAXIS_API uint64_t next_hook_id();

// The hooks of one kind on a tensor or a node, in registration order
template <typename Hook>
class HookList {
public:
    uint64_t add(Hook hook) {
        uint64_t id = next_hook_id();
        hooks_.emplace_back(id, std::move(hook));
        return id;
    }

    void remove(uint64_t id) {
        std::erase_if(hooks_, [id](const auto& entry) { return entry.first == id; });
    }

    bool empty() const { return hooks_.empty(); }

    // A copy: a hook may add or remove hooks while the list runs
    std::vector<Hook> hooks() const {
        std::vector<Hook> result;
        result.reserve(hooks_.size());
        for (const auto& entry : hooks_) {
            result.push_back(entry.second);
        }
        return result;
    }

private:
    std::vector<std::pair<uint64_t, Hook>> hooks_;
};

// From: torch/utils/hooks.py (RemovableHandle)
// Returned when a hook is registered: remove() unregisters it. It holds the
// list weakly, so a handle kept around does not keep a tensor or a node
// alive.
class RAXIS_API RemovableHandle {
public:
    template <typename Hook>
    RemovableHandle(const std::shared_ptr<HookList<Hook>>& list, uint64_t id)
        : remove_([weak = std::weak_ptr<HookList<Hook>>(list), id] {
              if (auto hooks = weak.lock()) hooks->remove(id);
          }) {}

    void remove() {
        if (remove_) {
            remove_();
            remove_ = nullptr;
        }
    }

private:
    std::function<void()> remove_;
};

} // namespace raxis::autograd
