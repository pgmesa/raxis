#pragma once

// Macros for wrapping ops with autograd support.

#include <raxis/tensor/tensor.h>
#include <raxis/core/scalar.h>
#include <raxis/ops/scalar_ops.h>

#include <raxis/autograd/engine.h>
#include <raxis/autograd/variable.h>
#include <raxis/autograd/functions/common.h>

#include <raxis/util/exception.h>


// ============================================================
// Unary ops: f(Tensor) -> Tensor
// ============================================================

#define RAXIS_AUTOGRAD_UNARY_IMPL(op_name, BackwardNode, ...)                  \
    Tensor op_name(const Tensor& self) {                                       \
        Tensor result = ops::op_name(self);                                    \
        if (autograd::requires_autograd(self)) {                               \
            auto grad_fn = std::make_shared<autograd::BackwardNode>(           \
                __VA_ARGS__);                                                  \
            grad_fn->set_num_inputs(1);                                        \
            autograd::set_next_edges(grad_fn, {self});                           \
            autograd::set_autograd_meta(result, grad_fn);                      \
        }                                                                      \
        return result;                                                         \
    }

// Backward node receives (self)
#define RAXIS_AUTOGRAD_UNARY_SAVE_SELF(op_name, BackwardNode)                  \
    RAXIS_AUTOGRAD_UNARY_IMPL(op_name, BackwardNode, self)

// Backward node receives (result)
#define RAXIS_AUTOGRAD_UNARY_SAVE_RESULT(op_name, BackwardNode)                \
    RAXIS_AUTOGRAD_UNARY_IMPL(op_name, BackwardNode, result)

// Backward node receives (self, result)
#define RAXIS_AUTOGRAD_UNARY_SAVE_BOTH(op_name, BackwardNode)                  \
    RAXIS_AUTOGRAD_UNARY_IMPL(op_name, BackwardNode, self, result)

// No-args backward node (e.g. NegBackward0 needs nothing)
#define RAXIS_AUTOGRAD_UNARY_SAVE_NONE(op_name, BackwardNode)                  \
    RAXIS_AUTOGRAD_UNARY_IMPL(op_name, BackwardNode)


#define RAXIS_AUTOGRAD_UNARY_NOT_IMPLEMENTED(op_name)                          \
    Tensor op_name(const Tensor& self) {                                       \
        Tensor result = ops::op_name(self);                                    \
        if (autograd::requires_autograd(self)) {                               \
            auto grad_fn = std::make_shared<autograd::NotImplementedBackward>( \
                #op_name);                                                     \
            grad_fn->set_num_inputs(1);                                        \
            autograd::set_next_edges(grad_fn, self);                           \
            autograd::set_autograd_meta(result, grad_fn);                      \
        }                                                                      \
        return result;                                                         \
    }

#define RAXIS_AUTOGRAD_UNARY_ALL_SAVE_NONE(op_name, BackwardNode)              \
    RAXIS_AUTOGRAD_UNARY_SAVE_NONE(op_name, BackwardNode)                      \
    RAXIS_AUTOGRAD_INPLACE_UNARY(op_name, BackwardNode)

#define RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(op_name, BackwardNode)              \
    RAXIS_AUTOGRAD_UNARY_SAVE_SELF(op_name, BackwardNode)                      \
    RAXIS_AUTOGRAD_INPLACE_UNARY(op_name, BackwardNode, self)

#define RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(op_name, BackwardNode)            \
    RAXIS_AUTOGRAD_UNARY_SAVE_RESULT(op_name, BackwardNode)                    \
    RAXIS_AUTOGRAD_INPLACE_UNARY_SAVE_RESULT(op_name, BackwardNode)

#define RAXIS_AUTOGRAD_UNARY_ALL_NOT_IMPLEMENTED(op_name)                      \
    RAXIS_AUTOGRAD_UNARY_NOT_IMPLEMENTED(op_name)                              \
    RAXIS_AUTOGRAD_INPLACE_UNARY(op_name, NotImplementedBackward, #op_name)

#define RAXIS_AUTOGRAD_UNARY_ZERO_GRAD(op_name, grad_name)                     \
    Tensor op_name(const Tensor& self) {                                       \
        Tensor result = ops::op_name(self);                                    \
        if (autograd::requires_autograd(self)) {                               \
            auto grad_fn = std::make_shared<autograd::ZeroBackward0>(          \
                grad_name);                                                    \
            grad_fn->set_num_inputs(1);                                        \
            autograd::set_next_edges(grad_fn, self);                           \
            autograd::set_autograd_meta(result, grad_fn);                      \
        }                                                                      \
        return result;                                                         \
    }                                                                          \
    RAXIS_AUTOGRAD_INPLACE_UNARY(op_name, ZeroBackward0, grad_name)

// ============================================================
// Binary ops: f(Tensor, Tensor) -> Tensor
// ============================================================

#define RAXIS_AUTOGRAD_BINARY(op_name, BackwardNode)                           \
    Tensor op_name(const Tensor& a, const Tensor& b) {                         \
        Tensor result = ops::op_name(a, b);                                    \
        if (autograd::requires_autograd(a, b)) {                               \
            auto grad_fn = std::make_shared<autograd::BackwardNode>(a, b);     \
            grad_fn->set_num_inputs(1);                                        \
            autograd::set_next_edges(grad_fn, a, b);                           \
            autograd::set_autograd_meta(result, grad_fn);                      \
        }                                                                      \
        return result;                                                         \
    }

// Binary op with Scalar overload: wraps scalar, reuses same backward node
#define RAXIS_AUTOGRAD_BINARY_SCALAR(op_name, BackwardNode)                    \
    Tensor op_name(const Tensor& a, Scalar s) {                                \
        Tensor b = wrapped_scalar_tensor(s);                                   \
        Tensor result = ops::op_name(a, b);                                    \
        if (autograd::requires_autograd(a)) {                                  \
            auto grad_fn = std::make_shared<autograd::BackwardNode>(a, b);     \
            grad_fn->set_num_inputs(1);                                        \
            autograd::set_next_edges(grad_fn, a, b);                           \
            autograd::set_autograd_meta(result, grad_fn);                      \
        }                                                                      \
        return result;                                                         \
    }

#define RAXIS_AUTOGRAD_BINARY_NOT_IMPLEMENTED(op_name)                         \
    Tensor op_name(const Tensor& a, const Tensor& b) {                         \
        Tensor result = ops::op_name(a, b);                                    \
        if (autograd::requires_autograd(a, b)) {                               \
            auto grad_fn = std::make_shared<autograd::NotImplementedBackward>( \
                #op_name);                                                     \
            grad_fn->set_num_inputs(1);                                        \
            autograd::set_next_edges(grad_fn, a, b);                           \
            autograd::set_autograd_meta(result, grad_fn);                      \
        }                                                                      \
        return result;                                                         \
    }

#define RAXIS_AUTOGRAD_BINARY_ALL(op_name, BackwardNode)                       \
    RAXIS_AUTOGRAD_BINARY(op_name, BackwardNode)                               \
    RAXIS_AUTOGRAD_BINARY_SCALAR(op_name, BackwardNode)                        \
    RAXIS_AUTOGRAD_INPLACE_BINARY(op_name, BackwardNode)                       \
    RAXIS_AUTOGRAD_INPLACE_BINARY_SCALAR(op_name, BackwardNode)


// ============================================================
// In-place ops (the steps are described in autograd/variable.h)
// ============================================================

// The node is built from the inputs before the kernel overwrites self;
// __VA_ARGS__ are its constructor arguments
#define RAXIS_AUTOGRAD_INPLACE_UNARY(op_name, BackwardNode, ...)               \
    Tensor& op_name##_(Tensor& self) {                                         \
        const bool requires_grad = autograd::requires_autograd(self);          \
        autograd::check_inplace(self, requires_grad);                          \
        std::shared_ptr<autograd::BackwardNode> grad_fn;                       \
        if (requires_grad) {                                                   \
            grad_fn = std::make_shared<autograd::BackwardNode>(__VA_ARGS__);   \
            grad_fn->set_num_inputs(1);                                        \
            autograd::set_next_edges(grad_fn, self);                           \
            autograd::save_original_self(*grad_fn, self);                      \
        }                                                                      \
        ops::op_name##_(self);                                                 \
        autograd::finish_inplace(self, grad_fn);                               \
        return self;                                                           \
    }

// The node saves the result: it is built after the kernel, from self at its
// new version (the edge to the old history is taken before)
#define RAXIS_AUTOGRAD_INPLACE_UNARY_SAVE_RESULT(op_name, BackwardNode)        \
    Tensor& op_name##_(Tensor& self) {                                         \
        const bool requires_grad = autograd::requires_autograd(self);          \
        autograd::check_inplace(self, requires_grad);                          \
        autograd::Edge self_edge;                                              \
        if (requires_grad) {                                                   \
            self_edge = autograd::gradient_edge(self);                         \
        }                                                                      \
        ops::op_name##_(self);                                                 \
        autograd::bump_version(self);                                          \
        if (requires_grad) {                                                   \
            auto grad_fn = std::make_shared<autograd::BackwardNode>(self);     \
            grad_fn->set_num_inputs(1);                                        \
            grad_fn->add_next_edge(std::move(self_edge));                      \
            autograd::rebase_history(self, grad_fn);                           \
        }                                                                      \
        return self;                                                           \
    }

#define RAXIS_AUTOGRAD_INPLACE_BINARY(op_name, BackwardNode)                   \
    Tensor& op_name##_(Tensor& a, const Tensor& b) {                           \
        const bool requires_grad = autograd::requires_autograd(a, b);          \
        autograd::check_inplace(a, requires_grad);                             \
        std::shared_ptr<autograd::BackwardNode> grad_fn;                       \
        if (requires_grad) {                                                   \
            grad_fn = std::make_shared<autograd::BackwardNode>(a, b);          \
            grad_fn->set_num_inputs(1);                                        \
            autograd::set_next_edges(grad_fn, a, b);                           \
            autograd::save_original_self(*grad_fn, a);                         \
        }                                                                      \
        ops::op_name##_(a, b);                                                 \
        autograd::finish_inplace(a, grad_fn);                                  \
        return a;                                                              \
    }

#define RAXIS_AUTOGRAD_INPLACE_BINARY_SCALAR(op_name, BackwardNode)            \
    Tensor& op_name##_(Tensor& a, Scalar s) {                                  \
        const bool requires_grad = autograd::requires_autograd(a);             \
        autograd::check_inplace(a, requires_grad);                             \
        std::shared_ptr<autograd::BackwardNode> grad_fn;                       \
        if (requires_grad) {                                                   \
            Tensor b = wrapped_scalar_tensor(s);                               \
            grad_fn = std::make_shared<autograd::BackwardNode>(a, b);          \
            grad_fn->set_num_inputs(1);                                        \
            autograd::set_next_edges(grad_fn, a, b);                           \
            autograd::save_original_self(*grad_fn, a);                         \
        }                                                                      \
        ops::op_name##_(a, s);                                                 \
        autograd::finish_inplace(a, grad_fn);                                  \
        return a;                                                              \
    }

// In-place ops without a derivative: they raise if autograd would have to
// record them, and otherwise only bump the version
#define RAXIS_AUTOGRAD_INPLACE_NO_GRAD_BINARY_TENSOR(op_name)                  \
    Tensor& op_name##_(Tensor& a, const Tensor& b) {                           \
        RAXIS_CHECK(!autograd::requires_autograd(a, b),                        \
            "the derivative for '", #op_name, "_' is not implemented");        \
        ops::op_name##_(a, b);                                                 \
        autograd::bump_version(a);                                             \
        return a;                                                              \
    }

#define RAXIS_AUTOGRAD_INPLACE_NO_GRAD_BINARY_SCALAR(op_name)                  \
    Tensor& op_name##_(Tensor& a, Scalar s) {                                  \
        RAXIS_CHECK(!autograd::requires_autograd(a),                           \
            "the derivative for '", #op_name, "_' is not implemented");        \
        ops::op_name##_(a, s);                                                 \
        autograd::bump_version(a);                                             \
        return a;                                                              \
    }
