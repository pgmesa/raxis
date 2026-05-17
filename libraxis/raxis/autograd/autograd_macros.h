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
    RAXIS_AUTOGRAD_INPLACE_GUARD_UNARY(op_name)

#define RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(op_name, BackwardNode)              \
    RAXIS_AUTOGRAD_UNARY_SAVE_SELF(op_name, BackwardNode)                      \
    RAXIS_AUTOGRAD_INPLACE_GUARD_UNARY(op_name)

#define RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(op_name, BackwardNode)            \
    RAXIS_AUTOGRAD_UNARY_SAVE_RESULT(op_name, BackwardNode)                    \
    RAXIS_AUTOGRAD_INPLACE_GUARD_UNARY(op_name)

#define RAXIS_AUTOGRAD_UNARY_ALL_SAVE_BOTH(op_name, BackwardNode)              \
    RAXIS_AUTOGRAD_UNARY_SAVE_BOTH(op_name, BackwardNode)                      \
    RAXIS_AUTOGRAD_INPLACE_GUARD_UNARY(op_name)

#define RAXIS_AUTOGRAD_UNARY_ALL_NOT_IMPLEMENTED(op_name)                      \
    RAXIS_AUTOGRAD_UNARY_NOT_IMPLEMENTED(op_name)                              \
    RAXIS_AUTOGRAD_INPLACE_GUARD_UNARY(op_name)

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
    RAXIS_AUTOGRAD_INPLACE_GUARD_UNARY(op_name)

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
    RAXIS_AUTOGRAD_INPLACE_GUARD_BINARY_TENSOR(op_name)                        \
    RAXIS_AUTOGRAD_INPLACE_GUARD_BINARY_SCALAR(op_name)


// ============================================================
// In-place ops: guard against autograd misuse
// ============================================================

// In-place on a leaf tensor with requires_grad is always an error.
// In-place on a non-leaf tensor in the graph is also an error (for now).
// This can be relaxed later with version counters.

#define RAXIS_AUTOGRAD_INPLACE_GUARD_UNARY(op_name)                            \
    Tensor& op_name##_(Tensor& self) {                                         \
        RAXIS_CHECK(!autograd::requires_autograd(self),                        \
            "a leaf Variable that requires grad is being used in an "          \
            "in-place operation (", #op_name, "_)");                           \
        return ops::op_name##_(self);                                          \
    }

#define RAXIS_AUTOGRAD_INPLACE_GUARD_BINARY_TENSOR(op_name)                    \
    Tensor& op_name##_(Tensor& a, const Tensor& b) {                           \
        RAXIS_CHECK(!autograd::requires_autograd(a, b),                        \
            "a leaf Variable that requires grad is being used in an "          \
            "in-place operation (", #op_name, "_)");                           \
        return ops::op_name##_(a, b);                                          \
    }

#define RAXIS_AUTOGRAD_INPLACE_GUARD_BINARY_SCALAR(op_name)                    \
    Tensor& op_name##_(Tensor& a, Scalar s) {                                  \
        RAXIS_CHECK(!autograd::requires_autograd(a),                           \
            "a leaf Variable that requires grad is being used in an "          \
            "in-place operation (", #op_name, "_)");                           \
        return ops::op_name##_(a, s);                                          \
    }