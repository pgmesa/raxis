#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// Kernels take a contiguous input and write a contiguous output of the same
// shape, normalizing along `dim`.
using softmax_fn = void(*)(const Tensor& result, const Tensor& self, int64_t dim);

DECLARE_DISPATCH(softmax_fn, softmax_stub);
DECLARE_DISPATCH(softmax_fn, log_softmax_stub);

// Contiguous grad and forward output in, contiguous grad_input out
using softmax_backward_fn = void(*)(
    const Tensor& grad_input, const Tensor& grad, const Tensor& output, int64_t dim);

DECLARE_DISPATCH(softmax_backward_fn, softmax_backward_stub);
DECLARE_DISPATCH(softmax_backward_fn, log_softmax_backward_stub);

RAXIS_API Tensor softmax(const Tensor& self, int64_t dim);
RAXIS_API Tensor log_softmax(const Tensor& self, int64_t dim);
RAXIS_API Tensor softmax_backward_data(const Tensor& grad, const Tensor& output, int64_t dim);
RAXIS_API Tensor log_softmax_backward_data(const Tensor& grad, const Tensor& output, int64_t dim);

namespace ops {

RAXIS_API Tensor softmax(const Tensor& self, int64_t dim);
RAXIS_API Tensor log_softmax(const Tensor& self, int64_t dim);

// From: aten/src/ATen/native/SoftMax.cpp (_softmax_backward_data,
// _log_softmax_backward_data). Both take the forward output, not the input.
RAXIS_API Tensor softmax_backward_data(const Tensor& grad, const Tensor& output, int64_t dim);
RAXIS_API Tensor log_softmax_backward_data(const Tensor& grad, const Tensor& output, int64_t dim);

} // namespace ops

} // namespace raxis
