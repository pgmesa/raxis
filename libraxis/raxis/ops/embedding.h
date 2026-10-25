#pragma once

#include <cstdint>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// grad_weight is a zeroed contiguous [num_weights, D], grad a contiguous
// [N, D] and indices N contiguous int64: row i of grad is added to row
// indices[i] of grad_weight
using embedding_dense_backward_fn = void(*)(
    const Tensor& grad_weight, const Tensor& grad, const Tensor& indices, int64_t padding_idx,
    bool scale_grad_by_freq);
DECLARE_DISPATCH(embedding_dense_backward_fn, embedding_dense_backward_stub);

namespace ops {

// From: aten/src/ATen/native/Embedding.cpp (embedding_symint)
RAXIS_API Tensor embedding(
    const Tensor& weight, const Tensor& indices, int64_t padding_idx = -1, bool scale_grad_by_freq = false);
// From: aten/src/ATen/native/Embedding.cpp (embedding_dense_backward_cpu)
RAXIS_API Tensor embedding_dense_backward(
    const Tensor& grad, const Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);

} // namespace ops

/// Rows of weight [num_embeddings, D] picked by indices: the result has
/// shape indices.shape + [D]. The row padding_idx gets no gradient, and with
/// scale_grad_by_freq each row's gradient is divided by how often its index
/// appears
RAXIS_API Tensor embedding(
    const Tensor& weight, const Tensor& indices, int64_t padding_idx = -1, bool scale_grad_by_freq = false);
/// Differentiable through EmbeddingDenseBackwardBackward0
RAXIS_API Tensor embedding_dense_backward(
    const Tensor& grad, const Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq);

} // namespace raxis
