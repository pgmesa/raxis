#include <raxis/ops/embedding.h>

#include <cstdint>

#include <raxis/ops/tensor_advanced_indexing.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

namespace {

void check_indices(const Tensor& indices, int arg, const char* op) {
    RAXIS_CHECK(indices.dtype() == dtypes::Int64 || indices.dtype() == dtypes::Int32,
        "Expected tensor for argument #", arg,
        " 'indices' to have one of the following scalar types: Long, Int; but got ", indices.dtype(),
        " instead (while checking arguments for ", op, ")");
}

} // namespace


namespace ops {

// The forward is a gather of whole rows; all the work of the op is in its
// backward
Tensor embedding(const Tensor& weight, const Tensor& indices, int64_t /*padding_idx*/, bool /*scale_grad_by_freq*/) {
    RAXIS_CHECK(weight.ndim() == 2, "'weight' must be 2-D");
    check_indices(indices, 1, "embedding");
    if (indices.ndim() == 1) {
        return ops::index_select(weight, 0, indices);
    }
    DimVector size(indices.shape().begin(), indices.shape().end());
    size.push_back(weight.shape()[1]);
    return ops::index_select(weight, 0, indices.reshape({-1})).view(size);
}

Tensor embedding_dense_backward(
    const Tensor& grad, const Tensor& indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq
) {
    check_indices(indices, 2, "embedding_backward");
    const int64_t numel = indices.numel();
    const int64_t embedding_dim = grad.shape()[grad.ndim() - 1];
    Tensor grad_weight = zeros({num_weights, embedding_dim}, grad.options());
    if (numel > 0 && embedding_dim > 0) {
        embedding_dense_backward_stub(
            DeviceType::CPU, grad_weight, grad.contiguous().view({numel, embedding_dim}),
            indices.to(dtypes::Int64).contiguous(), padding_idx, scale_grad_by_freq);
    }
    return grad_weight;
}

} // namespace ops

DEFINE_DISPATCH(embedding_dense_backward_stub);

} // namespace raxis
