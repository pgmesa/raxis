#include <raxis/ops/embedding.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/vec/functional_base.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/parallel.h>

#include <cstdint>
#include <memory>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

// From: aten/src/ATen/native/Embedding.cpp (embedding_dense_backward_cpu)
//
// Several positions can carry the same index, so their gradient rows add
// into the same row of grad_weight. Instead of atomics or a lock, the
// threads split the rows of grad_weight: each one scans all the indices
// and only adds the ones that fall in its range. Every row is then written
// by one thread, in index order, so the sum is deterministic.
void embedding_dense_backward_kernel(
    const Tensor& grad_weight, const Tensor& grad, const Tensor& indices, int64_t padding_idx,
    bool scale_grad_by_freq
) {
    const int64_t num_weights = grad_weight.shape()[0];
    const int64_t embedding_dim = grad_weight.shape()[1];
    const int64_t numel = indices.numel();
    const int64_t* indices_data = indices.data_ptr<int64_t>();

    std::unique_ptr<int64_t[]> counts;
    if (scale_grad_by_freq) {
        counts = std::make_unique<int64_t[]>(num_weights);
        for (int64_t i = 0; i < numel; ++i) {
            counts[indices_data[i]]++;
        }
    }

    grad.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using Vec = Vectorized<scalar_t>;
        scalar_t* grad_weight_data = grad_weight.mutable_data_ptr<scalar_t>();
        const scalar_t* grad_data = grad.data_ptr<scalar_t>();
        parallel_for(0, num_weights, 1000, [&](int64_t start, int64_t end) {
            for (int64_t i = 0; i < numel; ++i) {
                const int64_t k = indices_data[i];
                if (k == padding_idx || k < start || k >= end) {
                    continue;
                }
                // grad_weight[k] += scale * grad[i], as torch's add_ with alpha
                const Vec scale(static_cast<scalar_t>(scale_grad_by_freq ? 1.0 / counts[k] : 1.0));
                scalar_t* row = grad_weight_data + k * embedding_dim;
                map2([scale](Vec acc, Vec g) { return acc + scale * g; },
                     row, row, grad_data + i * embedding_dim, embedding_dim);
            }
        });
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(embedding_dense_backward_stub, &embedding_dense_backward_kernel);

} // namespace raxis
