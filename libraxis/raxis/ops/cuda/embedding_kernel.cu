#include <cstdint>
#include <limits>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/embedding.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


// Backward of embedding: grad_weight[indices[i]] += grad[i] for every row i.
// From: aten/src/ATen/native/cuda/Embedding.cu (embedding_backward_feature_kernel)
//
// Rows that share an index add into the same grad_weight row. Atomic adds
// would make the float sum order (and so the result) change between runs.
// PyTorch's feature kernel stays deterministic without sorting: a block owns
// 32 features (one per lane) and walks the indices in chunks of 32, one per
// warp. In each chunk, warps whose index repeats find each other with a
// ballot over the chunk's indices; the lowest warp of each group adds the
// others' rows, in warp order, from shared memory, and only it writes to
// grad_weight. Every row is added in index order, whatever the scheduling.
//
// PyTorch uses this kernel up to 3072 indices and sorts the indices for more
// (a segmented reduction over runs of equal indices). raxis uses it for all
// sizes: its parallelism is the embedding dim / 32 blocks, which is little
// for small dims (a candidate to revisit with a sort-based path).
// scale_grad_by_freq divides each row by the count of its index, counted
// first with integer atomics (exact, so still deterministic); PyTorch only
// supports it on the sort path.

namespace raxis {
namespace {

constexpr int kWarpSize = 32;
constexpr int kBlockDimY = 32;

__global__ void count_indices_kernel(int32_t* counts, const int64_t* indices, int32_t n) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&counts[indices[i]], 1);
    }
}

template <typename scalar_t>
__global__ void embedding_backward_feature_kernel(
    scalar_t* grad_weight, const scalar_t* grad, const int64_t* indices, const int32_t* counts,
    int32_t n, int32_t stride, int64_t num_weights, int64_t padding_idx
) {
    __shared__ scalar_t smem[kBlockDimY][kWarpSize];
    __shared__ int64_t indices_batch[kWarpSize * kBlockDimY];
    scalar_t* my_s = smem[threadIdx.y];
    const int32_t f = threadIdx.x + blockIdx.x * blockDim.x;

    for (int32_t batch_start = 0; batch_start < n; batch_start += kWarpSize * kBlockDimY) {
        // The block loads a batch of up to 1024 indices
        const int32_t tid = threadIdx.x + threadIdx.y * kWarpSize;
        if (batch_start + tid < n) {
            indices_batch[tid] = indices[batch_start + tid];
        }
        const int32_t batch_end = ::min(batch_start + kWarpSize * kBlockDimY, n);

        // and walks it in chunks of one index per warp
        for (int32_t chunk_start = batch_start; chunk_start < batch_end; chunk_start += kBlockDimY) {
            // Makes indices_batch visible, and keeps the previous chunk's
            // leaders from being overwritten before they are done
            __syncthreads();
            const int32_t n_this_chunk = ::min(batch_end - chunk_start, kBlockDimY);
            const int32_t src_row = chunk_start + threadIdx.y;
            const int64_t dst_row = src_row < n ? indices_batch[src_row - batch_start] : padding_idx;
            if (src_row < n) {
                RAXIS_CUDA_KERNEL_ASSERT(dst_row >= 0 && dst_row < num_weights);
            }
            if (src_row < n && f < stride && dst_row != padding_idx) {
                scalar_t value = grad[static_cast<int64_t>(src_row) * stride + f];
                if (counts != nullptr) {
                    value /= static_cast<scalar_t>(counts[dst_row]);
                }
                my_s[threadIdx.x] = value;
            }
            __syncthreads();

            if (src_row < n && dst_row != padding_idx) {
                // Lane l of this warp checks whether warp l of the chunk has
                // the same index; the ballot gathers the answers in a mask
                const bool match = threadIdx.x < n_this_chunk
                    && dst_row == indices_batch[chunk_start - batch_start + threadIdx.x];
                unsigned mask = __ballot_sync(0xffffffff, match);
                int first = __ffs(mask) - 1;
                if (static_cast<int>(threadIdx.y) == first) {
                    mask ^= 1u << first;
                    while (mask) {
                        first = __ffs(mask) - 1;
                        my_s[threadIdx.x] += smem[first][threadIdx.x];
                        mask ^= 1u << first;
                    }
                    if (f < stride) {
                        grad_weight[dst_row * stride + f] += my_s[threadIdx.x];
                    }
                }
            }
        }
    }
}

void embedding_dense_backward_kernel_cuda(
    const Tensor& grad_weight, const Tensor& grad, const Tensor& indices, int64_t padding_idx,
    bool scale_grad_by_freq
) {
    DeviceGuard guard(grad.device());
    const int64_t n = indices.numel();
    const int64_t stride = grad.shape()[1];
    const int64_t num_weights = grad_weight.shape()[0];
    RAXIS_CHECK(n <= std::numeric_limits<int32_t>::max() && stride <= std::numeric_limits<int32_t>::max(),
        "embedding_backward: sizes over 2^31 are not supported on CUDA yet");
    cudaStream_t stream = cuda::current_stream();

    Tensor counts;
    if (scale_grad_by_freq) {
        counts = zeros({num_weights}, grad.options().dtype(dtypes::Int32));
        count_indices_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
            counts.mutable_data_ptr<int32_t>(), indices.data_ptr<int64_t>(), static_cast<int32_t>(n));
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const dim3 grid(static_cast<unsigned>((stride + kWarpSize - 1) / kWarpSize));
    const dim3 block(kWarpSize, kBlockDimY);
    grad.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        embedding_backward_feature_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_weight.mutable_data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), indices.data_ptr<int64_t>(),
            counts.defined() ? counts.data_ptr<int32_t>() : nullptr,
            static_cast<int32_t>(n), static_cast<int32_t>(stride), num_weights, padding_idx);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // anonymous namespace

REGISTER_DISPATCH(embedding_dense_backward_stub, &embedding_dense_backward_kernel_cuda);

} // namespace raxis
