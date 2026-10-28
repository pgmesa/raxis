#include <bit>
#include <cstdint>
#include <limits>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>

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
// Its parallelism is the embedding dim / 32 blocks, which is little for
// many indices, so above 3072 indices both PyTorch and raxis sort them
// instead (embedding_backward_sorted below).
// scale_grad_by_freq divides each row by the count of its index, counted
// first with integer atomics (exact, so still deterministic).

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

// The sorted path.
// From: aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu (embedding_backward_cuda_kernel)
//
// Sorting the indices (stably, keeping each one's position) turns the rows
// that add into one grad_weight row into a run, a segment, of consecutive
// sorted positions. A segment can be long (a frequent token), so it is cut
// into partial segments of at most kRowsPerPartial positions: one thread
// per (partial segment, feature) sums its rows in order, then one thread
// per (segment, feature) sums its partials in order and writes the row.
// Every sum has a fixed order, so the result is deterministic. The segment
// and partial counts stay on the GPU (the kernels read them through
// pointers and launch for the most there can be), so there is no sync.
// PyTorch switches to atomic adds of the partials when there are few
// segments (unless deterministic algorithms are on); raxis does not.
constexpr int kRowsPerPartial = 10;
constexpr int kMaxBlock = 1024;

__global__ void iota_kernel(int32_t* out, int32_t n) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i;
    }
}

__global__ void segment_heads_kernel(bool* heads, const int64_t* sorted, int32_t n) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        heads[i] = i == 0 || sorted[i] != sorted[i - 1];
    }
}

__global__ void partials_per_segment_kernel(int32_t* partials, const int32_t* segment_offsets,
                                            const int32_t* num_segments, int32_t n) {
    const int32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s < *num_segments) {
        const int32_t end = s == *num_segments - 1 ? n : segment_offsets[s + 1];
        partials[s] = (end - segment_offsets[s] + kRowsPerPartial - 1) / kRowsPerPartial;
    }
}

__global__ void partial_offsets_kernel(int32_t* partial_offsets, int32_t* num_partials, const int32_t* partials,
                                       const int32_t* first_partial, const int32_t* segment_offsets,
                                       const int32_t* num_segments) {
    const int32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= *num_segments) {
        return;
    }
    for (int32_t i = 0; i < partials[s]; i++) {
        partial_offsets[first_partial[s] + i] = segment_offsets[s] + i * kRowsPerPartial;
    }
    if (s == *num_segments - 1) {
        *num_partials = first_partial[s] + partials[s];
    }
}

template <typename scalar_t>
__global__ void partial_sums_kernel(scalar_t* partial_sums, const scalar_t* grad, const int64_t* sorted,
                                    const int32_t* positions, const int32_t* counts, const int32_t* partial_offsets,
                                    const int32_t* num_partials, int32_t n, int32_t stride, int32_t stride_warped) {
    const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t p = static_cast<int32_t>(gid / stride_warped);
    const int32_t f = static_cast<int32_t>(gid % stride_warped);
    if (f >= stride || p >= *num_partials) {
        return;
    }
    const int32_t end = p == *num_partials - 1 ? n : partial_offsets[p + 1];
    scalar_t sum = 0;
    for (int32_t i = partial_offsets[p]; i < end; i++) {
        scalar_t value = grad[static_cast<int64_t>(positions[i]) * stride + f];
        if (counts != nullptr) {
            value /= static_cast<scalar_t>(counts[sorted[i]]);
        }
        sum += value;
    }
    partial_sums[static_cast<int64_t>(p) * stride + f] = sum;
}

template <typename scalar_t>
__global__ void sum_partials_kernel(scalar_t* grad_weight, const scalar_t* partial_sums, const int64_t* sorted,
                                    const int32_t* segment_offsets, const int32_t* first_partial,
                                    const int32_t* num_segments, const int32_t* num_partials, int32_t stride,
                                    int32_t stride_warped, int64_t num_weights, int64_t padding_idx) {
    const int64_t gid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int32_t s = static_cast<int32_t>(gid / stride_warped);
    const int32_t f = static_cast<int32_t>(gid % stride_warped);
    if (f >= stride || s >= *num_segments) {
        return;
    }
    const int64_t row = sorted[segment_offsets[s]];
    RAXIS_CUDA_KERNEL_ASSERT(row >= 0 && row < num_weights);
    if (row == padding_idx) {
        return;
    }
    const int32_t end = s == *num_segments - 1 ? *num_partials : first_partial[s + 1];
    scalar_t sum = 0;
    for (int32_t p = first_partial[s]; p < end; p++) {
        sum += partial_sums[static_cast<int64_t>(p) * stride + f];
    }
    grad_weight[row * stride + f] = sum;
}

template <typename scalar_t>
void embedding_backward_sorted(const Tensor& grad_weight, const Tensor& grad, const Tensor& indices,
                               const Tensor& counts, int64_t padding_idx) {
    const auto n = static_cast<int32_t>(indices.numel());
    const auto stride = static_cast<int32_t>(grad.shape()[1]);
    const int64_t num_weights = grad_weight.shape()[0];
    const cudaStream_t stream = cuda::current_stream();
    const TensorOptions int32 = grad.options().dtype(dtypes::Int32);
    const TensorOptions bytes = grad.options().dtype(dtypes::Uint8);
    const auto blocks = [](int64_t count, int threads) { return static_cast<unsigned>((count + threads - 1) / threads); };

    // Sort the indices, keeping their positions (stable: equal indices stay in order)
    Tensor sorted = empty({n}, indices.options());
    Tensor positions = empty({n}, int32);
    Tensor sorted_positions = empty({n}, int32);
    iota_kernel<<<blocks(n, 256), 256, 0, stream>>>(positions.mutable_data_ptr<int32_t>(), n);
    {
        const int end_bit = std::max(1, static_cast<int>(std::bit_width(static_cast<uint64_t>(num_weights))));
        size_t temp_bytes = 0;
        auto sort = [&](void* temp) {
            RAXIS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp, temp_bytes, indices.data_ptr<int64_t>(),
                sorted.mutable_data_ptr<int64_t>(), positions.data_ptr<int32_t>(), sorted_positions.mutable_data_ptr<int32_t>(), n, 0,
                end_bit, stream));
        };
        sort(nullptr);
        Tensor temp = empty({static_cast<int64_t>(temp_bytes)}, bytes);
        sort(temp.mutable_data_ptr());
    }

    // Segments: the sorted positions where a new index starts
    const int64_t max_segments = std::min<int64_t>(n, num_weights);
    Tensor heads = empty({n}, grad.options().dtype(dtypes::Bool));
    Tensor segment_offsets = empty({n}, int32);
    Tensor counters = empty({2}, int32);   // number of segments, of partials
    int32_t* num_segments = counters.mutable_data_ptr<int32_t>();
    int32_t* num_partials = num_segments + 1;
    segment_heads_kernel<<<blocks(n, 256), 256, 0, stream>>>(heads.mutable_data_ptr<bool>(), sorted.data_ptr<int64_t>(), n);
    {
        const thrust::counting_iterator<int32_t> iota(0);
        size_t temp_bytes = 0;
        auto select = [&](void* temp) {
            RAXIS_CUDA_CHECK(cub::DeviceSelect::Flagged(temp, temp_bytes, iota, heads.data_ptr<bool>(),
                segment_offsets.mutable_data_ptr<int32_t>(), num_segments, n, stream));
        };
        select(nullptr);
        Tensor temp = empty({static_cast<int64_t>(temp_bytes)}, bytes);
        select(temp.mutable_data_ptr());
    }

    // Partial segments: how many per segment, where each segment's start
    // (an exclusive prefix sum) and where each partial starts
    Tensor partials = zeros({max_segments}, int32);
    Tensor first_partial = empty({max_segments}, int32);
    partials_per_segment_kernel<<<blocks(max_segments, 128), 128, 0, stream>>>(
        partials.mutable_data_ptr<int32_t>(), segment_offsets.data_ptr<int32_t>(), num_segments, n);
    {
        size_t temp_bytes = 0;
        auto scan = [&](void* temp) {
            RAXIS_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, temp_bytes, partials.data_ptr<int32_t>(),
                first_partial.mutable_data_ptr<int32_t>(), static_cast<int>(max_segments), stream));
        };
        scan(nullptr);
        Tensor temp = empty({static_cast<int64_t>(temp_bytes)}, bytes);
        scan(temp.mutable_data_ptr());
    }
    const int64_t max_partials = n / kRowsPerPartial + max_segments;
    Tensor partial_offsets = empty({max_partials}, int32);
    partial_offsets_kernel<<<blocks(max_segments, 128), 128, 0, stream>>>(
        partial_offsets.mutable_data_ptr<int32_t>(), num_partials, partials.data_ptr<int32_t>(),
        first_partial.data_ptr<int32_t>(), segment_offsets.data_ptr<int32_t>(), num_segments);

    // The sums: partials, then segments
    const int32_t stride_warped = (stride + kWarpSize - 1) / kWarpSize * kWarpSize;
    const int block = std::min(stride_warped, kMaxBlock);
    Tensor partial_sums = empty({max_partials, stride}, grad.options());
    partial_sums_kernel<scalar_t><<<blocks(max_partials * stride_warped, block), block, 0, stream>>>(
        partial_sums.mutable_data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), sorted.data_ptr<int64_t>(),
        sorted_positions.data_ptr<int32_t>(), counts.defined() ? counts.data_ptr<int32_t>() : nullptr,
        partial_offsets.data_ptr<int32_t>(), num_partials, n, stride, stride_warped);
    sum_partials_kernel<scalar_t><<<blocks(max_segments * stride_warped, block), block, 0, stream>>>(
        grad_weight.mutable_data_ptr<scalar_t>(), partial_sums.data_ptr<scalar_t>(), sorted.data_ptr<int64_t>(),
        segment_offsets.data_ptr<int32_t>(), first_partial.data_ptr<int32_t>(), num_segments, num_partials, stride,
        stride_warped, num_weights, padding_idx);
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
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
        // From: Embedding.cu (embedding_dense_backward_cuda): the feature kernel up to 3072 indices
        if (n > 3072) {
            embedding_backward_sorted<scalar_t>(grad_weight, grad, indices, counts, padding_idx);
            return;
        }
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
