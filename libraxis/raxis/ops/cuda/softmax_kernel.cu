#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/softmax.h>
#include <raxis/tensor/tensor.h>


// softmax / log_softmax and their backward on the GPU.
// From: aten/src/ATen/native/cuda/SoftMax.cu and PersistentSoftmax.cuh,
// simplified. The front-end passes contiguous tensors, seen here as
// (outer, dim_size, inner):
//
//   - inner == 1 and dim_size <= 1024 (the classifier and attention case):
//     one warp per row, the whole row in registers (PyTorch's "persistent"
//     softmax). The max and the sum are warp reductions with xor shuffles,
//     a butterfly that leaves the result in every lane, so the row is read
//     from global memory once and written once.
//   - inner == 1 and longer rows: one block per row, reducing through
//     shared memory, and reading the row again for each pass.
//   - inner > 1 (softmax over a dim that is not the last): one thread per
//     (outer, inner) column, walking the dim. Neighbouring threads take
//     neighbouring columns, so every load of the walk is coalesced.
//
// All of them compute max(x), then sum(exp(x - max)), as the CPU kernel.

namespace raxis {
namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxWarpSoftmaxSize = 1024;

// Butterfly reductions: after log2(32) steps every lane holds the result
template <typename T>
__device__ T warp_max(T value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        const T other = __shfl_xor_sync(0xffffffff, value, offset);
        value = value > other ? value : other;
    }
    return value;
}

template <typename T>
__device__ T warp_sum(T value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    return value;
}

// ---------------------------------------------------------------------------
// One warp per row: lane l holds elements l, l + 32, ... (kIterations of them)
// ---------------------------------------------------------------------------

template <typename scalar_t, int kIterations, bool is_log>
__global__ void softmax_warp_forward(scalar_t* dst, const scalar_t* src, int64_t rows, int dim_size) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * (blockDim.x / kWarpSize) + threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    if (row >= rows) {
        return;
    }
    src += row * dim_size;
    dst += row * dim_size;

    scalar_t elements[kIterations];
    scalar_t max_value = -std::numeric_limits<scalar_t>::infinity();
    #pragma unroll
    for (int i = 0; i < kIterations; ++i) {
        const int col = lane + i * kWarpSize;
        elements[i] = col < dim_size ? src[col] : -std::numeric_limits<scalar_t>::infinity();
        max_value = elements[i] > max_value ? elements[i] : max_value;
    }
    max_value = warp_max(max_value);

    scalar_t sum = 0;
    #pragma unroll
    for (int i = 0; i < kIterations; ++i) {
        if (lane + i * kWarpSize < dim_size) {
            sum += ::exp(elements[i] - max_value);
        }
    }
    sum = warp_sum(sum);

    const scalar_t log_sum = ::log(sum);
    #pragma unroll
    for (int i = 0; i < kIterations; ++i) {
        const int col = lane + i * kWarpSize;
        if (col < dim_size) {
            dst[col] = is_log ? elements[i] - max_value - log_sum : ::exp(elements[i] - max_value) / sum;
        }
    }
}

// softmax: gI = y * (g - sum(g * y)); log_softmax: gI = g - exp(y) * sum(g)
template <typename scalar_t, int kIterations, bool is_log>
__global__ void softmax_warp_backward(
    scalar_t* grad_input, const scalar_t* grad, const scalar_t* output, int64_t rows, int dim_size
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * (blockDim.x / kWarpSize) + threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    if (row >= rows) {
        return;
    }
    grad += row * dim_size;
    output += row * dim_size;
    grad_input += row * dim_size;

    scalar_t g[kIterations];
    scalar_t y[kIterations];
    scalar_t sum = 0;
    #pragma unroll
    for (int i = 0; i < kIterations; ++i) {
        const int col = lane + i * kWarpSize;
        const bool valid = col < dim_size;
        g[i] = valid ? grad[col] : scalar_t(0);
        y[i] = valid ? output[col] : scalar_t(0);
        sum += is_log ? g[i] : g[i] * y[i];
    }
    sum = warp_sum(sum);

    #pragma unroll
    for (int i = 0; i < kIterations; ++i) {
        const int col = lane + i * kWarpSize;
        if (col < dim_size) {
            grad_input[col] = is_log ? g[i] - ::exp(y[i]) * sum : y[i] * (g[i] - sum);
        }
    }
}

// kIterations is a compile-time constant, so the row lives in registers:
// one instantiation per power of two up to 1024 / 32
template <typename scalar_t, bool is_log, typename Launch>
void dispatch_warp_iterations(int dim_size, const Launch& launch) {
    const int iterations = (dim_size + kWarpSize - 1) / kWarpSize;
    if (iterations <= 1) launch.template operator()<1>();
    else if (iterations <= 2) launch.template operator()<2>();
    else if (iterations <= 4) launch.template operator()<4>();
    else if (iterations <= 8) launch.template operator()<8>();
    else if (iterations <= 16) launch.template operator()<16>();
    else launch.template operator()<32>();
}

// ---------------------------------------------------------------------------
// One block per row
// ---------------------------------------------------------------------------

constexpr int kBlockThreads = 512;

// The warp results go through shared memory to the first warp, which
// reduces them; the result is broadcast back through shared[0]
template <bool is_max, typename T>
__device__ T block_reduce(T value, T* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = is_max ? warp_max(value) : warp_sum(value);
    __syncthreads();  // shared may still be read by the previous reduction
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        const int num_warps = blockDim.x / kWarpSize;
        const T ident = is_max ? -std::numeric_limits<T>::infinity() : T(0);
        value = lane < num_warps ? shared[lane] : ident;
        value = is_max ? warp_max(value) : warp_sum(value);
        if (lane == 0) {
            shared[0] = value;
        }
    }
    __syncthreads();
    return shared[0];
}

template <typename scalar_t, bool is_log>
__global__ void softmax_block_forward(scalar_t* dst, const scalar_t* src, int dim_size) {
    __shared__ scalar_t shared[kBlockThreads / kWarpSize];
    src += static_cast<int64_t>(blockIdx.x) * dim_size;
    dst += static_cast<int64_t>(blockIdx.x) * dim_size;

    scalar_t max_value = -std::numeric_limits<scalar_t>::infinity();
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        max_value = src[i] > max_value ? src[i] : max_value;
    }
    max_value = block_reduce<true>(max_value, shared);

    scalar_t sum = 0;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum += ::exp(src[i] - max_value);
    }
    sum = block_reduce<false>(sum, shared);

    const scalar_t log_sum = ::log(sum);
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        dst[i] = is_log ? src[i] - max_value - log_sum : ::exp(src[i] - max_value) / sum;
    }
}

template <typename scalar_t, bool is_log>
__global__ void softmax_block_backward(
    scalar_t* grad_input, const scalar_t* grad, const scalar_t* output, int dim_size
) {
    __shared__ scalar_t shared[kBlockThreads / kWarpSize];
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * dim_size;
    grad += offset;
    output += offset;
    grad_input += offset;

    scalar_t sum = 0;
    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        sum += is_log ? grad[i] : grad[i] * output[i];
    }
    sum = block_reduce<false>(sum, shared);

    for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        grad_input[i] = is_log ? grad[i] - ::exp(output[i]) * sum : output[i] * (grad[i] - sum);
    }
}

// ---------------------------------------------------------------------------
// inner > 1: one thread per (outer, inner) column
// ---------------------------------------------------------------------------

template <typename scalar_t, bool is_log>
__global__ void softmax_column_forward(
    scalar_t* dst, const scalar_t* src, int64_t outer, int64_t dim_size, int64_t inner
) {
    const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= outer * inner) {
        return;
    }
    const int64_t base = (column / inner) * dim_size * inner + column % inner;
    scalar_t max_value = -std::numeric_limits<scalar_t>::infinity();
    for (int64_t d = 0; d < dim_size; ++d) {
        const scalar_t x = src[base + d * inner];
        max_value = x > max_value ? x : max_value;
    }
    scalar_t sum = 0;
    for (int64_t d = 0; d < dim_size; ++d) {
        sum += ::exp(src[base + d * inner] - max_value);
    }
    const scalar_t log_sum = ::log(sum);
    for (int64_t d = 0; d < dim_size; ++d) {
        const scalar_t x = src[base + d * inner];
        dst[base + d * inner] = is_log ? x - max_value - log_sum : ::exp(x - max_value) / sum;
    }
}

template <typename scalar_t, bool is_log>
__global__ void softmax_column_backward(
    scalar_t* grad_input, const scalar_t* grad, const scalar_t* output,
    int64_t outer, int64_t dim_size, int64_t inner
) {
    const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= outer * inner) {
        return;
    }
    const int64_t base = (column / inner) * dim_size * inner + column % inner;
    scalar_t sum = 0;
    for (int64_t d = 0; d < dim_size; ++d) {
        const int64_t i = base + d * inner;
        sum += is_log ? grad[i] : grad[i] * output[i];
    }
    for (int64_t d = 0; d < dim_size; ++d) {
        const int64_t i = base + d * inner;
        grad_input[i] = is_log ? grad[i] - ::exp(output[i]) * sum : output[i] * (grad[i] - sum);
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct SoftmaxShape {
    int64_t outer = 1;
    int64_t dim_size = 1;
    int64_t inner = 1;
};

SoftmaxShape softmax_shape(const Tensor& t, int64_t dim) {
    SoftmaxShape s;
    const IntArrayRef shape = t.shape();
    s.dim_size = shape[dim];
    for (int64_t d = 0; d < dim; ++d) s.outer *= shape[d];
    for (int64_t d = dim + 1; d < t.ndim(); ++d) s.inner *= shape[d];
    return s;
}

constexpr int kWarpBlockThreads = 128;
constexpr int kColumnBlockThreads = 256;

unsigned blocks_for(int64_t items, int64_t per_block) {
    return static_cast<unsigned>((items + per_block - 1) / per_block);
}

template <bool is_log>
void softmax_kernel_cuda(const Tensor& result, const Tensor& self, int64_t dim) {
    DeviceGuard guard(self.device());
    const SoftmaxShape s = softmax_shape(self, dim);
    cudaStream_t stream = cuda::current_stream();
    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        scalar_t* dst = result.mutable_data_ptr<scalar_t>();
        const scalar_t* src = self.data_ptr<scalar_t>();
        if (s.inner == 1 && s.dim_size <= kMaxWarpSoftmaxSize) {
            const unsigned blocks = blocks_for(s.outer, kWarpBlockThreads / kWarpSize);
            dispatch_warp_iterations<scalar_t, is_log>(static_cast<int>(s.dim_size), [&]<int kIterations>() {
                softmax_warp_forward<scalar_t, kIterations, is_log><<<blocks, kWarpBlockThreads, 0, stream>>>(
                    dst, src, s.outer, static_cast<int>(s.dim_size));
            });
        } else if (s.inner == 1) {
            softmax_block_forward<scalar_t, is_log><<<static_cast<unsigned>(s.outer), kBlockThreads, 0, stream>>>(
                dst, src, static_cast<int>(s.dim_size));
        } else {
            softmax_column_forward<scalar_t, is_log>
                <<<blocks_for(s.outer * s.inner, kColumnBlockThreads), kColumnBlockThreads, 0, stream>>>(
                    dst, src, s.outer, s.dim_size, s.inner);
        }
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template <bool is_log>
void softmax_backward_kernel_cuda(const Tensor& grad_input, const Tensor& grad, const Tensor& output, int64_t dim) {
    DeviceGuard guard(output.device());
    const SoftmaxShape s = softmax_shape(output, dim);
    cudaStream_t stream = cuda::current_stream();
    output.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        scalar_t* gi = grad_input.mutable_data_ptr<scalar_t>();
        const scalar_t* g = grad.data_ptr<scalar_t>();
        const scalar_t* y = output.data_ptr<scalar_t>();
        if (s.inner == 1 && s.dim_size <= kMaxWarpSoftmaxSize) {
            const unsigned blocks = blocks_for(s.outer, kWarpBlockThreads / kWarpSize);
            dispatch_warp_iterations<scalar_t, is_log>(static_cast<int>(s.dim_size), [&]<int kIterations>() {
                softmax_warp_backward<scalar_t, kIterations, is_log><<<blocks, kWarpBlockThreads, 0, stream>>>(
                    gi, g, y, s.outer, static_cast<int>(s.dim_size));
            });
        } else if (s.inner == 1) {
            softmax_block_backward<scalar_t, is_log><<<static_cast<unsigned>(s.outer), kBlockThreads, 0, stream>>>(
                gi, g, y, static_cast<int>(s.dim_size));
        } else {
            softmax_column_backward<scalar_t, is_log>
                <<<blocks_for(s.outer * s.inner, kColumnBlockThreads), kColumnBlockThreads, 0, stream>>>(
                    gi, g, y, s.outer, s.dim_size, s.inner);
        }
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // anonymous namespace

REGISTER_DISPATCH(softmax_stub, &softmax_kernel_cuda<false>);
REGISTER_DISPATCH(log_softmax_stub, &softmax_kernel_cuda<true>);
REGISTER_DISPATCH(softmax_backward_stub, &softmax_backward_kernel_cuda<false>);
REGISTER_DISPATCH(log_softmax_backward_stub, &softmax_backward_kernel_cuda<true>);

} // namespace raxis
