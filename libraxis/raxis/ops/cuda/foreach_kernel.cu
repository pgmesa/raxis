#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/foreach_ops.h>
#include <raxis/ops/cuda/block_reduce.cuh>
#include <raxis/ops/cuda/multi_tensor_apply.cuh>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


// _foreach_norm and _foreach_mul_ on the GPU, on top of multi_tensor_apply.
// From: aten/src/ATen/native/cuda/ForeachReduceOp.cu (LpNormFunctor,
// lpnorm_cleanup) and ForeachBinaryOpScalarTensor.cu
//
// A norm is a reduction per tensor, but multi_tensor_apply hands each block
// one chunk of 65536 elements, so a large tensor is reduced by many blocks.
// Two kernels instead of atomics or a semaphore: the first writes one partial
// result per chunk to a zeroed [n_tensors, max_chunks] buffer, the second
// runs one block per tensor that reduces its row. The order of every addition
// is fixed, so the norms are deterministic, and they follow PyTorch's order
// (4 accumulators per thread, then the block reduction), so they are bitwise
// equal to torch's _foreach_norm.

namespace raxis {
namespace {

using cuda::kChunkSize;
using cuda::kMultiTensorBlockSize;
using Metadata = cuda::TensorListMetadata<1>;

constexpr int kILP = 4;

template <ForeachNormType norm, typename T>
__device__ inline T norm_accumulate(T acc, T x) {
    if constexpr (norm == ForeachNormType::L0) {
        return acc + (x != T(0) ? T(1) : T(0));
    } else if constexpr (norm == ForeachNormType::L1) {
        return acc + ::abs(x);
    } else if constexpr (norm == ForeachNormType::L2) {
        return acc + x * x;
    } else {
        return cuda::max_propagate_nan(acc, ::abs(x));
    }
}

template <ForeachNormType norm, typename T>
__device__ inline T norm_block_reduce(T value, T* shared) {
    if constexpr (norm == ForeachNormType::LInf) {
        return cuda::block_max_nonnegative(value, shared);
    } else {
        return cuda::block_sum(value, shared);
    }
}

// kILP elements loaded as one 16-byte (float) or two (double) loads
// From: aten/src/ATen/native/cuda/MemoryAccess.cuh (aligned_vector)
template <typename T>
struct alignas(sizeof(T) * kILP) AlignedVector {
    T val[kILP];
};

template <typename T>
__device__ inline bool is_aligned(const T* p) {
    return reinterpret_cast<uintptr_t>(p) % (kILP * sizeof(T)) == 0;
}

// One partial norm per chunk: partials[tensor * max_chunks + chunk]
template <ForeachNormType norm, typename T>
__global__ void __launch_bounds__(kMultiTensorBlockSize)
norm_partial_kernel(Metadata tl, T* partials, int max_chunks) {
    __shared__ T shared[cuda::kWarpSize];
    const int t = tl.block_to_tensor[blockIdx.x];
    const int64_t chunk = tl.block_to_chunk[blockIdx.x];
    const T* x = static_cast<const T*>(tl.addresses[0][t]) + chunk * kChunkSize;
    // Elements left in the tensor from this chunk on (only the first
    // kChunkSize belong to this block), as PyTorch's functor
    const int64_t n = tl.numel[t] - chunk * kChunkSize;

    T vals[kILP] = {};
    if (n % kILP == 0 && is_aligned(x)) {
        for (int64_t i = threadIdx.x; i * kILP < n && i * kILP < kChunkSize; i += blockDim.x) {
            const AlignedVector<T> r_x = reinterpret_cast<const AlignedVector<T>*>(x)[i];
#pragma unroll
            for (int ii = 0; ii < kILP; ii++) {
                vals[ii] = norm_accumulate<norm>(vals[ii], r_x.val[ii]);
            }
        }
    } else {
        for (int64_t start = 0; start < n && start < kChunkSize; start += blockDim.x * kILP) {
#pragma unroll
            for (int ii = 0; ii < kILP; ii++) {
                const int64_t i = start + threadIdx.x + ii * blockDim.x;
                if (i < n && i < kChunkSize) {
                    vals[ii] = norm_accumulate<norm>(vals[ii], x[i]);
                }
            }
        }
    }
    T val = T(0);
#pragma unroll
    for (int ii = 0; ii < kILP; ii++) {
        val = norm == ForeachNormType::LInf ? cuda::max_propagate_nan(val, vals[ii]) : val + vals[ii];
    }
    val = norm_block_reduce<norm>(val, shared);
    if (threadIdx.x == 0) {
        partials[static_cast<int64_t>(tl.tensor_index[t]) * max_chunks + chunk] = val;
    }
}

// Block i reduces the max_chunks partial results of tensor i into out[i]
template <ForeachNormType norm, typename T>
__global__ void __launch_bounds__(kMultiTensorBlockSize)
norm_cleanup_kernel(const T* partials, T* out, int max_chunks) {
    __shared__ T shared[cuda::kWarpSize];
    const T* row = partials + static_cast<int64_t>(blockIdx.x) * max_chunks;
    T val = T(0);
    for (int i = threadIdx.x; i < max_chunks; i += blockDim.x) {
        val = norm == ForeachNormType::LInf ? cuda::max_propagate_nan(val, row[i]) : val + row[i];
    }
    val = norm_block_reduce<norm>(val, shared);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = norm == ForeachNormType::L2 ? ::sqrt(val) : val;
    }
}

template <ForeachNormType norm, typename T>
void launch_norm(ArrayRef<Tensor> tensors, const Tensor& out) {
    int64_t max_chunks = 1;
    for (const Tensor& t : tensors) {
        max_chunks = std::max(max_chunks, (t.numel() + kChunkSize - 1) / kChunkSize);
    }
    const auto n_tensors = static_cast<int64_t>(tensors.size());
    // Zeroed: chunks of the smaller tensors (and empty tensors) add nothing
    const Tensor partials = zeros({n_tensors * max_chunks}, out.options());
    T* partials_ptr = partials.mutable_data_ptr<T>();
    const cudaStream_t stream = cuda::current_stream();
    cuda::multi_tensor_apply<1>({tensors}, {}, [&](const Metadata& tl, int n_blocks) {
        norm_partial_kernel<norm, T><<<n_blocks, kMultiTensorBlockSize, 0, stream>>>(
            tl, partials_ptr, static_cast<int>(max_chunks));
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
    norm_cleanup_kernel<norm, T><<<static_cast<unsigned>(n_tensors), kMultiTensorBlockSize, 0, stream>>>(
        partials_ptr, out.mutable_data_ptr<T>(), static_cast<int>(max_chunks));
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

void foreach_norm_kernel_cuda(ArrayRef<Tensor> tensors, ForeachNormType norm, const Tensor& out) {
    DeviceGuard guard(out.device());
    out.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        switch (norm) {
            case ForeachNormType::L0: launch_norm<ForeachNormType::L0, scalar_t>(tensors, out); break;
            case ForeachNormType::L1: launch_norm<ForeachNormType::L1, scalar_t>(tensors, out); break;
            case ForeachNormType::L2: launch_norm<ForeachNormType::L2, scalar_t>(tensors, out); break;
            case ForeachNormType::LInf: launch_norm<ForeachNormType::LInf, scalar_t>(tensors, out); break;
        }
    });
}

// x *= *scalar on every chunk; the scalar is read on the device, so
// clip_grad_norm_ needs no host sync for its clip coefficient
template <typename T>
__global__ void __launch_bounds__(kMultiTensorBlockSize) mul_scalar_tensor_kernel(Metadata tl, const T* scalar) {
    const auto [x, n] = cuda::block_chunk<T>(tl, 0);
    const T s = *scalar;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        x[i] = x[i] * s;
    }
}

void foreach_mul_kernel_cuda(ArrayRef<Tensor> tensors, const Tensor& scalar) {
    DeviceGuard guard(scalar.device());
    scalar.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* s = scalar.data_ptr<scalar_t>();
        cuda::multi_tensor_apply<1>({tensors}, {}, [&](const Metadata& tl, int n_blocks) {
            mul_scalar_tensor_kernel<scalar_t><<<n_blocks, kMultiTensorBlockSize, 0, cuda::current_stream()>>>(tl, s);
            RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

} // anonymous namespace

REGISTER_DISPATCH(foreach_norm_stub, &foreach_norm_kernel_cuda);
REGISTER_DISPATCH(foreach_mul_stub, &foreach_mul_kernel_cuda);

} // namespace raxis
