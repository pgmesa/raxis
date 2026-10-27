#include <algorithm>
#include <cmath>
#include <cstdint>

#include <raxis/core/device_guard.h>
#include <raxis/core/philox.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/dropout.h>
#include <raxis/tensor/tensor.h>


// Dropout on the GPU with the same random numbers as the CPU.
// From: aten/src/ATen/native/cuda/Dropout.cu (fused_dropout_kernel) for the
// scheme, ops/cpu/dropout_kernel.cpp for the counter layout.
//
// PyTorch's CUDA dropout gives each thread its own Philox subsequence
// (curand_init(seed, thread, offset)) and walks the tensor with a grid
// stride, so which number an element gets depends on the launch
// configuration. raxis fixes it instead: element i takes word (i % 64) / 16
// of block offset + (i / 64) * 16 + i % 16, the order in which the CPU
// kernel draws its batches of 16 blocks. Thread t computes block offset + t
// and writes the 4 elements that use it, 16 apart, so neighbouring threads
// still write neighbouring elements (coalesced) and the mask is bitwise the
// one the CPU draws for the same seed.

namespace raxis {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void fused_dropout_kernel(
    scalar_t* out, bool* mask, const scalar_t* in, int64_t n,
    uint64_t threshold, scalar_t scale, uint64_t seed, uint64_t offset
) {
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t first = (t / kPhiloxLanes) * kPhiloxBatchWords + t % kPhiloxLanes;
    if (first >= n) {
        return;
    }
    const PhiloxBlock r = philox4x32_10(offset + static_cast<uint64_t>(t), seed);
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        const int64_t i = first + w * kPhiloxLanes;
        if (i < n) {
            // Kept when the 32-bit draw is below p_keep * 2^32: an exact
            // integer comparison, as on the CPU
            const bool keep = r[w] < threshold;
            mask[i] = keep;
            out[i] = in[i] * static_cast<scalar_t>(keep) * scale;
        }
    }
}

template <typename scalar_t>
__global__ void dropout_backward_kernel(
    scalar_t* grad_in, const scalar_t* grad, const bool* mask, int64_t n, scalar_t scale
) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        grad_in[i] = grad[i] * static_cast<scalar_t>(mask[i]) * scale;
    }
}

void fused_dropout_kernel_cuda(
    const Tensor& output, const Tensor& mask, const Tensor& input,
    double p_keep, uint64_t seed, uint64_t offset
) {
    DeviceGuard guard(input.device());
    const int64_t n = input.numel();
    const auto threshold = static_cast<uint64_t>(std::ldexp(p_keep, 32));
    // One thread per Philox block, the blocks the generator reserved
    const auto threads = static_cast<int64_t>(philox_blocks(n, 1));
    const auto blocks = static_cast<unsigned>((threads + kThreads - 1) / kThreads);
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const auto scale = static_cast<scalar_t>(p_keep == 0 ? 0.0 : 1.0 / p_keep);
        fused_dropout_kernel<scalar_t><<<blocks, kThreads, 0, cuda::current_stream()>>>(
            output.mutable_data_ptr<scalar_t>(), mask.mutable_data_ptr<bool>(),
            input.data_ptr<scalar_t>(), n, threshold, scale, seed, offset);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void dropout_backward_kernel_cuda(const Tensor& grad_input, const Tensor& grad, const Tensor& mask, double scale) {
    DeviceGuard guard(grad.device());
    const int64_t n = grad.numel();
    const auto blocks = static_cast<unsigned>(std::min<int64_t>((n + kThreads - 1) / kThreads, 65535));
    grad.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        dropout_backward_kernel<scalar_t><<<blocks, kThreads, 0, cuda::current_stream()>>>(
            grad_input.mutable_data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(), n, static_cast<scalar_t>(scale));
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // anonymous namespace

REGISTER_DISPATCH(fused_dropout_stub, &fused_dropout_kernel_cuda);
REGISTER_DISPATCH(dropout_backward_stub, &dropout_backward_kernel_cuda);

} // namespace raxis
