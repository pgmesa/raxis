#include <cstdint>
#include <type_traits>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/atomic.cuh>
#include <raxis/ops/padding.h>
#include <raxis/ops/padding_index.h>
#include <raxis/tensor/tensor.h>


// Reflect, replicate and circular padding on the GPU.
// From: aten/src/ATen/native/cuda/ReflectionPad.cu, ReplicationPadding.cu
//
// One thread per output element of the [planes, D, H, W] view (see
// ops/padding_index.h): it finds the input element it reads with the same
// index() as the CPU kernel. The backward runs the same map the other way
// and adds each output gradient into the input element it was read from;
// several outputs share an input (the mirrored, replicated or wrapped
// borders), so the adds are atomic, as PyTorch's, and float gradients at
// those borders can change in the last bits between runs.

namespace raxis {
namespace {

constexpr int kThreads = 256;

// The input offset (within its plane) that output element `o` of a plane reads
template <typename PaddingType>
__device__ inline int64_t input_offset(int64_t o, const PaddingShape& p) {
    const int64_t ow = o % p.oshape[2];
    const int64_t oh = (o / p.oshape[2]) % p.oshape[1];
    const int64_t od = o / (p.oshape[2] * p.oshape[1]);
    const int64_t id = PaddingType::index(od, p.sizes[0], p.pads[0], p.offsets[0]);
    const int64_t ih = PaddingType::index(oh, p.sizes[1], p.pads[1], p.offsets[1]);
    const int64_t iw = PaddingType::index(ow, p.sizes[2], p.pads[2], p.offsets[2]);
    return (id * p.ishape[1] + ih) * p.ishape[2] + iw;
}

template <typename scalar_t, typename PaddingType>
__global__ void padding_kernel(scalar_t* output, const scalar_t* input, PaddingShape p, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t output_plane = p.oshape[0] * p.oshape[1] * p.oshape[2];
    const int64_t plane = i / output_plane;
    const int64_t input_plane = p.ishape[0] * p.ishape[1] * p.ishape[2];
    output[i] = input[plane * input_plane + input_offset<PaddingType>(i - plane * output_plane, p)];
}

template <typename scalar_t, typename PaddingType>
__global__ void padding_backward_kernel(scalar_t* grad_input, const scalar_t* grad_output, PaddingShape p, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t output_plane = p.oshape[0] * p.oshape[1] * p.oshape[2];
    const int64_t plane = i / output_plane;
    const int64_t input_plane = p.ishape[0] * p.ishape[1] * p.ishape[2];
    cuda::atomic_add(grad_input + plane * input_plane + input_offset<PaddingType>(i - plane * output_plane, p),
                     grad_output[i]);
}

unsigned blocks_for(int64_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

void padding_kernel_cuda(const Tensor& output, const Tensor& input, IntArrayRef padding, PaddingMode mode) {
    const int64_t n = output.numel();
    if (n == 0) {
        return;
    }
    DeviceGuard guard(input.device());
    const PaddingShape p = padding_shape(input, output, padding, mode);
    input.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        dispatch_padding_type(mode, [&]<typename PaddingType>(PaddingType) {
            padding_kernel<scalar_t, PaddingType><<<blocks_for(n), kThreads, 0, cuda::current_stream()>>>(
                output.mutable_data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), p, n);
        });
    });
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

void padding_backward_kernel_cuda(const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding,
                                  PaddingMode mode) {
    const int64_t n = grad_output.numel();
    if (n == 0) {
        return;
    }
    DeviceGuard guard(grad_output.device());
    const PaddingShape p = padding_shape(grad_input, grad_output, padding, mode);
    grad_output.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        dispatch_padding_type(mode, [&]<typename PaddingType>(PaddingType) {
            padding_backward_kernel<scalar_t, PaddingType><<<blocks_for(n), kThreads, 0, cuda::current_stream()>>>(
                grad_input.mutable_data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), p, n);
        });
    });
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

} // anonymous namespace

REGISTER_DISPATCH(padding_stub, &padding_kernel_cuda);
REGISTER_DISPATCH(padding_backward_stub, &padding_backward_kernel_cuda);

} // namespace raxis
