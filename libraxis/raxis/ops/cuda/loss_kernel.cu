#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/block_reduce.cuh>
#include <raxis/ops/loss.h>
#include <raxis/tensor/tensor.h>


// nll_loss forward and backward on the GPU.
// From: aten/src/ATen/native/cuda/Loss.cu (nll_loss_forward_reduce_cuda_kernel_2d,
// nll_loss_forward_no_reduce_cuda_kernel, nll_loss_backward_*_cuda_kernel)
//
// The reduced loss is one block, as in PyTorch: each thread sums a strided
// share of the samples and the block adds the partial sums. The block grows
// with the batch (about 16 samples per thread, 32 to 1024 threads): each
// sample is a dependent pair of loads (target, then input[i, target]), so a
// single warp over the 16384 targets of a language model batch is latency
// bound. The result and the total weight stay on the GPU (0-d tensors), so
// neither the forward nor the backward waits for the GPU; .item() on the
// loss is the only sync.

namespace raxis {
namespace {

constexpr int kThreads = 256;

// From: aten/src/ATen/native/cuda/Loss.cu (nll_loss_threads)
int nll_loss_threads(int64_t batch_size) {
    const auto log2 = static_cast<int64_t>(std::round(std::log2(static_cast<double>(batch_size / 16))));
    return static_cast<int>(std::clamp<int64_t>(int64_t{1} << std::max<int64_t>(log2, 0), 32, 1024));
}

template <typename scalar_t>
__global__ void nll_loss_forward_reduce_kernel(
    scalar_t* output, scalar_t* total_weight, const scalar_t* input, const int64_t* target,
    const scalar_t* weight, int64_t batch_size, int64_t num_classes, bool mean, int64_t ignore_index
) {
    __shared__ scalar_t shared[cuda::kWarpSize];
    scalar_t loss_sum = 0;
    scalar_t weight_sum = 0;
    for (int64_t i = threadIdx.x; i < batch_size; i += blockDim.x) {
        const int64_t cls = target[i];
        if (cls != ignore_index) {
            RAXIS_CUDA_KERNEL_ASSERT(cls >= 0 && cls < num_classes);
            const scalar_t cls_weight = weight != nullptr ? weight[cls] : scalar_t(1);
            loss_sum -= input[i * num_classes + cls] * cls_weight;
            weight_sum += cls_weight;
        }
    }
    loss_sum = cuda::block_sum(loss_sum, shared);
    weight_sum = cuda::block_sum(weight_sum, shared);
    if (threadIdx.x == 0) {
        // Mean over zero valid targets is 0/0 = NaN, as in PyTorch
        *output = mean ? loss_sum / weight_sum : loss_sum;
        *total_weight = weight_sum;
    }
}

template <typename scalar_t>
__global__ void nll_loss_forward_no_reduce_kernel(
    scalar_t* output, const scalar_t* input, const int64_t* target, const scalar_t* weight,
    int64_t batch_size, int64_t num_classes, int64_t ignore_index
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= batch_size) {
        return;
    }
    const int64_t cls = target[i];
    if (cls == ignore_index) {
        output[i] = 0;
        return;
    }
    RAXIS_CUDA_KERNEL_ASSERT(cls >= 0 && cls < num_classes);
    output[i] = -input[i * num_classes + cls] * (weight != nullptr ? weight[cls] : scalar_t(1));
}

// grad_input is zero except at [i, target[i]]
template <typename scalar_t>
__global__ void nll_loss_backward_reduce_kernel(
    scalar_t* grad_input, const scalar_t* grad, const int64_t* target, const scalar_t* weight,
    const scalar_t* total_weight, int64_t batch_size, int64_t num_classes, bool mean, int64_t ignore_index
) {
    const scalar_t reduced_grad = mean ? -(*grad / *total_weight) : -*grad;
    for (int64_t i = threadIdx.x; i < batch_size; i += blockDim.x) {
        const int64_t cls = target[i];
        if (cls != ignore_index) {
            RAXIS_CUDA_KERNEL_ASSERT(cls >= 0 && cls < num_classes);
            grad_input[i * num_classes + cls] = weight != nullptr ? weight[cls] * reduced_grad : reduced_grad;
        }
    }
}

template <typename scalar_t>
__global__ void nll_loss_backward_no_reduce_kernel(
    scalar_t* grad_input, const scalar_t* grad, const int64_t* target, const scalar_t* weight,
    int64_t batch_size, int64_t num_classes, int64_t ignore_index
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= batch_size) {
        return;
    }
    const int64_t cls = target[i];
    if (cls == ignore_index) {
        return;
    }
    RAXIS_CUDA_KERNEL_ASSERT(cls >= 0 && cls < num_classes);
    const scalar_t cls_weight = weight != nullptr ? weight[cls] : scalar_t(1);
    grad_input[i * num_classes + cls] = -cls_weight * grad[i];
}

unsigned blocks_for(int64_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

void nll_loss_forward_kernel_cuda(
    const Tensor& output, const Tensor& total_weight, const Tensor& self, const Tensor& target,
    const Tensor& weight, Reduction reduction, int64_t ignore_index
) {
    DeviceGuard guard(self.device());
    const int64_t batch_size = self.ndim() == 2 ? self.shape()[0] : 1;
    const int64_t num_classes = self.shape()[self.ndim() - 1];
    const bool per_sample = reduction == Reduction::None && self.ndim() == 2;
    if (!per_sample && target.numel() == 0) {
        output.fill_(reduction == Reduction::Mean ? std::numeric_limits<double>::quiet_NaN() : 0.0);
        total_weight.zero_();
        return;
    }
    cudaStream_t stream = cuda::current_stream();
    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* w = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
        if (per_sample) {
            total_weight.zero_();
            if (batch_size == 0) {
                return;
            }
            nll_loss_forward_no_reduce_kernel<scalar_t><<<blocks_for(batch_size), kThreads, 0, stream>>>(
                output.mutable_data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), target.data_ptr<int64_t>(),
                w, batch_size, num_classes, ignore_index);
        } else {
            nll_loss_forward_reduce_kernel<scalar_t><<<1, nll_loss_threads(batch_size), 0, stream>>>(
                output.mutable_data_ptr<scalar_t>(), total_weight.mutable_data_ptr<scalar_t>(),
                self.data_ptr<scalar_t>(), target.data_ptr<int64_t>(), w, batch_size, num_classes,
                reduction == Reduction::Mean, ignore_index);
        }
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void nll_loss_backward_kernel_cuda(
    const Tensor& grad_input, const Tensor& grad, const Tensor& target, const Tensor& weight,
    const Tensor& total_weight, Reduction reduction, int64_t ignore_index
) {
    DeviceGuard guard(grad_input.device());
    const int64_t batch_size = grad_input.ndim() == 2 ? grad_input.shape()[0] : 1;
    const int64_t num_classes = grad_input.shape()[grad_input.ndim() - 1];
    const bool per_sample = reduction == Reduction::None && grad_input.ndim() == 2;
    if (batch_size == 0 || target.numel() == 0) {
        return;
    }
    cudaStream_t stream = cuda::current_stream();
    grad_input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* w = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
        if (per_sample) {
            nll_loss_backward_no_reduce_kernel<scalar_t><<<blocks_for(batch_size), kThreads, 0, stream>>>(
                grad_input.mutable_data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), target.data_ptr<int64_t>(),
                w, batch_size, num_classes, ignore_index);
        } else {
            nll_loss_backward_reduce_kernel<scalar_t><<<1, nll_loss_threads(batch_size), 0, stream>>>(
                grad_input.mutable_data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(), target.data_ptr<int64_t>(),
                w, total_weight.data_ptr<scalar_t>(), batch_size, num_classes,
                reduction == Reduction::Mean, ignore_index);
        }
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // anonymous namespace

REGISTER_DISPATCH(nll_loss_forward_stub, &nll_loss_forward_kernel_cuda);
REGISTER_DISPATCH(nll_loss_backward_stub, &nll_loss_backward_kernel_cuda);

} // namespace raxis
