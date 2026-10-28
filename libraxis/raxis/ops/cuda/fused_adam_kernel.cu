#include <cstdint>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/fused_adam.h>
#include <raxis/ops/cuda/multi_tensor_apply.cuh>
#include <raxis/tensor/tensor.h>


// Fused Adam / AdamW on the GPU: one launch updates many parameters
// (multi_tensor_apply), the per-tensor value is the step count.
// From: aten/src/ATen/native/cuda/fused_adam_utils.cuh (FusedAdamMathFunctor, adam_math)

namespace raxis {
namespace {

using cuda::kMultiTensorBlockSize;
using Metadata = cuda::TensorListMetadata<4>;   // param, grad, exp_avg, exp_avg_sq

constexpr int kILP = 4;   // elements per thread and iteration: one float4

struct AdamParams {
    float lr, beta1, beta2, weight_decay, eps;
    bool maximize;
    bool decoupled_weight_decay;
};

__device__ inline void adam_math(float& param, float grad, float& exp_avg, float& exp_avg_sq, const AdamParams& o,
                                 float bias_correction1, float bias_correction2_sqrt) {
    if (o.maximize) {
        grad = -grad;
    }
    if (o.weight_decay != 0) {
        if (o.decoupled_weight_decay) {
            param -= o.lr * o.weight_decay * param;
        } else {
            grad += param * o.weight_decay;
        }
    }
    exp_avg = fmaf(o.beta1, exp_avg, fmaf(-o.beta1, grad, grad));
    exp_avg_sq = fmaf(o.beta2, exp_avg_sq, fmaf(-o.beta2, grad * grad, grad * grad));
    const float step_size = o.lr / bias_correction1;
    const float denom = ::sqrtf(exp_avg_sq) / bias_correction2_sqrt + o.eps;
    param -= step_size * exp_avg / denom;
}

__global__ void __launch_bounds__(kMultiTensorBlockSize) fused_adam_kernel(Metadata tl, AdamParams o) {
    const float step = tl.value[tl.block_to_tensor[blockIdx.x]];
    // The bias corrections in float from the step, as PyTorch's functor
    const float bias_correction1 = 1.0f - ::powf(o.beta1, step);
    const float bias_correction2_sqrt = ::sqrtf(1.0f - ::powf(o.beta2, step));
    const auto [param, n] = cuda::block_chunk<float>(tl, 0);
    const float* grad = cuda::block_chunk<float>(tl, 1).data;
    float* exp_avg = cuda::block_chunk<float>(tl, 2).data;
    float* exp_avg_sq = cuda::block_chunk<float>(tl, 3).data;

    const bool aligned = n % kILP == 0
        && (reinterpret_cast<uintptr_t>(param) | reinterpret_cast<uintptr_t>(grad)
            | reinterpret_cast<uintptr_t>(exp_avg) | reinterpret_cast<uintptr_t>(exp_avg_sq)) % 16 == 0;
    if (aligned) {
        for (int64_t i = threadIdx.x; i * kILP < n; i += blockDim.x) {
            float4 p = reinterpret_cast<float4*>(param)[i];
            const float4 g = reinterpret_cast<const float4*>(grad)[i];
            float4 m = reinterpret_cast<float4*>(exp_avg)[i];
            float4 v = reinterpret_cast<float4*>(exp_avg_sq)[i];
            adam_math(p.x, g.x, m.x, v.x, o, bias_correction1, bias_correction2_sqrt);
            adam_math(p.y, g.y, m.y, v.y, o, bias_correction1, bias_correction2_sqrt);
            adam_math(p.z, g.z, m.z, v.z, o, bias_correction1, bias_correction2_sqrt);
            adam_math(p.w, g.w, m.w, v.w, o, bias_correction1, bias_correction2_sqrt);
            reinterpret_cast<float4*>(param)[i] = p;
            reinterpret_cast<float4*>(exp_avg)[i] = m;
            reinterpret_cast<float4*>(exp_avg_sq)[i] = v;
        }
        return;
    }
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        adam_math(param[i], grad[i], exp_avg[i], exp_avg_sq[i], o, bias_correction1, bias_correction2_sqrt);
    }
}

void fused_adam_kernel_cuda(ArrayRef<Tensor> params, ArrayRef<Tensor> grads, ArrayRef<Tensor> exp_avgs,
                            ArrayRef<Tensor> exp_avg_sqs, ArrayRef<double> steps, const FusedAdamOptions& options) {
    RAXIS_CHECK_NOT_IMPLEMENTED(params[0].dtype() == dtypes::Float32,
        "fused Adam on CUDA supports float32 params only, got ", params[0].dtype());
    DeviceGuard guard(params[0].device());
    const AdamParams o{
        static_cast<float>(options.lr), static_cast<float>(options.beta1), static_cast<float>(options.beta2),
        static_cast<float>(options.weight_decay), static_cast<float>(options.eps), options.maximize,
        options.decoupled_weight_decay};
    cuda::multi_tensor_apply<4>({params, grads, exp_avgs, exp_avg_sqs}, steps, [&](const Metadata& tl, int n_blocks) {
        fused_adam_kernel<<<n_blocks, kMultiTensorBlockSize, 0, cuda::current_stream()>>>(tl, o);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // anonymous namespace

REGISTER_DISPATCH(fused_adam_stub, &fused_adam_kernel_cuda);

} // namespace raxis
