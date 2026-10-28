#include <cstdint>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/fused_adam.h>
#include <raxis/tensor/tensor.h>


// Fused Adam / AdamW on the GPU.
// From: aten/src/ATen/native/cuda/MultiTensorApply.cuh
// (multi_tensor_apply_for_fused_optimizer) and fused_adam_utils.cuh
// (FusedAdamMathFunctor, adam_math).
//
// An optimizer step touches every parameter: tens to hundreds of tensors,
// many of them small. One launch per tensor (or worse, one per elementwise
// op per tensor, as the unfused loop) pays a launch for little work each
// time. multi_tensor_apply packs many tensors into one launch: the kernel
// arguments carry a table of their pointers, and each block looks up which
// tensor and which chunk of 65536 elements it owns (block_to_tensor,
// block_to_chunk). When the table or the block list is full, the kernel is
// launched and the table starts over; a tensor cut by a launch continues in
// slot 0 of the next one. The table must fit in the 4 KB of kernel
// arguments: 36 tensors and 320 blocks for the 4 lists of Adam.

namespace raxis {
namespace {

constexpr int kDepth = 4;            // param, grad, exp_avg, exp_avg_sq
constexpr int kMaxTensors = 36;
constexpr int kMaxBlocks = 320;
constexpr int64_t kChunkSize = 65536;
constexpr int kBlockSize = 512;
constexpr int kILP = 4;              // elements per thread and iteration: one float4

struct TensorListMetadata {
    void* addresses[kDepth][kMaxTensors];
    int64_t numel[kMaxTensors];
    float step[kMaxTensors];
    uint8_t block_to_tensor[kMaxBlocks];
    int32_t block_to_chunk[kMaxBlocks];
};

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

__global__ void __launch_bounds__(kBlockSize) fused_adam_kernel(TensorListMetadata tl, AdamParams o) {
    const int t = tl.block_to_tensor[blockIdx.x];
    const int64_t begin = static_cast<int64_t>(tl.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t n = ::min(tl.numel[t] - begin, kChunkSize);
    // The bias corrections in float from the step, as PyTorch's functor
    const float bias_correction1 = 1.0f - ::powf(o.beta1, tl.step[t]);
    const float bias_correction2_sqrt = ::sqrtf(1.0f - ::powf(o.beta2, tl.step[t]));
    float* param = static_cast<float*>(tl.addresses[0][t]) + begin;
    const float* grad = static_cast<const float*>(tl.addresses[1][t]) + begin;
    float* exp_avg = static_cast<float*>(tl.addresses[2][t]) + begin;
    float* exp_avg_sq = static_cast<float*>(tl.addresses[3][t]) + begin;

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
    const ArrayRef<Tensor> lists[kDepth] = {params, grads, exp_avgs, exp_avg_sqs};

    TensorListMetadata tl;
    int n_tensors = 0;
    int n_blocks = 0;
    auto launch = [&] {
        fused_adam_kernel<<<n_blocks, kBlockSize, 0, cuda::current_stream()>>>(tl, o);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    };
    for (size_t i = 0; i < params.size(); i++) {
        const int64_t numel = params[i].numel();
        if (numel == 0) {
            continue;
        }
        for (int d = 0; d < kDepth; d++) {
            tl.addresses[d][n_tensors] = const_cast<void*>(lists[d][i].data_ptr());
        }
        tl.numel[n_tensors] = numel;
        tl.step[n_tensors] = static_cast<float>(steps[i]);
        n_tensors++;
        const int64_t chunks = (numel + kChunkSize - 1) / kChunkSize;
        for (int64_t chunk = 0; chunk < chunks; chunk++) {
            tl.block_to_tensor[n_blocks] = static_cast<uint8_t>(n_tensors - 1);
            tl.block_to_chunk[n_blocks] = static_cast<int32_t>(chunk);
            n_blocks++;
            const bool tensors_full = n_tensors == kMaxTensors && chunk == chunks - 1;
            if (!tensors_full && n_blocks < kMaxBlocks) {
                continue;
            }
            launch();
            n_blocks = 0;
            if (chunk == chunks - 1) {
                n_tensors = 0;
            } else {
                // The rest of this tensor's chunks go in the next launch, from slot 0
                for (int d = 0; d < kDepth; d++) {
                    tl.addresses[d][0] = tl.addresses[d][n_tensors - 1];
                }
                tl.numel[0] = tl.numel[n_tensors - 1];
                tl.step[0] = tl.step[n_tensors - 1];
                n_tensors = 1;
            }
        }
    }
    if (n_blocks > 0) {
        launch();
    }
}

} // anonymous namespace

REGISTER_DISPATCH(fused_adam_stub, &fused_adam_kernel_cuda);

} // namespace raxis
