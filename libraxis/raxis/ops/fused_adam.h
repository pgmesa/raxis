#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/array_ref.h>


namespace raxis {

struct FusedAdamOptions {
    double lr;
    double beta1;
    double beta2;
    double weight_decay;
    double eps;
    bool maximize;
    // AdamW: p -= lr * weight_decay * p, instead of adding p * weight_decay to the gradient
    bool decoupled_weight_decay;
};

// One Adam step on every parameter: params[i], exp_avgs[i] and exp_avg_sqs[i]
// are updated in place from grads[i]; steps[i] is the parameter's step count
// after this step (for the bias corrections). All the tensors are
// contiguous, of one dtype and on one device.
using fused_adam_fn = void(*)(ArrayRef<Tensor> params, ArrayRef<Tensor> grads, ArrayRef<Tensor> exp_avgs,
                              ArrayRef<Tensor> exp_avg_sqs, ArrayRef<double> steps,
                              const FusedAdamOptions& options);

DECLARE_DISPATCH(fused_adam_fn, fused_adam_stub);

/// torch.optim.Adam / AdamW with fused=True: the whole update of every
/// parameter in one kernel per chunk of parameters, instead of a chain of
/// elementwise ops per parameter (each one a pass over memory).
/// No autograd: optimizers run under no_grad.
/// From: aten/src/ATen/native/FusedAdam.cpp (_fused_adam_kernel_cpu_ / _fused_adamw_kernel_cpu_)
/// and native/cuda/fused_adam_impl.cu, fused_adamw_impl.cu
RAXIS_API void fused_adam_(ArrayRef<Tensor> params, ArrayRef<Tensor> grads, ArrayRef<Tensor> exp_avgs,
                           ArrayRef<Tensor> exp_avg_sqs, ArrayRef<double> steps, const FusedAdamOptions& options);

} // namespace raxis
