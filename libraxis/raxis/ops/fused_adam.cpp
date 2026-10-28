#include <raxis/ops/fused_adam.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>


namespace raxis {

void fused_adam_(ArrayRef<Tensor> params, ArrayRef<Tensor> grads, ArrayRef<Tensor> exp_avgs,
                 ArrayRef<Tensor> exp_avg_sqs, ArrayRef<double> steps, const FusedAdamOptions& options) {
    const size_t n = params.size();
    RAXIS_CHECK(grads.size() == n && exp_avgs.size() == n && exp_avg_sqs.size() == n && steps.size() == n,
        "fused_adam_: expected params, grads, exp_avgs, exp_avg_sqs and steps of the same length");
    if (n == 0) {
        return;
    }
    const Tensor& first = params[0];
    RAXIS_CHECK(first.dtype().is_floating(), "fused_adam_: expected floating point params, got ", first.dtype());
    for (size_t i = 0; i < n; i++) {
        for (const Tensor* t : {&params[i], &grads[i], &exp_avgs[i], &exp_avg_sqs[i]}) {
            RAXIS_CHECK(t->defined() && t->device() == first.device() && t->dtype() == first.dtype(),
                "fused_adam_: expected all tensors on ", first.device(), " with dtype ", first.dtype());
            RAXIS_CHECK(t->is_contiguous(), "fused_adam_: expected contiguous tensors");
        }
        RAXIS_CHECK(grads[i].numel() == params[i].numel() && exp_avgs[i].numel() == params[i].numel()
                    && exp_avg_sqs[i].numel() == params[i].numel(),
            "fused_adam_: a gradient or a state has a different number of elements than its param");
    }
    fused_adam_stub(first.device().type(), params, grads, exp_avgs, exp_avg_sqs, steps, options);
    // In place without autograd (as logical_not_): copies saved of them are stale now
    for (size_t i = 0; i < n; i++) {
        params[i].unsafe_get_tensor_impl()->storage()->bump_version();
        exp_avgs[i].unsafe_get_tensor_impl()->storage()->bump_version();
        exp_avg_sqs[i].unsafe_get_tensor_impl()->storage()->bump_version();
    }
}

DEFINE_DISPATCH(fused_adam_stub);

} // namespace raxis
