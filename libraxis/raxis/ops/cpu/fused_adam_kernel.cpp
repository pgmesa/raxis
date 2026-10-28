#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/fused_adam.h>
#include <raxis/ops/cpu/vec/vec.h>

#include <raxis/tensor/tensor.h>
#include <raxis/util/parallel.h>


namespace raxis {

inline namespace CPU_CAPABILITY {

// From: aten/src/ATen/native/cpu/FusedAdamKernel.cpp (adam_math)
//
// The same update as the GPU kernel, with PyTorch's CPU expressions (so the
// results are bitwise torch's): the bias corrections in double, as the
// unfused optimizer, and exp_avg.lerp_(grad, 1 - beta1) written as
// lerp: base + w * (end - base) when |w| < 0.5, end - (end - base) * (1 - w)
// otherwise, the more accurate form for each weight. The vector loop and
// the scalar tail round differently (fmadd), as in PyTorch.
template <typename scalar_t>
void adam_math(scalar_t* param_ptr, scalar_t* exp_avg_ptr, scalar_t* exp_avg_sq_ptr, const scalar_t* grad_ptr,
               double lr, double bias_correction1, double exp_avg_grad_coefficient,
               double exp_avg_sq_grad_coefficient, double bias_correction2_sqrt, double eps, double weight_decay,
               double beta2, bool maximize, bool decoupled, int64_t size) {
    using Vec = vec::Vectorized<scalar_t>;
    const double step_size = lr / bias_correction1;
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec param_vec = Vec::loadu(param_ptr + d);
        Vec grad_vec = Vec::loadu(grad_ptr + d);
        if (maximize) grad_vec = grad_vec * Vec(scalar_t(-1.0));
        if (weight_decay != 0.0) {
            if (decoupled) {
                param_vec = param_vec * Vec(scalar_t(1 - lr * weight_decay));
            } else {
                grad_vec += param_vec * Vec(scalar_t(weight_decay));
            }
        }
        Vec exp_avg_vec = Vec::loadu(exp_avg_ptr + d);
        const Vec lerp_weight = Vec(scalar_t(exp_avg_grad_coefficient));
        const Vec mask = lerp_weight.abs() < Vec(0.5);
        const Vec coeff = Vec::blendv(lerp_weight - Vec(1), lerp_weight, mask);
        const Vec base = Vec::blendv(grad_vec, exp_avg_vec, mask);
        exp_avg_vec = vec::fmadd(coeff, grad_vec - exp_avg_vec, base);

        const Vec exp_avg_sq_vec = Vec::loadu(exp_avg_sq_ptr + d) * Vec(scalar_t(beta2))
            + Vec(scalar_t(exp_avg_sq_grad_coefficient)) * grad_vec * grad_vec;
        exp_avg_vec.store(exp_avg_ptr + d);
        exp_avg_sq_vec.store(exp_avg_sq_ptr + d);

        const Vec denom_vec = (exp_avg_sq_vec.sqrt() / Vec(scalar_t(bias_correction2_sqrt))) + Vec(scalar_t(eps));
        param_vec = param_vec + Vec(scalar_t(-step_size)) * exp_avg_vec / denom_vec;
        param_vec.store(param_ptr + d);
    }
    for (; d < size; d++) {
        scalar_t grad_val = grad_ptr[d];
        if (maximize) grad_val = -grad_val;
        if (weight_decay != 0.0) {
            if (decoupled) {
                param_ptr[d] = param_ptr[d] * scalar_t(1 - lr * weight_decay);
            } else {
                grad_val += param_ptr[d] * scalar_t(weight_decay);
            }
        }
        if (std::abs(scalar_t(exp_avg_grad_coefficient)) < scalar_t(0.5)) {
            exp_avg_ptr[d] = exp_avg_ptr[d] + scalar_t(exp_avg_grad_coefficient) * (grad_val - exp_avg_ptr[d]);
        } else {
            exp_avg_ptr[d] = grad_val - (grad_val - exp_avg_ptr[d]) * (scalar_t(1) - scalar_t(exp_avg_grad_coefficient));
        }
        exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] * scalar_t(beta2);
        exp_avg_sq_ptr[d] = exp_avg_sq_ptr[d] + scalar_t(exp_avg_sq_grad_coefficient) * grad_val * grad_val;
        const scalar_t denom = std::sqrt(exp_avg_sq_ptr[d]) / scalar_t(bias_correction2_sqrt) + scalar_t(eps);
        param_ptr[d] = param_ptr[d] - scalar_t(step_size) * exp_avg_ptr[d] / denom;
    }
}

// Threads split each parameter in whole cache lines (adam_fused_step_impl),
// so the vector loop never stops early inside a chunk: the result does not
// depend on the thread count
void fused_adam_kernel(ArrayRef<Tensor> params, ArrayRef<Tensor> grads, ArrayRef<Tensor> exp_avgs,
                       ArrayRef<Tensor> exp_avg_sqs, ArrayRef<double> steps, const FusedAdamOptions& o) {
    params[0].dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        constexpr int64_t kLine = 64 / sizeof(scalar_t);
        for (size_t i = 0; i < params.size(); i++) {
            const double bias_correction1 = 1 - std::pow(o.beta1, steps[i]);
            const double bias_correction2_sqrt = std::sqrt(1 - std::pow(o.beta2, steps[i]));
            scalar_t* param = params[i].mutable_data_ptr<scalar_t>();
            const scalar_t* grad = grads[i].data_ptr<scalar_t>();
            scalar_t* exp_avg = exp_avgs[i].mutable_data_ptr<scalar_t>();
            scalar_t* exp_avg_sq = exp_avg_sqs[i].mutable_data_ptr<scalar_t>();
            const int64_t numel = params[i].numel();
            parallel_for(0, (numel + kLine - 1) / kLine, 0, [&](int64_t begin, int64_t end) {
                begin *= kLine;
                end = std::min(end * kLine, numel);
                adam_math(param + begin, exp_avg + begin, exp_avg_sq + begin, grad + begin, o.lr,
                          bias_correction1, 1 - o.beta1, 1 - o.beta2, bias_correction2_sqrt, o.eps, o.weight_decay,
                          o.beta2, o.maximize, o.decoupled_weight_decay, end - begin);
            });
        }
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(fused_adam_stub, &fused_adam_kernel);

} // namespace raxis
