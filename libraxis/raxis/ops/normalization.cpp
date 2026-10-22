#include <raxis/ops/normalization.h>

#include <cmath>
#include <cstdint>

#include <raxis/util/exception.h>

#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

namespace {

void check_channel_param(const Tensor& t, const char* name, const Tensor& input) {
    if (!t.defined()) return;
    const int64_t channels = input.shape()[1];
    RAXIS_CHECK(t.numel() == channels, name, " should contain ", channels, " elements not ", t.numel());
    RAXIS_CHECK(t.dtype() == input.dtype(),
        "batch_norm: expected ", name, " to have dtype ", input.dtype(), ", but got ", t.dtype());
}

void check_batch_norm_args(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& running_mean, const Tensor& running_var
) {
    RAXIS_CHECK(input.ndim() >= 2,
        "batch_norm: expected an input of shape (N, C, *), but got ", input.shape());
    RAXIS_CHECK(input.dtype().is_floating(),
        "batch_norm: expected a floating point input, but got ", input.dtype());
    RAXIS_CHECK(running_mean.defined() == running_var.defined(),
        "running_mean and running_var must either both be None or neither be None");
    check_channel_param(weight, "weight", input);
    check_channel_param(bias, "bias", input);
    check_channel_param(running_mean, "running_mean", input);
    check_channel_param(running_var, "running_var", input);
    // Updated in place through their data pointers
    RAXIS_CHECK(!running_mean.defined() || (running_mean.is_contiguous() && running_var.is_contiguous()),
        "batch_norm: running_mean and running_var must be contiguous");
}

Tensor contiguous_or_undefined(const Tensor& t) {
    return t.defined() ? t.contiguous() : t;
}

} // namespace


namespace ops {

std::tuple<Tensor, Tensor, Tensor> native_batch_norm(
    const Tensor& input_, const Tensor& weight_, const Tensor& bias_,
    const Tensor& running_mean, const Tensor& running_var,
    bool training, double momentum, double eps
) {
    check_batch_norm_args(input_, weight_, bias_, running_mean, running_var);
    const Tensor input = input_.contiguous();
    const Tensor weight = contiguous_or_undefined(weight_);
    const Tensor bias = contiguous_or_undefined(bias_);
    Tensor output = empty(input.shape(), input.options());

    if (!training) {
        RAXIS_CHECK(running_mean.defined(),
            "batch_norm: running_mean and running_var are required when not training");
        if (input.numel() > 0) {
            batch_norm_stub(DeviceType::CPU, output, input, weight, bias, Tensor(), Tensor(),
                            running_mean, running_var, false, eps);
        }
        return {output, empty({0}, input.options()), empty({0}, input.options())};
    }

    // From: aten/src/ATen/native/Normalization.cpp (batch_norm_cpu_update_stats_template)
    RAXIS_CHECK(input.numel() != 0,
        "input tensor must have at least one element, but got input_sizes = ", input.shape());
    const int64_t channels = input.shape()[1];
    const int64_t n = input.numel() / channels;
    Tensor save_mean = empty({channels}, input.options());
    Tensor save_invstd = empty({channels}, input.options());
    Tensor var_sum = empty({channels}, input.options());
    batch_norm_collect_stats_stub(DeviceType::CPU, save_mean, var_sum, input);

    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* mean = save_mean.data_ptr<scalar_t>();
        const scalar_t* var_sums = var_sum.data_ptr<scalar_t>();
        scalar_t* invstd = save_invstd.mutable_data_ptr<scalar_t>();
        scalar_t* run_mean = running_mean.defined() ? running_mean.mutable_data_ptr<scalar_t>() : nullptr;
        scalar_t* run_var = running_var.defined() ? running_var.mutable_data_ptr<scalar_t>() : nullptr;
        const auto m = static_cast<scalar_t>(momentum);
        for (int64_t c = 0; c < channels; ++c) {
            // The batch is normalized with the biased variance, but the
            // running estimate uses the unbiased one (divided by n - 1)
            const double var = var_sums[c] / n;
            invstd[c] = (var != 0 || eps != 0) ? static_cast<scalar_t>(1 / std::sqrt(var + eps)) : scalar_t(0);
            if (run_mean) {
                run_mean[c] = m * mean[c] + (1 - m) * run_mean[c];
                const double unbiased_var = var_sums[c] / (n - 1);
                run_var[c] = m * unbiased_var + (1 - m) * run_var[c];
            }
        }
    });

    batch_norm_stub(DeviceType::CPU, output, input, weight, bias, save_mean, save_invstd,
                    running_mean, running_var, true, eps);
    return {output, save_mean, save_invstd};
}

std::tuple<Tensor, Tensor, Tensor> native_batch_norm_backward(
    const Tensor& grad_output_, const Tensor& input_, const Tensor& weight_,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd,
    bool training, double eps, std::array<bool, 3> output_mask
) {
    const Tensor input = input_.contiguous();
    const Tensor grad_output = grad_output_.contiguous();
    const Tensor weight = contiguous_or_undefined(weight_);
    const int64_t channels = input.shape()[1];

    Tensor grad_input, grad_weight, grad_bias;
    if (output_mask[0]) {
        grad_input = empty(input.shape(), input.options());
    }
    if (output_mask[1]) {
        grad_weight = empty({channels}, input.options());
    }
    if (output_mask[2]) {
        grad_bias = empty({channels}, input.options());
    }
    batch_norm_backward_stub(
        DeviceType::CPU, grad_input, grad_weight, grad_bias, grad_output, input, weight,
        running_mean, running_var, save_mean, save_invstd, training, eps);
    return {grad_input, grad_weight, grad_bias};
}

} // namespace ops

DEFINE_DISPATCH(batch_norm_stub);
DEFINE_DISPATCH(batch_norm_collect_stats_stub);
DEFINE_DISPATCH(batch_norm_backward_stub);

} // namespace raxis
