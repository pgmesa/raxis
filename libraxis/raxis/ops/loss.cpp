#include <raxis/ops/loss.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>
#include <raxis/tensor/factories.h>
#include <raxis/ops/activation.h>
#include <raxis/ops/binary_ops.h>
#include <raxis/ops/device_check.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/scalar_ops.h>
#include <raxis/ops/softmax.h>
#include <raxis/ops/unary_ops.h>


namespace raxis {

Reduction parse_reduction(const std::string& reduction) {
    if (reduction == "mean") return Reduction::Mean;
    if (reduction == "sum") return Reduction::Sum;
    if (reduction == "none") return Reduction::None;
    RAXIS_CHECK(false, "'", reduction, "' is not a valid value for reduction");
    return Reduction::Mean;
}

namespace {

Tensor apply_reduction(const Tensor& loss, Reduction reduction) {
    switch (reduction) {
        case Reduction::Mean: return mean(loss);
        case Reduction::Sum: return sum(loss);
        case Reduction::None: return loss;
    }
    return loss;
}

void check_nll_loss_inputs(const Tensor& self, const Tensor& target, const Tensor& weight) {
    RAXIS_CHECK(self.ndim() == 1 || self.ndim() == 2,
        "nll_loss: input must be 1-D or 2-D, got ", self.ndim(), "-D");
    RAXIS_CHECK(target.ndim() == self.ndim() - 1,
        "nll_loss: expected target of ", self.ndim() - 1, " dims, got ", target.ndim());
    RAXIS_CHECK(target.dtype() == dtypes::Int64,
        "nll_loss: expected target of dtype int64, got ", target.dtype());
    RAXIS_CHECK(self.dtype().is_floating(),
        "nll_loss: expected a floating point input, got ", self.dtype());
    if (self.ndim() == 2) {
        RAXIS_CHECK(self.shape()[0] == target.shape()[0],
            "nll_loss: expected input batch size (", self.shape()[0],
            ") to match target batch size (", target.shape()[0], ")");
    }
    const int64_t num_classes = self.shape()[self.ndim() - 1];
    if (weight.defined()) {
        RAXIS_CHECK(weight.ndim() == 1 && weight.numel() == num_classes,
            "weight tensor should be defined either for all ", num_classes,
            " classes or no classes but got weight tensor of shape: ", weight.shape());
        RAXIS_CHECK(weight.dtype() == self.dtype(),
            "nll_loss: expected weight of dtype ", self.dtype(), ", got ", weight.dtype());
    }
}

void check_target(int64_t cls, int64_t num_classes) {
    RAXIS_CHECK(cls >= 0 && cls < num_classes,
        "nll_loss: target ", cls, " is out of bounds for ", num_classes, " classes");
}

} // namespace

namespace ops {

// From: aten/src/ATen/native/LossNLL.cpp (nll_loss_out_frame)
//
// The reduced loss is summed in scalar_t with a cascade: element b goes into
// partial sum 0, and every 2^p elements partial sum j is flushed into j + 1
// (j = 0..7). Each partial sum adds up values of similar magnitude, so the
// rounding error grows with log(n) instead of n, without a double
// accumulator. p >= 4 grows with the batch so that 8 levels always suffice.
NllLossResult nll_loss_forward(
    const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index
) {
    check_cpu_only("nll_loss", self, target, weight);
    check_nll_loss_inputs(self, target, weight);
    const int64_t batch_size = self.ndim() == 2 ? self.shape()[0] : 1;
    const int64_t num_classes = self.shape()[self.ndim() - 1];

    const Tensor input = self.contiguous();
    const Tensor target_c = target.contiguous();
    const Tensor weight_c = weight.defined() ? weight.contiguous() : weight;
    const int64_t* t = target_c.data_ptr<int64_t>();

    const bool per_sample = reduction == Reduction::None && self.ndim() == 2;
    Tensor output = per_sample
        ? empty({batch_size}, self.options())
        : empty({}, self.options());

    double total_weight = 0.0;
    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* x = input.data_ptr<scalar_t>();
        const scalar_t* w = weight_c.defined() ? weight_c.data_ptr<scalar_t>() : nullptr;
        scalar_t* out = output.mutable_data_ptr<scalar_t>();

        if (per_sample) {
            parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
                for (int64_t i = begin; i < end; ++i) {
                    const int64_t cls = t[i];
                    if (cls == ignore_index) {
                        out[i] = 0;
                        continue;
                    }
                    check_target(cls, num_classes);
                    out[i] = -x[i * num_classes + cls] * (w ? w[cls] : static_cast<scalar_t>(1));
                }
            });
            return;
        }

        if (target.numel() == 0) {
            out[0] = reduction == Reduction::Mean ? std::numeric_limits<scalar_t>::quiet_NaN() : 0;
            return;
        }

        constexpr int64_t levels = 8;
        const int64_t ceil_log2 = batch_size <= 2 ? 1 : std::bit_width(static_cast<uint64_t>(batch_size - 1));
        const int64_t level_power = std::max<int64_t>(4, ceil_log2 / levels);
        const int64_t level_mask = (int64_t{1} << level_power) - 1;

        scalar_t weight_sums[levels] = {};
        scalar_t loss_sums[levels] = {};
        int64_t num_ignored = 0;
        for (int64_t b = 0; b < batch_size; ++b) {
            const int64_t cls = t[b];
            if (cls == ignore_index) {
                ++num_ignored;
                continue;
            }
            check_target(cls, num_classes);
            const scalar_t value = x[b * num_classes + cls];
            if (w) {
                loss_sums[0] -= value * w[cls];
                weight_sums[0] += w[cls];
            } else {
                loss_sums[0] -= value;
            }
            for (int64_t j = 0; j + 1 < levels; ++j) {
                if ((b & (level_mask << (j * level_power))) != 0) {
                    break;
                }
                weight_sums[j + 1] += weight_sums[j];
                loss_sums[j + 1] += loss_sums[j];
                weight_sums[j] = 0;
                loss_sums[j] = 0;
            }
        }

        const scalar_t weight_total = w
            ? std::accumulate(std::begin(weight_sums), std::end(weight_sums), scalar_t{0})
            : static_cast<scalar_t>(batch_size - num_ignored);
        scalar_t loss = std::accumulate(std::begin(loss_sums), std::end(loss_sums), scalar_t{0});
        // Mean over zero valid targets is 0/0 = NaN, as in PyTorch
        if (reduction == Reduction::Mean) {
            loss /= weight_total;
        }
        out[0] = loss;
        total_weight = static_cast<double>(weight_total);
    });
    return {output, total_weight};
}

// d loss / d self is zero everywhere except at [i, target[i]], where it is
// -weight[target[i]] * grad_i (grad / total_weight for the mean)
// From: aten/src/ATen/native/LossNLL.cpp (nll_loss_backward_out_frame)
Tensor nll_loss_backward(
    const Tensor& grad, const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index, double total_weight
) {
    check_cpu_only("nll_loss_backward", grad, self, target, weight);
    const int64_t batch_size = self.ndim() == 2 ? self.shape()[0] : 1;
    const int64_t num_classes = self.shape()[self.ndim() - 1];
    const bool per_sample = reduction == Reduction::None && self.ndim() == 2;

    Tensor grad_input = zeros(self.shape(), self.options());
    const Tensor target_c = target.contiguous();
    const Tensor grad_c = grad.contiguous();
    const Tensor weight_c = weight.defined() ? weight.contiguous() : weight;
    const int64_t* t = target_c.data_ptr<int64_t>();

    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        scalar_t* gi = grad_input.mutable_data_ptr<scalar_t>();
        const scalar_t* g = grad_c.data_ptr<scalar_t>();
        const scalar_t* w = weight_c.defined() ? weight_c.data_ptr<scalar_t>() : nullptr;
        const scalar_t reduced_grad = reduction == Reduction::Mean
            ? -(g[0] / static_cast<scalar_t>(total_weight))
            : -g[0];

        parallel_for(0, batch_size, 0, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                const int64_t cls = t[i];
                if (cls == ignore_index) continue;
                const scalar_t cls_weight = w ? w[cls] : static_cast<scalar_t>(1);
                gi[i * num_classes + cls] = per_sample ? -cls_weight * g[i] : cls_weight * reduced_grad;
            }
        });
    });
    return grad_input;
}

} // namespace ops

// -- Composite losses --

Tensor mse_loss(const Tensor& self, const Tensor& target, Reduction reduction) {
    Tensor diff = self - target;
    return apply_reduction(diff * diff, reduction);
}

Tensor cross_entropy(
    const Tensor& self, const Tensor& target, const Tensor& weight, Reduction reduction, int64_t ignore_index
) {
    int64_t class_dim = self.ndim() == 1 ? 0 : 1;
    return nll_loss(log_softmax(self, class_dim), target, weight, reduction, ignore_index);
}

// -(t * log(x) + (1 - t) * log(1 - x)), with the logs clamped to >= -100 so
// that x = 0 or x = 1 give a finite loss (as PyTorch does)
Tensor binary_cross_entropy(const Tensor& self, const Tensor& target, Reduction reduction) {
    Tensor min_log = wrapped_scalar_tensor(-100.0);
    Tensor log_x = maximum(log(self), min_log);
    Tensor log_1mx = maximum(log(self.rsub(1.0)), min_log);
    Tensor loss = -(target * log_x + target.rsub(1.0) * log_1mx);
    return apply_reduction(loss, reduction);
}

// Stable form of bce(sigmoid(x), t): max(x, 0) - x * t + log(1 + exp(-|x|)).
// exp never overflows because its argument is always <= 0.
Tensor binary_cross_entropy_with_logits(const Tensor& self, const Tensor& target, Reduction reduction) {
    Tensor loss = relu(self) - self * target + log1p(exp(-abs(self)));
    return apply_reduction(loss, reduction);
}

} // namespace raxis
