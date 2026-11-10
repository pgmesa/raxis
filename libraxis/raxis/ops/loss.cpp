#include <raxis/ops/loss.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>
#include <raxis/tensor/factories.h>
#include <raxis/ops/activation.h>
#include <raxis/ops/binary_ops.h>
#include <raxis/ops/device_check.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/scalar_ops.h>
#include <raxis/ops/softmax.h>
#include <raxis/ops/tensor_advanced_indexing.h>
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

// From: aten/src/ATen/native/LossNLL.cpp (nll_loss_out_frame)
//
// The reduced loss is summed in scalar_t with a cascade: element b goes into
// partial sum 0, and every 2^p elements partial sum j is flushed into j + 1
// (j = 0..7). Each partial sum adds up values of similar magnitude, so the
// rounding error grows with log(n) instead of n, without a double
// accumulator. p >= 4 grows with the batch so that 8 levels always suffice.
void nll_loss_forward_cpu(
    const Tensor& output, const Tensor& total_weight, const Tensor& self, const Tensor& target,
    const Tensor& weight, Reduction reduction, int64_t ignore_index
) {
    const int64_t batch_size = self.ndim() == 2 ? self.shape()[0] : 1;
    const int64_t num_classes = self.shape()[self.ndim() - 1];
    const int64_t* t = target.data_ptr<int64_t>();
    const bool per_sample = reduction == Reduction::None && self.ndim() == 2;

    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* x = self.data_ptr<scalar_t>();
        const scalar_t* w = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
        scalar_t* out = output.mutable_data_ptr<scalar_t>();
        scalar_t* tw = total_weight.mutable_data_ptr<scalar_t>();
        *tw = 0;

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
        *tw = weight_total;
    });
}

// d loss / d self is zero everywhere except at [i, target[i]], where it is
// -weight[target[i]] * grad_i (grad / total_weight for the mean)
// From: aten/src/ATen/native/LossNLL.cpp (nll_loss_backward_out_frame)
void nll_loss_backward_cpu(
    const Tensor& grad_input, const Tensor& grad, const Tensor& target, const Tensor& weight,
    const Tensor& total_weight, Reduction reduction, int64_t ignore_index
) {
    const int64_t batch_size = grad_input.ndim() == 2 ? grad_input.shape()[0] : 1;
    const int64_t num_classes = grad_input.shape()[grad_input.ndim() - 1];
    const bool per_sample = reduction == Reduction::None && grad_input.ndim() == 2;
    const int64_t* t = target.data_ptr<int64_t>();

    grad_input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        scalar_t* gi = grad_input.mutable_data_ptr<scalar_t>();
        const scalar_t* g = grad.data_ptr<scalar_t>();
        const scalar_t* w = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
        const scalar_t reduced_grad = reduction == Reduction::Mean
            ? -(g[0] / *total_weight.data_ptr<scalar_t>())
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
}

} // namespace

namespace ops {

NllLossResult nll_loss_forward(
    const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index
) {
    check_nll_loss_inputs(self, target, weight);
    check_same_device("nll_loss", self, target, weight);
    const int64_t batch_size = self.ndim() == 2 ? self.shape()[0] : 1;
    const bool per_sample = reduction == Reduction::None && self.ndim() == 2;
    Tensor output = per_sample ? empty({batch_size}, self.options()) : empty({}, self.options());
    Tensor total_weight = empty({}, self.options());
    nll_loss_forward_stub(
        self.device().type(), output, total_weight, self.contiguous(), target.contiguous(),
        weight.defined() ? weight.contiguous() : weight, reduction, ignore_index);
    return {output, total_weight};
}

Tensor nll_loss_backward(
    const Tensor& grad, const Tensor& self, const Tensor& target, const Tensor& weight,
    Reduction reduction, int64_t ignore_index, const Tensor& total_weight
) {
    check_same_device("nll_loss_backward", self, grad, target, weight, total_weight);
    Tensor grad_input = zeros(self.shape(), self.options());
    nll_loss_backward_stub(
        self.device().type(), grad_input, grad.contiguous(), target.contiguous(),
        weight.defined() ? weight.contiguous() : weight, total_weight, reduction, ignore_index);
    return grad_input;
}

} // namespace ops

// -- Composite losses --

Tensor mse_loss(const Tensor& self, const Tensor& target, Reduction reduction) {
    Tensor diff = self - target;
    return apply_reduction(diff * diff, reduction);
}

// With label smoothing the target class gets 1 - ls + ls / C of the
// probability and every other class ls / C, so the loss is
// (1 - ls) * nll + ls / C * (sum over the classes of -log p), with the
// classes weighted and the ignored targets left out as in nll_loss.
// From: aten/src/ATen/native/LossNLL.cpp (cross_entropy_loss_label_smoothing)
Tensor cross_entropy(
    const Tensor& self, const Tensor& target, const Tensor& weight, Reduction reduction, int64_t ignore_index,
    double label_smoothing
) {
    RAXIS_CHECK(label_smoothing >= 0.0 && label_smoothing <= 1.0,
                "label_smoothing must be between 0.0 and 1.0. Got: ", label_smoothing);
    int64_t class_dim = self.ndim() == 1 ? 0 : 1;
    Tensor input = log_softmax(self, class_dim);
    Tensor nll = nll_loss(input, target, weight, reduction, ignore_index);
    if (label_smoothing == 0.0) {
        return nll;
    }
    int64_t n_classes = input.shape()[class_dim];
    Tensor smooth_loss;
    if (weight.defined()) {
        std::vector<int64_t> weight_shape(input.ndim(), 1);
        weight_shape[class_dim] = weight.shape()[0];
        smooth_loss = -(input * weight.view(weight_shape)).sum({class_dim});
    } else {
        smooth_loss = -input.sum({class_dim});
    }
    Tensor ignore_mask = eq(target, Scalar(ignore_index));
    smooth_loss = masked_fill(smooth_loss, ignore_mask, 0.0);

    Tensor ret;
    switch (reduction) {
        case Reduction::Mean:
            if (weight.defined()) {
                // Normalized by the weights of the kept targets, as nll_loss
                Tensor kept_target = masked_fill(target, ignore_mask, 0);
                Tensor target_weights = masked_fill(index_select(weight, 0, kept_target.flatten()), ignore_mask.flatten(), 0.0);
                ret = smooth_loss.sum() / target_weights.sum();
            } else {
                ret = smooth_loss.sum() / logical_not(ignore_mask).sum();
            }
            break;
        case Reduction::Sum:
            ret = smooth_loss.sum();
            break;
        case Reduction::None:
            ret = smooth_loss;
            break;
    }
    return nll * (1.0 - label_smoothing) + ret * (label_smoothing / static_cast<double>(n_classes));
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

DEFINE_DISPATCH(nll_loss_forward_stub);
DEFINE_DISPATCH(nll_loss_backward_stub);

REGISTER_ALL_CPU_DISPATCH(nll_loss_forward_stub, &nll_loss_forward_cpu)
REGISTER_ALL_CPU_DISPATCH(nll_loss_backward_stub, &nll_loss_backward_cpu)

} // namespace raxis
