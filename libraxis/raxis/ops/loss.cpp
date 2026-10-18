#include <raxis/ops/loss.h>

#include <cmath>
#include <limits>
#include <string>

#include <raxis/util/exception.h>
#include <raxis/tensor/factories.h>
#include <raxis/ops/activation.h>
#include <raxis/ops/binary_ops.h>
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

void check_nll_loss_inputs(const Tensor& self, const Tensor& target) {
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
}

} // namespace

namespace ops {

NllLossResult nll_loss_forward(
    const Tensor& self, const Tensor& target, Reduction reduction, int64_t ignore_index
) {
    check_nll_loss_inputs(self, target);
    int64_t batch_size = self.ndim() == 2 ? self.shape()[0] : 1;
    int64_t num_classes = self.shape()[self.ndim() - 1];

    Tensor input = self.contiguous();
    Tensor target_c = target.contiguous();
    const int64_t* t = target_c.data_ptr<int64_t>();

    bool per_sample = reduction == Reduction::None && self.ndim() == 2;
    Tensor output = per_sample
        ? empty({batch_size}, self.options())
        : empty({}, self.options());

    double total_weight = 0.0;
    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* x = input.data_ptr<scalar_t>();
        scalar_t* out = output.mutable_data_ptr<scalar_t>();
        double total = 0.0;

        for (int64_t i = 0; i < batch_size; ++i) {
            int64_t cls = t[i];
            double loss = 0.0;
            if (cls != ignore_index) {
                RAXIS_CHECK(cls >= 0 && cls < num_classes,
                    "nll_loss: target ", cls, " is out of bounds for ", num_classes, " classes");
                loss = -static_cast<double>(x[i * num_classes + cls]);
                total_weight += 1.0;
            }
            if (per_sample) {
                out[i] = static_cast<scalar_t>(loss);
            }
            total += loss;
        }

        if (!per_sample) {
            // Mean over zero valid targets is 0/0 = NaN, as in PyTorch
            if (reduction == Reduction::Mean) {
                total = total_weight > 0.0
                    ? total / total_weight
                    : std::numeric_limits<double>::quiet_NaN();
            }
            out[0] = static_cast<scalar_t>(total);
        }
    });
    return {output, total_weight};
}

// d loss / d self is zero everywhere except at [i, target[i]], where it is
// -grad_i (scaled by 1 / total_weight for the mean)
Tensor nll_loss_backward(
    const Tensor& grad, const Tensor& self, const Tensor& target,
    Reduction reduction, int64_t ignore_index, double total_weight
) {
    int64_t batch_size = self.ndim() == 2 ? self.shape()[0] : 1;
    int64_t num_classes = self.shape()[self.ndim() - 1];
    bool per_sample = reduction == Reduction::None && self.ndim() == 2;

    Tensor grad_input = zeros(self.shape(), self.options());
    Tensor target_c = target.contiguous();
    Tensor grad_c = grad.contiguous();
    const int64_t* t = target_c.data_ptr<int64_t>();

    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        scalar_t* gi = grad_input.mutable_data_ptr<scalar_t>();
        const scalar_t* g = grad_c.data_ptr<scalar_t>();
        scalar_t scale = static_cast<scalar_t>(1);
        if (reduction == Reduction::Mean) {
            scale = static_cast<scalar_t>(1.0 / total_weight);
        }

        for (int64_t i = 0; i < batch_size; ++i) {
            int64_t cls = t[i];
            if (cls == ignore_index) continue;
            scalar_t g_i = per_sample ? g[i] : g[0];
            gi[i * num_classes + cls] = -g_i * scale;
        }
    });
    return grad_input;
}

} // namespace ops

// -- Composite losses --

Tensor mse_loss(const Tensor& self, const Tensor& target, Reduction reduction) {
    Tensor diff = self - target;
    return apply_reduction(diff * diff, reduction);
}

Tensor cross_entropy(const Tensor& self, const Tensor& target, Reduction reduction, int64_t ignore_index) {
    int64_t class_dim = self.ndim() == 1 ? 0 : 1;
    return nll_loss(log_softmax(self, class_dim), target, reduction, ignore_index);
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
