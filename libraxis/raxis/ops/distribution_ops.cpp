
#include <raxis/ops/distribution_ops.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <raxis/core/dtype.h>
#include <raxis/core/generator.h>
#include <raxis/core/philox.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/ops/binary_ops.h>
#include <raxis/ops/copy.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/sorting.h>
#include <raxis/ops/unary_ops.h>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>


namespace raxis {

namespace {

// Runs a kernel that fills a contiguous tensor: on self directly, or on a
// contiguous buffer copied into self, so element i of the stream always goes
// to the i-th element in row-major order, whatever the strides of self.
template <typename Fill>
void fill_contiguous(Tensor& self, int words_per_element, const Fill& fill) {
    const auto [seed, offset] =
        get_default_generator().philox_engine_inputs(philox_blocks(self.numel(), words_per_element));
    if (self.is_contiguous()) {
        fill(self, seed, offset);
        return;
    }
    Tensor buffer = empty(self.shape(), self.options());
    fill(buffer, seed, offset);
    Tensor out = self;
    copy(buffer, out);
}

} // namespace

namespace ops {

Tensor& uniform_(Tensor& self, double low, double high) {
    RAXIS_CHECK(
        self.dtype().is_floating(),
        "uniform_ expects a floating-point dtype, got ", self.dtype()
    );
    RAXIS_CHECK(
        low <= high,
        "uniform_ expects low (", low, ") <= high (", high, ")"
    );
    if (self.numel() == 0) return self;

    fill_contiguous(self, real_words_per_element(self.dtype()), [&](const Tensor& t, uint64_t seed, uint64_t offset) {
        uniform_stub(DeviceType::CPU, t, low, high, seed, offset);
    });
    return self;
}

Tensor& normal_(Tensor& self, double mean, double std) {
    RAXIS_CHECK(
        self.dtype().is_floating(),
        "normal_ expects a floating-point dtype, got ", self.dtype()
    );
    RAXIS_CHECK(std >= 0.0, "normal_ expects std >= 0, got ", std);
    if (self.numel() == 0) return self;

    fill_contiguous(self, real_words_per_element(self.dtype()), [&](const Tensor& t, uint64_t seed, uint64_t offset) {
        normal_stub(DeviceType::CPU, t, mean, std, seed, offset);
    });
    return self;
}

Tensor& randint_(Tensor& self, int64_t low, int64_t high) {
    RAXIS_CHECK(
        self.dtype().is_integral(true),
        "randint_ expects an integral or bool dtype, got ", self.dtype()
    );
    RAXIS_CHECK(
        low < high,
        "randint_ expects low (", low, ") < high (", high, ")"
    );

    if (self.numel() == 0) return self;

    auto range = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);

    self.dtype().dispatch_integral_and_bool_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using lim = std::numeric_limits<scalar_t>;
        RAXIS_CHECK(
            low >= static_cast<int64_t>(lim::lowest()) &&
            (high - 1) <= static_cast<int64_t>(lim::max()),
            "randint_ range [", low, ", ", high, ") is outside the range of ", self.dtype()
        );
    });

    fill_contiguous(self, kRandintWordsPerElement, [&](const Tensor& t, uint64_t seed, uint64_t offset) {
        randint_stub(DeviceType::CPU, t, low, range, seed, offset);
    });
    return self;
}

// From: aten/src/ATen/native/Distributions.cpp (multinomial_out) and
// cpu/MultinomialKernel.cpp (multinomial_with_replacement_apply)
Tensor multinomial(const Tensor& self, int64_t num_samples, bool replacement) {
    RAXIS_CHECK(self.ndim() > 0 && self.ndim() <= 2, "prob_dist must be 1 or 2 dim");
    RAXIS_CHECK(self.dtype().is_floating(),
        "multinomial only supports floating-point dtypes for input, got: ", self.dtype());
    RAXIS_CHECK(num_samples > 0, "cannot sample n_sample <= 0 samples");
    const int64_t n_categories = self.shape()[self.ndim() - 1];
    RAXIS_CHECK(replacement || num_samples <= n_categories,
        "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
    // Indices travel through float in torch's GPU kernels; kept for the same limits
    RAXIS_CHECK(n_categories <= (int64_t{1} << 24), "number of categories cannot exceed 2^24");

    const int64_t n_dist = self.ndim() == 2 ? self.shape()[0] : 1;
    DimVector result_shape = self.ndim() == 1 ? DimVector{num_samples} : DimVector{n_dist, num_samples};
    if (n_dist == 0) {
        return empty(result_shape, TensorOptions().dtype(dtypes::Int64));
    }
    // torch checks this asynchronously on this path and row by row in the
    // kernel on the other; NaN fails both comparisons
    RAXIS_CHECK(ops::amin(self).item().to<double>() >= 0 && ops::amax(self).item().to<double>() < INFINITY,
        "probability tensor contains either `inf`, `nan` or element < 0");
    const Tensor row_sums = self.ndim() == 1 ? ops::sum(self) : ops::sum(self, {1});
    RAXIS_CHECK(ops::amin(row_sums).item().to<double>() > 0,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    if (!replacement || num_samples == 1) {
        // The Gumbel-max trick: argmax(log p + g) with g ~ Gumbel(0, 1) is a
        // sample of p, and the top k of those keys are a sample of k without
        // replacement. Taking exp, argmax(log p + g) = argmax(p / q) with
        // q = -log(u) ~ Exp(1), u ~ U(0, 1): one draw per category, and the
        // weights need no normalization
        Tensor u = empty(self.shape(), self.options());
        ops::uniform_(u);
        // -log1p(-u) is Exp(1) for u in [0, 1), as torch's exponential_
        const Tensor q = ops::neg(ops::log1p(ops::neg(u)));
        const Tensor keys = ops::div(self, q);
        if (num_samples == 1) {
            return ops::argmax(keys, -1, /*keepdim=*/true);
        }
        return std::get<1>(ops::topk(keys, num_samples));
    }

    // With replacement: inverse transform sampling, a binary search of each
    // uniform draw in the row's normalized cumulative distribution. The
    // draws come from one Philox fill, so the rows run in parallel and the
    // samples do not depend on the number of threads
    Tensor uniform = empty(result_shape, TensorOptions().dtype(dtypes::Float64));
    ops::uniform_(uniform);
    Tensor result = empty(result_shape, TensorOptions().dtype(dtypes::Int64));
    const Tensor self_c = self.contiguous();
    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* self_data = self_c.data_ptr<scalar_t>();
        const double* uniform_data = uniform.data_ptr<double>();
        int64_t* result_data = result.mutable_data_ptr<int64_t>();
        parallel_for(0, n_dist, 1, [&](int64_t begin, int64_t end) {
            std::vector<scalar_t> cum_dist(n_categories);
            for (int64_t i = begin; i < end; ++i) {
                const scalar_t* probs = self_data + i * n_categories;
                scalar_t sum = 0;
                for (int64_t j = 0; j < n_categories; ++j) {
                    sum += probs[j];
                    cum_dist[j] = sum;
                }
                for (int64_t j = 0; j < n_categories; ++j) {
                    cum_dist[j] /= sum;
                }
                // The last bucket must end at exactly 1
                cum_dist[n_categories - 1] = 1;
                for (int64_t s = 0; s < num_samples; ++s) {
                    const double sample = uniform_data[i * num_samples + s];
                    // First slot with cum_dist[slot] >= sample
                    int64_t left = 0;
                    int64_t right = n_categories;
                    while (right - left > 0) {
                        const int64_t mid = left + (right - left) / 2;
                        if (cum_dist[mid] < sample) {
                            left = mid + 1;
                        } else {
                            right = mid;
                        }
                    }
                    result_data[i * num_samples + s] = left;
                }
            }
        });
    });
    return result;
}

} // namespace ops

Tensor multinomial(const Tensor& self, int64_t num_samples, bool replacement) {
    return ops::multinomial(self, num_samples, replacement);
}


DEFINE_DISPATCH(uniform_stub);
DEFINE_DISPATCH(normal_stub);
DEFINE_DISPATCH(randint_stub);

} // namespace raxis
