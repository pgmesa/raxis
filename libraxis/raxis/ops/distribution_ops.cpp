
#include <raxis/ops/distribution_ops.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include <raxis/core/dtype.h>
#include <raxis/core/generator.h>
#include <raxis/core/philox.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/ops/copy.h>

#include <raxis/util/exception.h>


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

} // namespace ops


DEFINE_DISPATCH(uniform_stub);
DEFINE_DISPATCH(normal_stub);
DEFINE_DISPATCH(randint_stub);

} // namespace raxis
