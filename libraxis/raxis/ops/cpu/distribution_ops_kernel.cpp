
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/distribution_ops.h>
#include <raxis/ops/cpu/philox_batch.h>
#include <raxis/ops/cpu/vec/vec.h>

#include <raxis/core/philox.h>

#include <raxis/tensor/tensor.h>


#include <cstdint>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

// Elements per parallel task
constexpr int64_t kDistributionGrain = 8192;

// Words per element, as real_words_per_element in distribution_ops.h
template <typename scalar_t>
constexpr int64_t kRealWords = sizeof(scalar_t) == 8 ? 2 : 1;

// Uniform sample in [0, 1) for element j of a batch of
// kPhiloxBatchWords / kRealWords elements: float keeps 24 random bits of
// word j and double 53 bits of words j and j + 32 (contiguous loads, so the
// loops over j vectorize). The int32/int64 casts are exact (the values fit)
// and convert faster than from unsigned.
// From: aten/src/ATen/core/TransformationHelper.h (uniform_real)
template <typename scalar_t>
inline scalar_t unit_uniform(const uint32_t* r, int64_t j) {
    if constexpr (std::is_same_v<scalar_t, double>) {
        const uint64_t bits = (static_cast<uint64_t>(r[j + kPhiloxBatchWords / 2]) << 32) | r[j];
        return static_cast<double>(static_cast<int64_t>(bits & ((uint64_t{1} << 53) - 1))) * 0x1p-53;
    } else {
        return static_cast<float>(static_cast<int32_t>(r[j] & ((1u << 24) - 1))) * 0x1p-24f;
    }
}

// From: aten/src/ATen/native/cpu/DistributionTemplates.h (uniform_kernel)
template <typename scalar_t>
void uniform_impl(scalar_t* out, int64_t n, double low, double high, uint64_t seed, uint64_t offset) {
    constexpr int64_t per_batch = kPhiloxBatchWords / kRealWords<scalar_t>;
    const auto from = static_cast<scalar_t>(low);
    const auto to = static_cast<scalar_t>(high);
    const scalar_t range = to - from;
    philox_for_each_batch(n, per_batch, seed, offset, kDistributionGrain,
        [&](const uint32_t* r, int64_t first, int64_t count) {
            scalar_t* dst = out + first;
            for (int64_t j = 0; j < count; ++j) {
                const scalar_t value = unit_uniform<scalar_t>(r, j) * range + from;
                // x * range + from can round up to `to`
                dst[j] = value == to ? from : value;
            }
        });
}

// Box-Muller on 16 uniforms in place: the pairs (data[j], data[j + 8]) become
// two independent normals radius * cos(theta) and radius * sin(theta), with
// radius = sqrt(-2 log(1 - u1)) (1 - u1 is in (0, 1], so log is finite) and
// theta = 2 pi u2. Pairing j with j + 8 instead of 2j with 2j + 1 keeps the
// loads contiguous, so each half is one AVX2 register for float.
// From: aten/src/ATen/native/cpu/DistributionTemplates.h (NormalFill16)
template <typename scalar_t>
inline void box_muller_16(scalar_t* data, vec::Vectorized<scalar_t> mean, vec::Vectorized<scalar_t> stddev) {
    using Vec = vec::Vectorized<scalar_t>;
    const Vec one(1);
    const Vec minus_two(-2);
    const Vec two_pi(static_cast<scalar_t>(2.0 * 3.14159265358979323846));
    for (int j = 0; j < 8; j += Vec::size()) {
        const Vec u1 = one - Vec::loadu(data + j);
        const Vec u2 = Vec::loadu(data + j + 8);
        const Vec radius = (minus_two * u1.log()).sqrt();
        const Vec theta = two_pi * u2;
        vec::fmadd(radius * theta.cos(), stddev, mean).store(data + j);
        vec::fmadd(radius * theta.sin(), stddev, mean).store(data + j + 8);
    }
}

// PyTorch fills the whole tensor with uniforms, transforms it in place 16 at
// a time and redraws the last 16 when numel is not a multiple of 16. Here
// every batch transforms its own uniforms in a stack buffer (a batch is a
// multiple of 16 elements) and stores the first `count`, so the tail needs
// no special case and threads never share a chunk.
// From: aten/src/ATen/native/cpu/DistributionTemplates.h (normal_fill)
template <typename scalar_t>
void normal_impl(scalar_t* out, int64_t n, double mean_, double std_, uint64_t seed, uint64_t offset) {
    constexpr int64_t per_batch = kPhiloxBatchWords / kRealWords<scalar_t>;
    static_assert(per_batch % 16 == 0);
    const vec::Vectorized<scalar_t> mean(static_cast<scalar_t>(mean_));
    const vec::Vectorized<scalar_t> stddev(static_cast<scalar_t>(std_));
    philox_for_each_batch(n, per_batch, seed, offset, kDistributionGrain,
        [&](const uint32_t* r, int64_t first, int64_t count) {
            alignas(32) scalar_t samples[per_batch];
            for (int64_t j = 0; j < per_batch; ++j) {
                samples[j] = unit_uniform<scalar_t>(r, j);
            }
            for (int64_t j = 0; j < per_batch; j += 16) {
                box_muller_16(samples + j, mean, stddev);
            }
            scalar_t* dst = out + first;
            for (int64_t j = 0; j < count; ++j) {
                dst[j] = samples[j];
            }
        });
}

// High 64 bits of the 128-bit product a * b, from four 32 x 32 -> 64-bit
// products (portable: MSVC has no 128-bit integer type)
inline uint64_t mulhi_u64(uint64_t a, uint64_t b) {
    const uint64_t a_lo = static_cast<uint32_t>(a);
    const uint64_t a_hi = a >> 32;
    const uint64_t b_lo = static_cast<uint32_t>(b);
    const uint64_t b_hi = b >> 32;
    const uint64_t lo_lo = a_lo * b_lo;
    const uint64_t hi_lo = a_hi * b_lo;
    const uint64_t lo_hi = a_lo * b_hi;
    // Cannot overflow: lo_hi <= (2^32 - 1)^2 and the other terms are < 2^32
    const uint64_t cross = (lo_lo >> 32) + static_cast<uint32_t>(hi_lo) + lo_hi;
    return a_hi * b_hi + (hi_lo >> 32) + (cross >> 32);
}

// PyTorch maps a 32 or 64-bit draw to [0, range) with draw % range
// (TransformationHelper.h, uniform_int_from_to): a 64-bit division per
// element, and the result comes from the low bits of the draw. Lemire's
// multiply-shift floor(draw * range / 2^64) takes the high 64 bits of one
// multiply instead. Both have a bias below range / 2^64, negligible here.
template <typename scalar_t>
void randint_impl(scalar_t* out, int64_t n, int64_t low, uint64_t range, uint64_t seed, uint64_t offset) {
    constexpr int64_t per_batch = kPhiloxBatchWords / kRandintWordsPerElement;
    philox_for_each_batch(n, per_batch, seed, offset, kDistributionGrain,
        [&](const uint32_t* r, int64_t first, int64_t count) {
            scalar_t* dst = out + first;
            for (int64_t j = 0; j < count; ++j) {
                const uint64_t draw = (static_cast<uint64_t>(r[j + per_batch]) << 32) | r[j];
                dst[j] = static_cast<scalar_t>(low + static_cast<int64_t>(mulhi_u64(draw, range)));
            }
        });
}

void uniform_kernel(const Tensor& self, double low, double high, uint64_t seed, uint64_t offset) {
    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        uniform_impl<scalar_t>(self.mutable_data_ptr<scalar_t>(), self.numel(), low, high, seed, offset);
    });
}

void normal_kernel(const Tensor& self, double mean, double stddev, uint64_t seed, uint64_t offset) {
    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        normal_impl<scalar_t>(self.mutable_data_ptr<scalar_t>(), self.numel(), mean, stddev, seed, offset);
    });
}

void randint_kernel(const Tensor& self, int64_t low, uint64_t range, uint64_t seed, uint64_t offset) {
    self.dtype().dispatch_integral_and_bool_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        randint_impl<scalar_t>(self.mutable_data_ptr<scalar_t>(), self.numel(), low, range, seed, offset);
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(uniform_stub, &uniform_kernel);
REGISTER_DISPATCH(normal_stub,  &normal_kernel);
REGISTER_DISPATCH(randint_stub, &randint_kernel);

} // namespace raxis
