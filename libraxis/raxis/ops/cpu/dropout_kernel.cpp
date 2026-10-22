#include <raxis/ops/dropout.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/core/philox.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

#if defined(CPU_CAPABILITY_AVX2)
#include <immintrin.h>
#endif


namespace raxis {

inline namespace CPU_CAPABILITY {

// Elements per parallel task
constexpr int64_t kDropoutGrain = 8192;

#if defined(CPU_CAPABILITY_AVX2)

// Philox for 8 lanes, one 32-bit lane per block. AVX2 has no 32x32 -> 64-bit
// multiply on all 8 lanes: _mm256_mul_epu32 multiplies the even lanes into
// 64-bit products, so the odd lanes are shifted down and multiplied apart,
// and the low and high halves are blended back into 8 lanes.
inline void mulhilo_epu32(__m256i a, __m256i m, __m256i& lo, __m256i& hi) {
    const __m256i even = _mm256_mul_epu32(a, m);
    const __m256i odd = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), m);
    lo = _mm256_blend_epi32(even, _mm256_slli_epi64(odd, 32), 0b10101010);
    hi = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0b10101010);
}

// Same output as philox4x32_10_lanes<kDropoutLanes>: two registers per word
inline void philox_batch(uint64_t counter, uint64_t key, uint32_t* out) {
    alignas(32) uint32_t lo[kDropoutLanes];
    alignas(32) uint32_t hi[kDropoutLanes];
    for (int l = 0; l < kDropoutLanes; ++l) {
        const uint64_t ctr = counter + static_cast<uint64_t>(l);
        lo[l] = static_cast<uint32_t>(ctr);
        hi[l] = static_cast<uint32_t>(ctr >> 32);
    }
    const __m256i sa = _mm256_set1_epi32(static_cast<int>(detail::kPhiloxSA));
    const __m256i sb = _mm256_set1_epi32(static_cast<int>(detail::kPhiloxSB));
    __m256i c0[2], c1[2], c2[2], c3[2];
    for (int h = 0; h < 2; ++h) {
        c0[h] = _mm256_load_si256(reinterpret_cast<const __m256i*>(lo + 8 * h));
        c1[h] = _mm256_load_si256(reinterpret_cast<const __m256i*>(hi + 8 * h));
        c2[h] = _mm256_setzero_si256();
        c3[h] = _mm256_setzero_si256();
    }
    auto key0 = static_cast<uint32_t>(key);
    auto key1 = static_cast<uint32_t>(key >> 32);
    for (int round = 0; round < 10; ++round) {
        const __m256i k0 = _mm256_set1_epi32(static_cast<int>(key0));
        const __m256i k1 = _mm256_set1_epi32(static_cast<int>(key1));
        for (int h = 0; h < 2; ++h) {
            __m256i lo0, hi0, lo1, hi1;
            mulhilo_epu32(c0[h], sa, lo0, hi0);
            mulhilo_epu32(c2[h], sb, lo1, hi1);
            c0[h] = _mm256_xor_si256(_mm256_xor_si256(hi1, c1[h]), k0);
            c2[h] = _mm256_xor_si256(_mm256_xor_si256(hi0, c3[h]), k1);
            c1[h] = lo1;
            c3[h] = lo0;
        }
        key0 += detail::kPhilox10A;
        key1 += detail::kPhilox10B;
    }
    for (int h = 0; h < 2; ++h) {
        auto* dst = reinterpret_cast<__m256i*>(out + 8 * h);
        _mm256_storeu_si256(dst + 0 * kDropoutLanes / 8, c0[h]);
        _mm256_storeu_si256(dst + 1 * kDropoutLanes / 8, c1[h]);
        _mm256_storeu_si256(dst + 2 * kDropoutLanes / 8, c2[h]);
        _mm256_storeu_si256(dst + 3 * kDropoutLanes / 8, c3[h]);
    }
}

#else

inline void philox_batch(uint64_t counter, uint64_t key, uint32_t* out) {
    philox4x32_10_lanes<kDropoutLanes>(counter, key, out);
}

#endif

// From: aten/src/ATen/native/Dropout.cpp (native_dropout_cpu) for the math,
// and PyTorch's CUDA dropout for the generation (Philox counters reserved
// from the generator, so any thread can draw any batch).
//
// Element i is kept when its 32-bit draw r is below p_keep * 2^32, i.e.
// u = r / 2^32 < p_keep: an exact integer comparison with no conversion to
// floating point. One pass draws, writes the mask and
// output = input * keep * scale, the same operation order as PyTorch's
// input.mul(mask).mul_(scale) (so NaN inputs stay NaN when dropped).
template <typename scalar_t>
void fused_dropout_impl(
    scalar_t* out, bool* mask, const scalar_t* in, int64_t n,
    double p_keep, uint64_t seed, uint64_t offset
) {
    const auto threshold = static_cast<uint64_t>(std::ldexp(p_keep, 32));
    const auto scale = static_cast<scalar_t>(p_keep == 0 ? 0.0 : 1.0 / p_keep);
    const int64_t batches = (n + kDropoutBatch - 1) / kDropoutBatch;

    parallel_for(0, batches, kDropoutGrain / kDropoutBatch, [&](int64_t begin, int64_t end) {
        uint32_t r[kDropoutBatch];
        for (int64_t batch = begin; batch < end; ++batch) {
            philox_batch(offset + static_cast<uint64_t>(batch * kDropoutLanes), seed, r);
            const int64_t base = batch * kDropoutBatch;
            const int64_t count = std::min<int64_t>(kDropoutBatch, n - base);
            if (count == kDropoutBatch) {
                // Constant trip counts and one simple statement per loop, so
                // the compiler vectorizes both
                bool* m = mask + base;
                for (int64_t j = 0; j < kDropoutBatch; ++j) {
                    m[j] = r[j] < threshold;
                }
                for (int64_t j = 0; j < kDropoutBatch; ++j) {
                    out[base + j] = in[base + j] * static_cast<scalar_t>(m[j]) * scale;
                }
                continue;
            }
            for (int64_t j = 0; j < count; ++j) {
                const bool keep = r[j] < threshold;
                mask[base + j] = keep;
                out[base + j] = in[base + j] * static_cast<scalar_t>(keep) * scale;
            }
        }
    });
}

// grad_input = grad * mask * scale, in PyTorch's order
// From: aten/src/ATen/native/Dropout.cpp (native_dropout_backward)
template <typename scalar_t>
void dropout_backward_impl(scalar_t* grad_in, const scalar_t* grad, const bool* mask, int64_t n, double scale_) {
    const auto scale = static_cast<scalar_t>(scale_);
    parallel_for(0, n, kDropoutGrain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            grad_in[i] = grad[i] * static_cast<scalar_t>(mask[i]) * scale;
        }
    });
}

void fused_dropout_kernel(
    const Tensor& output, const Tensor& mask, const Tensor& input,
    double p_keep, uint64_t seed, uint64_t offset
) {
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        fused_dropout_impl<scalar_t>(
            output.mutable_data_ptr<scalar_t>(), mask.mutable_data_ptr<bool>(),
            input.data_ptr<scalar_t>(), input.numel(), p_keep, seed, offset);
    });
}

void dropout_backward_kernel(const Tensor& grad_input, const Tensor& grad, const Tensor& mask, double scale) {
    grad.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        dropout_backward_impl<scalar_t>(
            grad_input.mutable_data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(), grad.numel(), scale);
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(fused_dropout_stub, &fused_dropout_kernel);
REGISTER_DISPATCH(dropout_backward_stub, &dropout_backward_kernel);

} // namespace raxis
