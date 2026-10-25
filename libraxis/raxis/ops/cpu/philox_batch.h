#pragma once

#include <algorithm>
#include <cstdint>

#include <raxis/core/philox.h>

#include <raxis/util/parallel.h>

#if defined(CPU_CAPABILITY_AVX2) || defined(CPU_CAPABILITY_AVX512)
#include <immintrin.h>
#endif


namespace raxis {

inline namespace CPU_CAPABILITY {

#if defined(CPU_CAPABILITY_AVX512)

// The AVX2 algorithm below with all 16 lanes in one register per word: the
// even and odd products are merged with a mask blend
inline void mulhilo_epu32(__m512i a, __m512i m, __m512i& lo, __m512i& hi) {
    const __m512i even = _mm512_mul_epu32(a, m);
    const __m512i odd = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), m);
    lo = _mm512_mask_blend_epi32(0xAAAA, even, _mm512_slli_epi64(odd, 32));
    hi = _mm512_mask_blend_epi32(0xAAAA, _mm512_srli_epi64(even, 32), odd);
}

// Same output as philox4x32_10_lanes<kPhiloxLanes>: one register per word
inline void philox_batch(uint64_t counter, uint64_t key, uint32_t* out) {
    static_assert(kPhiloxLanes == 16);
    alignas(64) uint32_t lo[kPhiloxLanes];
    alignas(64) uint32_t hi[kPhiloxLanes];
    for (int l = 0; l < kPhiloxLanes; ++l) {
        const uint64_t ctr = counter + static_cast<uint64_t>(l);
        lo[l] = static_cast<uint32_t>(ctr);
        hi[l] = static_cast<uint32_t>(ctr >> 32);
    }
    const __m512i sa = _mm512_set1_epi32(static_cast<int>(detail::kPhiloxSA));
    const __m512i sb = _mm512_set1_epi32(static_cast<int>(detail::kPhiloxSB));
    __m512i c0 = _mm512_load_si512(lo);
    __m512i c1 = _mm512_load_si512(hi);
    __m512i c2 = _mm512_setzero_si512();
    __m512i c3 = _mm512_setzero_si512();
    auto key0 = static_cast<uint32_t>(key);
    auto key1 = static_cast<uint32_t>(key >> 32);
    for (int round = 0; round < 10; ++round) {
        __m512i lo0, hi0, lo1, hi1;
        mulhilo_epu32(c0, sa, lo0, hi0);
        mulhilo_epu32(c2, sb, lo1, hi1);
        c0 = _mm512_xor_si512(_mm512_xor_si512(hi1, c1), _mm512_set1_epi32(static_cast<int>(key0)));
        c2 = _mm512_xor_si512(_mm512_xor_si512(hi0, c3), _mm512_set1_epi32(static_cast<int>(key1)));
        c1 = lo1;
        c3 = lo0;
        key0 += detail::kPhilox10A;
        key1 += detail::kPhilox10B;
    }
    _mm512_storeu_si512(out + 0 * kPhiloxLanes, c0);
    _mm512_storeu_si512(out + 1 * kPhiloxLanes, c1);
    _mm512_storeu_si512(out + 2 * kPhiloxLanes, c2);
    _mm512_storeu_si512(out + 3 * kPhiloxLanes, c3);
}

#elif defined(CPU_CAPABILITY_AVX2)

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

// Same output as philox4x32_10_lanes<kPhiloxLanes>: two registers per word
inline void philox_batch(uint64_t counter, uint64_t key, uint32_t* out) {
    alignas(32) uint32_t lo[kPhiloxLanes];
    alignas(32) uint32_t hi[kPhiloxLanes];
    for (int l = 0; l < kPhiloxLanes; ++l) {
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
        _mm256_storeu_si256(dst + 0 * kPhiloxLanes / 8, c0[h]);
        _mm256_storeu_si256(dst + 1 * kPhiloxLanes / 8, c1[h]);
        _mm256_storeu_si256(dst + 2 * kPhiloxLanes / 8, c2[h]);
        _mm256_storeu_si256(dst + 3 * kPhiloxLanes / 8, c3[h]);
    }
}

#else

inline void philox_batch(uint64_t counter, uint64_t key, uint32_t* out) {
    philox4x32_10_lanes<kPhiloxLanes>(counter, key, out);
}

#endif

// Calls f(words, first, count) for every batch of `per_batch` elements of
// [0, n), in parallel: words holds the kPhiloxBatchWords draws of the batch
// (blocks offset + batch * kPhiloxLanes onwards, see philox_blocks), first is
// the index of its first element and count is per_batch except in the last
// batch. Element i always gets the same words, whatever the thread count.
template <typename F>
void philox_for_each_batch(int64_t n, int64_t per_batch, uint64_t seed, uint64_t offset, int64_t grain, const F& f) {
    const int64_t batches = (n + per_batch - 1) / per_batch;
    parallel_for(0, batches, std::max<int64_t>(1, grain / per_batch), [&](int64_t begin, int64_t end) {
        alignas(64) uint32_t words[kPhiloxBatchWords];
        for (int64_t batch = begin; batch < end; ++batch) {
            philox_batch(offset + static_cast<uint64_t>(batch) * kPhiloxLanes, seed, words);
            const int64_t first = batch * per_batch;
            f(static_cast<const uint32_t*>(words), first, std::min(per_batch, n - first));
        }
    });
}

}} // namespace raxis::CPU_CAPABILITY
