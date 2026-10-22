#pragma once

#include <array>
#include <cstdint>


namespace raxis {

// Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3",
// SC11), a counter-based generator.
//
// A sequential generator like mt19937 has a state that every draw advances,
// so the i-th number can only be reached by drawing the previous ones: one
// thread at a time, under a lock. Philox has no state: it is a keyed
// bijection that scrambles a 128-bit counter into 128 random bits with 10
// rounds of multiply and xor. The i-th number of a stream is computed
// directly as philox(start + i / 4, seed)[i % 4], so threads produce their
// slices independently, and the values do not depend on how the work is
// split (or on the number of threads).
//
// From: aten/src/ATen/core/PhiloxRNGEngine.h (philox_engine::rand,
// single_round).

using PhiloxBlock = std::array<uint32_t, 4>;

namespace detail {

inline constexpr uint32_t kPhilox10A = 0x9E3779B9;
inline constexpr uint32_t kPhilox10B = 0xBB67AE85;
inline constexpr uint32_t kPhiloxSA = 0xD2511F53;
inline constexpr uint32_t kPhiloxSB = 0xCD9E8D57;

inline PhiloxBlock philox_round(const PhiloxBlock& ctr, uint32_t key0, uint32_t key1) {
    const uint64_t product0 = static_cast<uint64_t>(kPhiloxSA) * ctr[0];
    const uint64_t product1 = static_cast<uint64_t>(kPhiloxSB) * ctr[2];
    const auto hi0 = static_cast<uint32_t>(product0 >> 32);
    const auto hi1 = static_cast<uint32_t>(product1 >> 32);
    return {
        hi1 ^ ctr[1] ^ key0,
        static_cast<uint32_t>(product1),
        hi0 ^ ctr[3] ^ key1,
        static_cast<uint32_t>(product0),
    };
}

} // namespace detail

inline PhiloxBlock philox4x32_10(PhiloxBlock ctr, uint64_t key) {
    auto key0 = static_cast<uint32_t>(key);
    auto key1 = static_cast<uint32_t>(key >> 32);
    for (int round = 0; round < 9; ++round) {
        ctr = detail::philox_round(ctr, key0, key1);
        key0 += detail::kPhilox10A;
        key1 += detail::kPhilox10B;
    }
    return detail::philox_round(ctr, key0, key1);
}

// Block `counter` of the stream of `key` (the upper 64 counter bits are 0)
inline PhiloxBlock philox4x32_10(uint64_t counter, uint64_t key) {
    return philox4x32_10(
        PhiloxBlock{static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32), 0, 0}, key);
}

// Lanes consecutive blocks starting at `counter`, written word-major:
// out[w * Lanes + l] is word w of block counter + l. The lanes are
// independent, so with one array per word (structure of arrays) every step
// of a round is the same operation on Lanes values, which maps to SIMD (one
// register per word); word-major output lets a SIMD version store each
// register as is. Every implementation must keep this order, so a stream
// gives the same numbers on every CPU.
template <int Lanes>
inline void philox4x32_10_lanes(uint64_t counter, uint64_t key, uint32_t* out) {
    uint32_t c0[Lanes], c1[Lanes], c2[Lanes], c3[Lanes];
    for (int l = 0; l < Lanes; ++l) {
        const uint64_t ctr = counter + static_cast<uint64_t>(l);
        c0[l] = static_cast<uint32_t>(ctr);
        c1[l] = static_cast<uint32_t>(ctr >> 32);
        c2[l] = 0;
        c3[l] = 0;
    }
    auto key0 = static_cast<uint32_t>(key);
    auto key1 = static_cast<uint32_t>(key >> 32);
    for (int round = 0; round < 10; ++round) {
        for (int l = 0; l < Lanes; ++l) {
            const uint64_t product0 = static_cast<uint64_t>(detail::kPhiloxSA) * c0[l];
            const uint64_t product1 = static_cast<uint64_t>(detail::kPhiloxSB) * c2[l];
            const auto next0 = static_cast<uint32_t>(product1 >> 32) ^ c1[l] ^ key0;
            const auto next2 = static_cast<uint32_t>(product0 >> 32) ^ c3[l] ^ key1;
            c1[l] = static_cast<uint32_t>(product1);
            c3[l] = static_cast<uint32_t>(product0);
            c0[l] = next0;
            c2[l] = next2;
        }
        key0 += detail::kPhilox10A;
        key1 += detail::kPhilox10B;
    }
    for (int l = 0; l < Lanes; ++l) {
        out[0 * Lanes + l] = c0[l];
        out[1 * Lanes + l] = c1[l];
        out[2 * Lanes + l] = c2[l];
        out[3 * Lanes + l] = c3[l];
    }
}

} // namespace raxis
