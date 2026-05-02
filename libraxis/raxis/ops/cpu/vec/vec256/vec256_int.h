#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <raxis/ops/cpu/vec/vec_base.h>


namespace raxis::vec {

inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<int32_t> {
    using value_type = int32_t;
    using size_type = int;
    static constexpr size_type kSize = 8;

private:
    __m256i values_;

    static inline __m256i invert(const __m256i& v) {
        return _mm256_xor_si256(v, _mm256_set1_epi32(-1));
    }

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm256_setzero_si256()) {}

    Vectorized(__m256i v) : values_(v) {}

    Vectorized(int32_t val) : values_(_mm256_set1_epi32(val)) {}

    operator __m256i() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm256_loadu_si256(static_cast<const __m256i*>(ptr));
        }
        alignas(32) int32_t tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(int32_t));
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm256_storeu_si256(static_cast<__m256i*>(ptr), values_);
            return;
        }
        alignas(32) int32_t tmp[kSize];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), values_);
        std::memcpy(ptr, tmp, count * sizeof(int32_t));
    }

    int32_t operator[](int idx) const {
        alignas(32) int32_t tmp[kSize];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), values_);
        return tmp[idx];
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        return _mm256_blend_epi32(a.values_, b.values_, mask);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return _mm256_blendv_epi8(a.values_, b.values_, mask.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        switch (count) {
            case 0: return a;
            case 1: return blend<1>(a, b);
            case 2: return blend<3>(a, b);
            case 3: return blend<7>(a, b);
            case 4: return blend<15>(a, b);
            case 5: return blend<31>(a, b);
            case 6: return blend<63>(a, b);
            case 7: return blend<127>(a, b);
        }
        return b;
    }

    template <typename step_t>
    static Vectorized arange(int32_t base = 0, step_t step = static_cast<step_t>(1)) {
        return Vectorized(_mm256_setr_epi32(
            base,
            base + step,
            base + 2 * step,
            base + 3 * step,
            base + 4 * step,
            base + 5 * step,
            base + 6 * step,
            base + 7 * step
        ));
    }

    // --- Unary ---

    Vectorized neg() const;

    Vectorized abs() const {
        return _mm256_abs_epi32(values_);
    }

    Vectorized sqrt() const {
        alignas(32) int32_t tmp[kSize];
        store(tmp);
        for (int i = 0; i < kSize; i++) {
            tmp[i] = static_cast<int32_t>(std::sqrt(static_cast<double>(tmp[i])));
        }
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
    }

    Vectorized reciprocal() const {
        return Vectorized(0);
    }

    // --- Reductions ---

    int32_t reduce_add() const {
        auto v = values_;
        auto v1 = _mm256_permute2f128_si256(v, v, 0x1);
        v = _mm256_add_epi32(v, v1);
        v1 = _mm256_shuffle_epi32(v, 0x4E);
        v = _mm256_add_epi32(v, v1);
        v1 = _mm256_shuffle_epi32(v, 0xB1);
        v = _mm256_add_epi32(v, v1);
        return _mm_cvtsi128_si32(_mm256_castsi256_si128(v));
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return _mm256_cmpeq_epi32(values_, other.values_);
    }

    Vectorized operator!=(const Vectorized& other) const {
        return invert(_mm256_cmpeq_epi32(values_, other.values_));
    }

    Vectorized operator>(const Vectorized& other) const {
        return _mm256_cmpgt_epi32(values_, other.values_);
    }

    Vectorized operator<(const Vectorized& other) const {
        return _mm256_cmpgt_epi32(other.values_, values_);
    }

    Vectorized operator>=(const Vectorized& other) const {
        return invert(_mm256_cmpgt_epi32(other.values_, values_));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return invert(_mm256_cmpgt_epi32(values_, other.values_));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    Vectorized isnan() const {
        return _mm256_setzero_si256();
    }

    int zero_mask() const {
        return _mm256_movemask_epi8(
            _mm256_cmpeq_epi32(values_, _mm256_setzero_si256())
        );
    }
};

// Free function specializations
// --- Arithmetic ---

template <>
inline Vectorized<int32_t> operator+(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm256_add_epi32(a, b);
}

template <>
inline Vectorized<int32_t> operator-(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm256_sub_epi32(a, b);
}

template <>
inline Vectorized<int32_t> operator*(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm256_mullo_epi32(a, b);
}

// No native SIMD integer division
template <>
inline Vectorized<int32_t> operator/(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    alignas(32) int32_t a_arr[Vectorized<int32_t>::size()];
    alignas(32) int32_t b_arr[Vectorized<int32_t>::size()];
    a.store(a_arr);
    b.store(b_arr);
    for (int i = 0; i < Vectorized<int32_t>::size(); i++) {
        a_arr[i] = a_arr[i] / b_arr[i];
    }
    return Vectorized<int32_t>::loadu(a_arr);
}

inline Vectorized<int32_t> Vectorized<int32_t>::neg() const {
    return Vectorized<int32_t>(0) - *this;
}

// --- Bitwise ---

template <>
inline Vectorized<int32_t> operator&(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm256_and_si256(a, b);
}

template <>
inline Vectorized<int32_t> operator|(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm256_or_si256(a, b);
}

template <>
inline Vectorized<int32_t> operator^(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm256_xor_si256(a, b);
}

inline Vectorized<int32_t> Vectorized<int32_t>::eq(const Vectorized<int32_t>& other) const {
    return (*this == other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::ne(const Vectorized<int32_t>& other) const {
    return (*this != other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::gt(const Vectorized<int32_t>& other) const {
    return (*this > other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::ge(const Vectorized<int32_t>& other) const {
    return (*this >= other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::lt(const Vectorized<int32_t>& other) const {
    return (*this < other) & Vectorized<int32_t>(1);
}

inline Vectorized<int32_t> Vectorized<int32_t>::le(const Vectorized<int32_t>& other) const {
    return (*this <= other) & Vectorized<int32_t>(1);
}

// --- maximum/minimum ---

template <>
inline Vectorized<int32_t> maximum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm256_max_epi32(a, b);
}

template <>
inline Vectorized<int32_t> minimum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm256_min_epi32(a, b);
}


template <>
struct Vectorized<int64_t> {
    using value_type = int64_t;
    using size_type = int;
    static constexpr size_type kSize = 4;

private:
    __m256i values_;

    static inline __m256i invert(const __m256i& v) {
        return _mm256_xor_si256(v, _mm256_set1_epi64x(-1));
    }

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm256_setzero_si256()) {}

    Vectorized(__m256i v) : values_(v) {}

    Vectorized(int64_t val) : values_(_mm256_set1_epi64x(val)) {}

    operator __m256i() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm256_loadu_si256(static_cast<const __m256i*>(ptr));
        }
        alignas(32) int64_t tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(int64_t));
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm256_storeu_si256(static_cast<__m256i*>(ptr), values_);
            return;
        }
        alignas(32) int64_t tmp[kSize];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), values_);
        std::memcpy(ptr, tmp, count * sizeof(int64_t));
    }

    int64_t operator[](int idx) const {
        alignas(32) int64_t tmp[kSize];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), values_);
        return tmp[idx];
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        return _mm256_blend_epi32(a.values_, b.values_,
            ((mask & 0x01) ? 0x03 : 0x00) |
            ((mask & 0x02) ? 0x0C : 0x00) |
            ((mask & 0x04) ? 0x30 : 0x00) |
            ((mask & 0x08) ? 0xC0 : 0x00));
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return _mm256_blendv_epi8(a.values_, b.values_, mask.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        switch (count) {
            case 0: return a;
            case 1: return blend<1>(a, b);
            case 2: return blend<3>(a, b);
            case 3: return blend<7>(a, b);
        }
        return b;
    }

    template <typename step_t>
    static Vectorized arange(int64_t base = 0, step_t step = static_cast<step_t>(1)) {
        return Vectorized(_mm256_setr_epi64x(
            base,
            base + step,
            base + 2 * step,
            base + 3 * step
        ));
    }

    // --- Unary ---

    Vectorized neg() const;

    Vectorized abs() const {
        auto zero = _mm256_setzero_si256();
        auto is_neg = _mm256_cmpgt_epi64(zero, values_);
        auto inv = _mm256_xor_si256(values_, is_neg);
        return _mm256_sub_epi64(inv, is_neg);
    }

    // --- Reductions ---

    int64_t reduce_add() const {
        auto hi = _mm256_permute2f128_si256(values_, values_, 0x1);
        auto sum = _mm256_add_epi64(values_, hi);
        auto shuf = _mm256_shuffle_epi32(sum, 0x4E);
        sum = _mm256_add_epi64(sum, shuf);
        return _mm_cvtsi128_si64(_mm256_castsi256_si128(sum));
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return _mm256_cmpeq_epi64(values_, other.values_);
    }

    Vectorized operator!=(const Vectorized& other) const {
        return invert(_mm256_cmpeq_epi64(values_, other.values_));
    }

    Vectorized operator>(const Vectorized& other) const {
        return _mm256_cmpgt_epi64(values_, other.values_);
    }

    Vectorized operator<(const Vectorized& other) const {
        return _mm256_cmpgt_epi64(other.values_, values_);
    }

    Vectorized operator>=(const Vectorized& other) const {
        return invert(_mm256_cmpgt_epi64(other.values_, values_));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return invert(_mm256_cmpgt_epi64(values_, other.values_));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    Vectorized isnan() const {
        return _mm256_setzero_si256();
    }

    int zero_mask() const {
        return _mm256_movemask_epi8(
            _mm256_cmpeq_epi64(values_, _mm256_setzero_si256())
        );
    }
};

// --- Arithmetic ---

template <>
inline Vectorized<int64_t> operator+(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm256_add_epi64(a, b);
}

template <>
inline Vectorized<int64_t> operator-(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm256_sub_epi64(a, b);
}

// AVX2 has no native 64-bit multiply; extract-and-repack
template <>
inline Vectorized<int64_t> operator*(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    int64_t a0 = _mm256_extract_epi64(a, 0);
    int64_t a1 = _mm256_extract_epi64(a, 1);
    int64_t a2 = _mm256_extract_epi64(a, 2);
    int64_t a3 = _mm256_extract_epi64(a, 3);
    int64_t b0 = _mm256_extract_epi64(b, 0);
    int64_t b1 = _mm256_extract_epi64(b, 1);
    int64_t b2 = _mm256_extract_epi64(b, 2);
    int64_t b3 = _mm256_extract_epi64(b, 3);
    return _mm256_set_epi64x(a3 * b3, a2 * b2, a1 * b1, a0 * b0);
}

template <>
inline Vectorized<int64_t> operator/(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    int64_t a0 = _mm256_extract_epi64(a, 0);
    int64_t a1 = _mm256_extract_epi64(a, 1);
    int64_t a2 = _mm256_extract_epi64(a, 2);
    int64_t a3 = _mm256_extract_epi64(a, 3);
    int64_t b0 = _mm256_extract_epi64(b, 0);
    int64_t b1 = _mm256_extract_epi64(b, 1);
    int64_t b2 = _mm256_extract_epi64(b, 2);
    int64_t b3 = _mm256_extract_epi64(b, 3);
    return _mm256_set_epi64x(a3 / b3, a2 / b2, a1 / b1, a0 / b0);
}

inline Vectorized<int64_t> Vectorized<int64_t>::neg() const {
    return Vectorized<int64_t>(0) - *this;
}

// --- Bitwise ---

template <>
inline Vectorized<int64_t> operator&(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm256_and_si256(a, b);
}

template <>
inline Vectorized<int64_t> operator|(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm256_or_si256(a, b);
}

template <>
inline Vectorized<int64_t> operator^(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm256_xor_si256(a, b);
}

inline Vectorized<int64_t> Vectorized<int64_t>::eq(const Vectorized<int64_t>& other) const {
    return (*this == other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::ne(const Vectorized<int64_t>& other) const {
    return (*this != other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::gt(const Vectorized<int64_t>& other) const {
    return (*this > other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::ge(const Vectorized<int64_t>& other) const {
    return (*this >= other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::lt(const Vectorized<int64_t>& other) const {
    return (*this < other) & Vectorized<int64_t>(1);
}

inline Vectorized<int64_t> Vectorized<int64_t>::le(const Vectorized<int64_t>& other) const {
    return (*this <= other) & Vectorized<int64_t>(1);
}

// --- maximum/minimum ---

template <>
inline Vectorized<int64_t> maximum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    auto cmp = _mm256_cmpgt_epi64(a, b);
    return _mm256_blendv_epi8(b, a, cmp);
}

template <>
inline Vectorized<int64_t> minimum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    auto cmp = _mm256_cmpgt_epi64(a, b);
    return _mm256_blendv_epi8(a, b, cmp);
}


template <>
struct Vectorized<uint8_t> {
    using value_type = uint8_t;
    using size_type = int;
    static constexpr size_type kSize = 32;

private:
    __m256i values_;

    static inline __m256i invert(const __m256i& v) {
        return _mm256_xor_si256(v, _mm256_set1_epi8(-1));
    }

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm256_setzero_si256()) {}

    Vectorized(__m256i v) : values_(v) {}

    Vectorized(uint8_t val) : values_(_mm256_set1_epi8(static_cast<char>(val))) {}

    operator __m256i() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm256_loadu_si256(static_cast<const __m256i*>(ptr));
        }
        alignas(32) uint8_t tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(uint8_t));
        return _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp));
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm256_storeu_si256(static_cast<__m256i*>(ptr), values_);
            return;
        }
        alignas(32) uint8_t tmp[kSize];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), values_);
        std::memcpy(ptr, tmp, count * sizeof(uint8_t));
    }

    uint8_t operator[](int idx) const {
        alignas(32) uint8_t tmp[kSize];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(tmp), values_);
        return tmp[idx];
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        alignas(32) uint8_t a_tmp[kSize], b_tmp[kSize];
        a.store(a_tmp);
        b.store(b_tmp);
        for (int i = 0; i < kSize; i++) {
            if ((mask >> i) & 1) a_tmp[i] = b_tmp[i];
        }
        return loadu(a_tmp);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return _mm256_blendv_epi8(a.values_, b.values_, mask.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        alignas(32) uint8_t a_tmp[kSize], b_tmp[kSize];
        a.store(a_tmp);
        b.store(b_tmp);
        for (int64_t i = 0; i < count; i++) {
            a_tmp[i] = b_tmp[i];
        }
        return loadu(a_tmp);
    }

    template <typename step_t>
    static Vectorized arange(uint8_t base = 0, step_t step = static_cast<step_t>(1)) {
        alignas(32) uint8_t tmp[kSize];
        for (int i = 0; i < kSize; i++) {
            tmp[i] = static_cast<uint8_t>(base + i * step);
        }
        return loadu(tmp);
    }

    // --- Unary ---

    Vectorized neg() const {
        return _mm256_sub_epi8(_mm256_setzero_si256(), values_);
    }

    Vectorized abs() const {
        return *this;
    }

    // --- Comparisons (unsigned) ---

    Vectorized operator==(const Vectorized& other) const {
        return _mm256_cmpeq_epi8(values_, other.values_);
    }

    Vectorized operator!=(const Vectorized& other) const {
        return invert(_mm256_cmpeq_epi8(values_, other.values_));
    }

    // Unsigned less-than: a < b iff max(a,b) != a
    Vectorized operator<(const Vectorized& other) const {
        auto max_val = _mm256_max_epu8(values_, other.values_);
        return invert(_mm256_cmpeq_epi8(max_val, values_));
    }

    // Unsigned less-equal: a <= b iff max(a,b) == b
    Vectorized operator<=(const Vectorized& other) const {
        auto max_val = _mm256_max_epu8(values_, other.values_);
        return _mm256_cmpeq_epi8(max_val, other.values_);
    }

    Vectorized operator>(const Vectorized& other) const {
        return other < *this;
    }

    Vectorized operator>=(const Vectorized& other) const {
        return other <= *this;
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    Vectorized isnan() const {
        return _mm256_setzero_si256();
    }

    int zero_mask() const {
        return _mm256_movemask_epi8(
            _mm256_cmpeq_epi8(values_, _mm256_setzero_si256()));
    }
};

// --- Arithmetic ---

template <>
inline Vectorized<uint8_t> operator+(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm256_add_epi8(a, b);
}

template <>
inline Vectorized<uint8_t> operator-(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm256_sub_epi8(a, b);
}

// AVX2 has no 8-bit multiply; widen to 16-bit, multiply, re-pack
template <>
inline Vectorized<uint8_t> operator*(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    __m256i mask_00ff = _mm256_set1_epi16(0x00FF);
    __m256i a_lo = _mm256_and_si256(a, mask_00ff);
    __m256i b_lo = _mm256_and_si256(b, mask_00ff);
    __m256i a_hi = _mm256_srli_epi16(a, 8);
    __m256i b_hi = _mm256_srli_epi16(b, 8);
    __m256i lo = _mm256_and_si256(_mm256_mullo_epi16(a_lo, b_lo), mask_00ff);
    __m256i hi = _mm256_slli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
    return _mm256_or_si256(lo, hi);
}

template <>
inline Vectorized<uint8_t> operator/(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    alignas(32) uint8_t a_arr[Vectorized<uint8_t>::size()];
    alignas(32) uint8_t b_arr[Vectorized<uint8_t>::size()];
    a.store(a_arr);
    b.store(b_arr);
    for (int i = 0; i < Vectorized<uint8_t>::size(); i++) {
        a_arr[i] = a_arr[i] / b_arr[i];
    }
    return Vectorized<uint8_t>::loadu(a_arr);
}

// --- Bitwise ---

template <>
inline Vectorized<uint8_t> operator&(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm256_and_si256(a, b);
}

template <>
inline Vectorized<uint8_t> operator|(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm256_or_si256(a, b);
}

template <>
inline Vectorized<uint8_t> operator^(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm256_xor_si256(a, b);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::eq(const Vectorized<uint8_t>& other) const {
    return (*this == other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::ne(const Vectorized<uint8_t>& other) const {
    return (*this != other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::gt(const Vectorized<uint8_t>& other) const {
    return (*this > other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::ge(const Vectorized<uint8_t>& other) const {
    return (*this >= other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::lt(const Vectorized<uint8_t>& other) const {
    return (*this < other) & Vectorized<uint8_t>(1);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::le(const Vectorized<uint8_t>& other) const {
    return (*this <= other) & Vectorized<uint8_t>(1);
}

// --- maximum/minimum (unsigned) ---

template <>
inline Vectorized<uint8_t> maximum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm256_max_epu8(a, b);
}

template <>
inline Vectorized<uint8_t> minimum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm256_min_epu8(a, b);
}

} // namespacer CPU_CAPABILITY
} // namespace raxis::vec  