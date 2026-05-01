#pragma once

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <raxis/ops/cpu/vec/vec_base.h>


namespace raxis::vec {

inline namespace CPU_CAPABILITY {

// AVX2 int32_t specialization: 8 x int32 in a 256-bit register.
// int64_t, uint8_t, uint16_t use the generic fallback from vec_base.h
// because AVX2 lacks native 64-bit multiply and the element counts
// for 8/16-bit types require more complex widening logic.

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

} // namespacer CPU_CAPABILITY
} // namespace raxis::vec  