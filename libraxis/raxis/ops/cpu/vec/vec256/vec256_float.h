#pragma once

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <raxis/ops/cpu/vec/vec_base.h>


namespace raxis::vec {

inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<float> {
    using value_type = float;
    using size_type = int;
    static constexpr size_type kSize = 8;

private:
    __m256 values_;

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm256_setzero_ps()) {}

    Vectorized(__m256 v) : values_(v) {}

    Vectorized(float val) : values_(_mm256_set1_ps(val)) {}

    operator __m256() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm256_loadu_ps(static_cast<const float*>(ptr));
        }
        alignas(32) float tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(float));
        return _mm256_load_ps(tmp);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm256_storeu_ps(static_cast<float*>(ptr), values_);
            return;
        }
        alignas(32) float tmp[kSize];
        _mm256_storeu_ps(tmp, values_);
        std::memcpy(ptr, tmp, count * sizeof(float));
    }

    float operator[](int idx) const {
        alignas(32) float tmp[kSize];
        _mm256_storeu_ps(tmp, values_);
        return tmp[idx];
    }

    Vectorized map(float (*const f)(float)) const {
        alignas(32) float tmp[kSize];
        store(tmp);
        for (int i = 0; i < kSize; i++) {
            tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        return _mm256_blend_ps(a.values_, b.values_, mask);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return _mm256_blendv_ps(a.values_, b.values_, mask.values_);
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
    static Vectorized arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
        return Vectorized(_mm256_setr_ps(
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

    Vectorized neg() const {
        return _mm256_xor_ps(values_, _mm256_set1_ps(-0.0f));
    }

    Vectorized abs() const {
        auto mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        return _mm256_and_ps(values_, mask);
    }

    Vectorized sqrt() const {
        return _mm256_sqrt_ps(values_);
    }

    Vectorized reciprocal() const {
        return _mm256_div_ps(_mm256_set1_ps(1.0f), values_);
    }

    Vectorized rsqrt() const {
        return _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(values_));
    }

    Vectorized ceil() const {
        return _mm256_ceil_ps(values_);
    }

    Vectorized floor() const {
        return _mm256_floor_ps(values_);
    }

    Vectorized round() const {
        return _mm256_round_ps(values_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    Vectorized trunc() const {
        return _mm256_round_ps(values_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    Vectorized frac() const;

    Vectorized exp() const { return map(std::exp); }
    Vectorized log() const { return map(std::log); }
    Vectorized log2() const { return map(std::log2); }

    Vectorized pow(const Vectorized& exponent) const {
        alignas(32) float base_arr[kSize];
        alignas(32) float exp_arr[kSize];
        store(base_arr);
        exponent.store(exp_arr);
        for (int i = 0; i < kSize; i++) {
            base_arr[i] = std::pow(base_arr[i], exp_arr[i]);
        }
        return _mm256_load_ps(base_arr);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return _mm256_cmp_ps(values_, other.values_, _CMP_EQ_OQ);
    }

    Vectorized operator!=(const Vectorized& other) const {
        return _mm256_cmp_ps(values_, other.values_, _CMP_NEQ_UQ);
    }

    Vectorized operator<(const Vectorized& other) const {
        return _mm256_cmp_ps(values_, other.values_, _CMP_LT_OQ);
    }

    Vectorized operator<=(const Vectorized& other) const {
        return _mm256_cmp_ps(values_, other.values_, _CMP_LE_OQ);
    }

    Vectorized operator>(const Vectorized& other) const {
        return _mm256_cmp_ps(values_, other.values_, _CMP_GT_OQ);
    }

    Vectorized operator>=(const Vectorized& other) const {
        return _mm256_cmp_ps(values_, other.values_, _CMP_GE_OQ);
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    // --- NaN / mask ---

    Vectorized isnan() const {
        return _mm256_cmp_ps(values_, values_, _CMP_UNORD_Q);
    }

    int zero_mask() const {
        return _mm256_movemask_ps(
            _mm256_cmp_ps(values_, _mm256_set1_ps(0.0f), _CMP_EQ_OQ)
        );
    }
};

// Free function specializations
// --- Arithmetic ---

template <>
inline Vectorized<float> operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_add_ps(a, b);
}

template <>
inline Vectorized<float> operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_sub_ps(a, b);
}

template <>
inline Vectorized<float> operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_mul_ps(a, b);
}

template <>
inline Vectorized<float> operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_div_ps(a, b);
}

inline Vectorized<float> Vectorized<float>::frac() const {
    return *this - this->trunc();
}

// --- Bitwise ---

template <>
inline Vectorized<float> operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_and_ps(a, b);
}

template <>
inline Vectorized<float> operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_or_ps(a, b);
}

template <>
inline Vectorized<float> operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm256_xor_ps(a, b);
}

inline Vectorized<float> Vectorized<float>::eq(const Vectorized<float>& other) const {
    return (*this == other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ne(const Vectorized<float>& other) const {
    return (*this != other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::lt(const Vectorized<float>& other) const {
    return (*this < other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::le(const Vectorized<float>& other) const {
    return (*this <= other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::gt(const Vectorized<float>& other) const {
    return (*this > other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ge(const Vectorized<float>& other) const {
    return (*this >= other) & Vectorized<float>(1.0f);
}

// --- maximum/minimum ---

template <>
inline Vectorized<float> maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
    Vectorized<float> max_val = _mm256_max_ps(a, b);
    Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
    return _mm256_or_ps(max_val, isnan);
}

template <>
inline Vectorized<float> minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
    Vectorized<float> min_val = _mm256_min_ps(a, b);
    Vectorized<float> isnan = _mm256_cmp_ps(a, b, _CMP_UNORD_Q);
    return _mm256_or_ps(min_val, isnan);
}

template <>
inline Vectorized<float> clamp(
    const Vectorized<float>& a,
    const Vectorized<float>& min_val,
    const Vectorized<float>& max_val
) {
    return _mm256_min_ps(max_val, _mm256_max_ps(min_val, a));
}

template <>
inline Vectorized<float> clamp_min(const Vectorized<float>& a, const Vectorized<float>& min_val) {
    return _mm256_max_ps(min_val, a);
}

template <>
inline Vectorized<float> clamp_max(const Vectorized<float>& a, const Vectorized<float>& max_val) {
    return _mm256_min_ps(max_val, a);
}

// --- FMA ---

template <>
inline Vectorized<float> fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c
) {
    return _mm256_fmadd_ps(a, b, c);
}

template <>
inline Vectorized<float> fnmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c
) {
    return _mm256_fnmadd_ps(a, b, c);
}

} // namespace CPU_CAPABILITY
} // namespace raxis::vec