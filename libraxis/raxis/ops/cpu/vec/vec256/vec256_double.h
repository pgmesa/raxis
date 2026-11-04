#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <sleef.h>

#include <raxis/ops/cpu/vec/vec_base.h>


namespace raxis::vec {

inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<double> {
    using value_type = double;
    using size_type = int;
    static constexpr size_type kSize = 4;

private:
    __m256d values_;

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm256_setzero_pd()) {}

    Vectorized(__m256d v) : values_(v) {}

    Vectorized(double val) : values_(_mm256_set1_pd(val)) {}

    operator __m256d() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm256_loadu_pd(static_cast<const double*>(ptr));
        }
        alignas(32) double tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(double));
        return _mm256_load_pd(tmp);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm256_storeu_pd(static_cast<double*>(ptr), values_);
            return;
        }
        alignas(32) double tmp[kSize];
        _mm256_storeu_pd(tmp, values_);
        std::memcpy(ptr, tmp, count * sizeof(double));
    }

    double operator[](int idx) const {
        alignas(32) double tmp[kSize];
        _mm256_storeu_pd(tmp, values_);
        return tmp[idx];
    }

    Vectorized map(double (*const f)(double)) const {
        alignas(32) double tmp[kSize];
        store(tmp);
        for (int i = 0; i < kSize; i++) {
            tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        return _mm256_blend_pd(a.values_, b.values_, mask);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return _mm256_blendv_pd(a.values_, b.values_, mask.values_);
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
    static Vectorized arange(double base = 0., step_t step = static_cast<step_t>(1)) {
        return Vectorized(_mm256_setr_pd(
            base, base + step, base + 2 * step, base + 3 * step
        ));
    }

    // --- Unary ---

    Vectorized neg() const {
        return _mm256_xor_pd(values_, _mm256_set1_pd(-0.0));
    }

    Vectorized abs() const {
        auto mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        return _mm256_and_pd(values_, mask);
    }

    Vectorized sqrt() const {
        return _mm256_sqrt_pd(values_);
    }

    Vectorized reciprocal() const {
        return _mm256_div_pd(_mm256_set1_pd(1.0), values_);
    }

    Vectorized rsqrt() const {
        return _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(values_));
    }

    Vectorized ceil() const {
        return _mm256_ceil_pd(values_);
    }

    Vectorized floor() const {
        return _mm256_floor_pd(values_);
    }

    Vectorized round() const {
        return _mm256_round_pd(values_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    Vectorized trunc() const {
        return _mm256_round_pd(values_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    Vectorized frac() const;

    // --- Transcendentals (Sleef) ---

    Vectorized exp() const {
        return Sleef_expd4_u10(values_);
    }

    Vectorized exp_u20() const {
        return exp();
    }

    Vectorized expm1() const {
        return Sleef_expm1d4_u10(values_);
    }

    Vectorized log() const {
        return Sleef_logd4_u10(values_);
    }

    Vectorized log2() const {
        return Sleef_log2d4_u10(values_);
    }

    Vectorized log10() const {
        return Sleef_log10d4_u10(values_);
    }

    Vectorized log1p() const {
        return Sleef_log1pd4_u10(values_);
    }

    Vectorized sin() const {
        return Sleef_sind4_u10(values_);
    }

    Vectorized cos() const {
        return Sleef_cosd4_u10(values_);
    }

    Vectorized tan() const {
        return Sleef_tand4_u10(values_);
    }

    Vectorized asin() const {
        return Sleef_asind4_u10(values_);
    }

    Vectorized acos() const {
        return Sleef_acosd4_u10(values_);
    }

    Vectorized atan() const {
        return Sleef_atand4_u10(values_);
    }

    Vectorized sinh() const {
        return Sleef_sinhd4_u10(values_);
    }

    Vectorized cosh() const {
        return Sleef_coshd4_u10(values_);
    }

    Vectorized tanh() const {
        return Sleef_tanhd4_u10(values_);
    }

    Vectorized erf() const {
        return Sleef_erfd4_u10(values_);
    }

    Vectorized erfc() const {
        return Sleef_erfcd4_u15(values_);
    }

    Vectorized pow(const Vectorized& exponent) const {
        return Sleef_powd4_u10(values_, exponent);
    }

    // Exact, as std::fmod
    Vectorized fmod(const Vectorized& divisor) const {
        return Sleef_fmodd4(values_, divisor);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return _mm256_cmp_pd(values_, other.values_, _CMP_EQ_OQ);
    }

    Vectorized operator!=(const Vectorized& other) const {
        return _mm256_cmp_pd(values_, other.values_, _CMP_NEQ_UQ);
    }

    Vectorized operator<(const Vectorized& other) const {
        return _mm256_cmp_pd(values_, other.values_, _CMP_LT_OQ);
    }

    Vectorized operator<=(const Vectorized& other) const {
        return _mm256_cmp_pd(values_, other.values_, _CMP_LE_OQ);
    }

    Vectorized operator>(const Vectorized& other) const {
        return _mm256_cmp_pd(values_, other.values_, _CMP_GT_OQ);
    }

    Vectorized operator>=(const Vectorized& other) const {
        return _mm256_cmp_pd(values_, other.values_, _CMP_GE_OQ);
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    // --- NaN / mask ---

    Vectorized isnan() const {
        return _mm256_cmp_pd(values_, values_, _CMP_UNORD_Q);
    }

    int zero_mask() const {
        return _mm256_movemask_pd(
            _mm256_cmp_pd(values_, _mm256_set1_pd(0.0), _CMP_EQ_OQ)
        );
    }
};

// Free function specializations
// --- Arithmetic ---

template <>
inline Vectorized<double> operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_add_pd(a, b);
}

template <>
inline Vectorized<double> operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_sub_pd(a, b);
}

template <>
inline Vectorized<double> operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_mul_pd(a, b);
}

template <>
inline Vectorized<double> operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_div_pd(a, b);
}

inline Vectorized<double> Vectorized<double>::frac() const {
    return *this - this->trunc();
}

// --- Bitwise ---

template <>
inline Vectorized<double> operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_and_pd(a, b);
}

template <>
inline Vectorized<double> operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_or_pd(a, b);
}

template <>
inline Vectorized<double> operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm256_xor_pd(a, b);
}

inline Vectorized<double> Vectorized<double>::eq(const Vectorized<double>& other) const {
    return (*this == other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ne(const Vectorized<double>& other) const {
    return (*this != other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::gt(const Vectorized<double>& other) const {
    return (*this > other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::ge(const Vectorized<double>& other) const {
    return (*this >= other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::lt(const Vectorized<double>& other) const {
    return (*this < other) & Vectorized<double>(1.0);
}

inline Vectorized<double> Vectorized<double>::le(const Vectorized<double>& other) const {
    return (*this <= other) & Vectorized<double>(1.0);
}

// --- maximum/minimum ---

template <>
inline Vectorized<double> maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
    Vectorized<double> max_val = _mm256_max_pd(a, b);
    Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
    return _mm256_or_pd(max_val, isnan);
}

template <>
inline Vectorized<double> minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
    Vectorized<double> min_val = _mm256_min_pd(a, b);
    Vectorized<double> isnan = _mm256_cmp_pd(a, b, _CMP_UNORD_Q);
    return _mm256_or_pd(min_val, isnan);
}

template <>
inline Vectorized<double> clamp(
    const Vectorized<double>& a,
    const Vectorized<double>& min_val,
    const Vectorized<double>& max_val
) {
    return _mm256_min_pd(max_val, _mm256_max_pd(min_val, a));
}

template <>
inline Vectorized<double> clamp_min(const Vectorized<double>& a, const Vectorized<double>& min_val) {
    return _mm256_max_pd(min_val, a);
}

template <>
inline Vectorized<double> clamp_max(const Vectorized<double>& a, const Vectorized<double>& max_val) {
    return _mm256_min_pd(max_val, a);
}

// --- FMA ---

template <>
inline Vectorized<double> fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c
) {
    return _mm256_fmadd_pd(a, b, c);
}

template <>
inline Vectorized<double> fnmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c
) {
    return _mm256_fnmadd_pd(a, b, c);
}

} // namespace CPU_CAPABILITY
} // namespace raxis::vec