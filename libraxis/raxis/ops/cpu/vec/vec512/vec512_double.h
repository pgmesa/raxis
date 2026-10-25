#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <sleef.h>

#include <raxis/ops/cpu/vec/vec_base.h>


// From: aten/src/ATen/cpu/vec/vec512/vec512_double.h
// Same structure as vec512_float.h: comparisons go through a __mmask8.

namespace raxis::vec {

inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<double> {
    using value_type = double;
    using size_type = int;
    static constexpr size_type kSize = 8;

private:
    __m512d values_;

    static __m512d mask_to_vec(__mmask8 mask) {
        return _mm512_castsi512_pd(_mm512_movm_epi64(mask));
    }

    static __mmask8 first_lanes(int64_t count) {
        return static_cast<__mmask8>((1u << count) - 1);
    }

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm512_setzero_pd()) {}

    Vectorized(__m512d v) : values_(v) {}

    Vectorized(double val) : values_(_mm512_set1_pd(val)) {}

    operator __m512d() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm512_loadu_pd(ptr);
        }
        return _mm512_maskz_loadu_pd(first_lanes(count), ptr);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm512_storeu_pd(ptr, values_);
            return;
        }
        _mm512_mask_storeu_pd(ptr, first_lanes(count), values_);
    }

    double operator[](int idx) const {
        alignas(64) double tmp[kSize];
        _mm512_store_pd(tmp, values_);
        return tmp[idx];
    }

    Vectorized map(double (*const f)(double)) const {
        alignas(64) double tmp[kSize];
        store(tmp);
        for (int i = 0; i < kSize; i++) {
            tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        return _mm512_mask_blend_pd(static_cast<__mmask8>(mask), a.values_, b.values_);
    }

    // Lanes of `mask` with the sign bit set take b, as _mm256_blendv_pd
    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        const __mmask8 m = _mm512_movepi64_mask(_mm512_castpd_si512(mask.values_));
        return _mm512_mask_blend_pd(m, a.values_, b.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        if (count >= kSize) {
            return b;
        }
        return _mm512_mask_blend_pd(first_lanes(count), a.values_, b.values_);
    }

    template <typename step_t>
    static Vectorized arange(double base = 0., step_t step = static_cast<step_t>(1)) {
        return Vectorized(_mm512_setr_pd(
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
        return _mm512_xor_pd(values_, _mm512_set1_pd(-0.0));
    }

    Vectorized abs() const {
        return _mm512_andnot_pd(_mm512_set1_pd(-0.0), values_);
    }

    Vectorized sqrt() const {
        return _mm512_sqrt_pd(values_);
    }

    Vectorized reciprocal() const {
        return _mm512_div_pd(_mm512_set1_pd(1.0), values_);
    }

    Vectorized rsqrt() const {
        return _mm512_div_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(values_));
    }

    Vectorized ceil() const {
        return _mm512_roundscale_pd(values_, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    }

    Vectorized floor() const {
        return _mm512_roundscale_pd(values_, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    }

    Vectorized round() const {
        return _mm512_roundscale_pd(values_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    Vectorized trunc() const {
        return _mm512_roundscale_pd(values_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    Vectorized frac() const;

    // --- Transcendentals (Sleef) ---

    Vectorized exp() const {
        return Sleef_expd8_u10(values_);
    }

    Vectorized exp_u20() const {
        return exp();
    }

    Vectorized expm1() const {
        return Sleef_expm1d8_u10(values_);
    }

    Vectorized log() const {
        return Sleef_logd8_u10(values_);
    }

    Vectorized log2() const {
        return Sleef_log2d8_u10(values_);
    }

    Vectorized log10() const {
        return Sleef_log10d8_u10(values_);
    }

    Vectorized log1p() const {
        return Sleef_log1pd8_u10(values_);
    }

    Vectorized sin() const {
        return Sleef_sind8_u10(values_);
    }

    Vectorized cos() const {
        return Sleef_cosd8_u10(values_);
    }

    Vectorized tan() const {
        return Sleef_tand8_u10(values_);
    }

    Vectorized asin() const {
        return Sleef_asind8_u10(values_);
    }

    Vectorized acos() const {
        return Sleef_acosd8_u10(values_);
    }

    Vectorized atan() const {
        return Sleef_atand8_u10(values_);
    }

    Vectorized sinh() const {
        return Sleef_sinhd8_u10(values_);
    }

    Vectorized cosh() const {
        return Sleef_coshd8_u10(values_);
    }

    Vectorized tanh() const {
        return Sleef_tanhd8_u10(values_);
    }

    Vectorized erf() const {
        return Sleef_erfd8_u10(values_);
    }

    Vectorized erfc() const {
        return Sleef_erfcd8_u15(values_);
    }

    Vectorized pow(const Vectorized& exponent) const {
        return Sleef_powd8_u10(values_, exponent);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_pd_mask(values_, other.values_, _CMP_EQ_OQ));
    }

    Vectorized operator!=(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_pd_mask(values_, other.values_, _CMP_NEQ_UQ));
    }

    Vectorized operator<(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_pd_mask(values_, other.values_, _CMP_LT_OQ));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_pd_mask(values_, other.values_, _CMP_LE_OQ));
    }

    Vectorized operator>(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_pd_mask(values_, other.values_, _CMP_GT_OQ));
    }

    Vectorized operator>=(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_pd_mask(values_, other.values_, _CMP_GE_OQ));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    // --- NaN / mask ---

    Vectorized isnan() const {
        return mask_to_vec(_mm512_cmp_pd_mask(values_, values_, _CMP_UNORD_Q));
    }

    int zero_mask() const {
        return _mm512_cmp_pd_mask(values_, _mm512_setzero_pd(), _CMP_EQ_OQ);
    }
};

// Free function specializations
// --- Arithmetic ---

template <>
inline Vectorized<double> operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm512_add_pd(a, b);
}

template <>
inline Vectorized<double> operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm512_sub_pd(a, b);
}

template <>
inline Vectorized<double> operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm512_mul_pd(a, b);
}

template <>
inline Vectorized<double> operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm512_div_pd(a, b);
}

inline Vectorized<double> Vectorized<double>::frac() const {
    return *this - this->trunc();
}

// --- Bitwise ---

template <>
inline Vectorized<double> operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm512_and_pd(a, b);
}

template <>
inline Vectorized<double> operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm512_or_pd(a, b);
}

template <>
inline Vectorized<double> operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
    return _mm512_xor_pd(a, b);
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
    const __m512d max_val = _mm512_max_pd(a, b);
    const __mmask8 isnan = _mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q);
    return _mm512_or_pd(max_val, _mm512_castsi512_pd(_mm512_movm_epi64(isnan)));
}

template <>
inline Vectorized<double> minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
    const __m512d min_val = _mm512_min_pd(a, b);
    const __mmask8 isnan = _mm512_cmp_pd_mask(a, b, _CMP_UNORD_Q);
    return _mm512_or_pd(min_val, _mm512_castsi512_pd(_mm512_movm_epi64(isnan)));
}

template <>
inline Vectorized<double> clamp(
    const Vectorized<double>& a,
    const Vectorized<double>& min_val,
    const Vectorized<double>& max_val
) {
    return _mm512_min_pd(max_val, _mm512_max_pd(min_val, a));
}

template <>
inline Vectorized<double> clamp_min(const Vectorized<double>& a, const Vectorized<double>& min_val) {
    return _mm512_max_pd(min_val, a);
}

template <>
inline Vectorized<double> clamp_max(const Vectorized<double>& a, const Vectorized<double>& max_val) {
    return _mm512_min_pd(max_val, a);
}

// --- FMA ---

template <>
inline Vectorized<double> fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c
) {
    return _mm512_fmadd_pd(a, b, c);
}

template <>
inline Vectorized<double> fnmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c
) {
    return _mm512_fnmadd_pd(a, b, c);
}

} // namespace CPU_CAPABILITY
} // namespace raxis::vec
