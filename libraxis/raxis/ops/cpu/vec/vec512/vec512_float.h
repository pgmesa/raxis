#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <sleef.h>

#include <raxis/ops/cpu/vec/vec_base.h>


// From: aten/src/ATen/cpu/vec/vec512/vec512_float.h
// AVX512 compares into a mask register (__mmask16, one bit per lane) instead
// of a vector. The comparisons still return all-ones lanes, as the AVX2 and
// generic versions do (kernels and with them or blendv them), built from the
// mask with movm. Masks also give partial loads and stores without a stack
// buffer, and set(a, b, count) as a single blend.

namespace raxis::vec {

inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<float> {
    using value_type = float;
    using size_type = int;
    static constexpr size_type kSize = 16;

private:
    __m512 values_;

    static __m512 mask_to_vec(__mmask16 mask) {
        return _mm512_castsi512_ps(_mm512_movm_epi32(mask));
    }

    static __mmask16 first_lanes(int64_t count) {
        return static_cast<__mmask16>((1u << count) - 1);
    }

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm512_setzero_ps()) {}

    Vectorized(__m512 v) : values_(v) {}

    Vectorized(float val) : values_(_mm512_set1_ps(val)) {}

    operator __m512() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm512_loadu_ps(ptr);
        }
        return _mm512_maskz_loadu_ps(first_lanes(count), ptr);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm512_storeu_ps(ptr, values_);
            return;
        }
        _mm512_mask_storeu_ps(ptr, first_lanes(count), values_);
    }

    float operator[](int idx) const {
        alignas(64) float tmp[kSize];
        _mm512_store_ps(tmp, values_);
        return tmp[idx];
    }

    Vectorized map(float (*const f)(float)) const {
        alignas(64) float tmp[kSize];
        store(tmp);
        for (int i = 0; i < kSize; i++) {
            tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        return _mm512_mask_blend_ps(static_cast<__mmask16>(mask), a.values_, b.values_);
    }

    // Lanes of `mask` with the sign bit set take b, as _mm256_blendv_ps
    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        const __mmask16 m = _mm512_movepi32_mask(_mm512_castps_si512(mask.values_));
        return _mm512_mask_blend_ps(m, a.values_, b.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        if (count >= kSize) {
            return b;
        }
        return _mm512_mask_blend_ps(first_lanes(count), a.values_, b.values_);
    }

    template <typename step_t>
    static Vectorized arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
        return Vectorized(_mm512_setr_ps(
            base,
            base + step,
            base + 2 * step,
            base + 3 * step,
            base + 4 * step,
            base + 5 * step,
            base + 6 * step,
            base + 7 * step,
            base + 8 * step,
            base + 9 * step,
            base + 10 * step,
            base + 11 * step,
            base + 12 * step,
            base + 13 * step,
            base + 14 * step,
            base + 15 * step
        ));
    }

    // --- Unary ---

    Vectorized neg() const {
        return _mm512_xor_ps(values_, _mm512_set1_ps(-0.0f));
    }

    Vectorized abs() const {
        return _mm512_andnot_ps(_mm512_set1_ps(-0.0f), values_);
    }

    Vectorized sqrt() const {
        return _mm512_sqrt_ps(values_);
    }

    Vectorized reciprocal() const {
        return _mm512_div_ps(_mm512_set1_ps(1.0f), values_);
    }

    Vectorized rsqrt() const {
        return _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_sqrt_ps(values_));
    }

    Vectorized ceil() const {
        return _mm512_roundscale_ps(values_, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);
    }

    Vectorized floor() const {
        return _mm512_roundscale_ps(values_, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    }

    Vectorized round() const {
        return _mm512_roundscale_ps(values_, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    }

    Vectorized trunc() const {
        return _mm512_roundscale_ps(values_, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    }

    Vectorized frac() const;

    // --- Transcendentals (Sleef) ---

    Vectorized exp() const {
        return Sleef_expf16_u10(values_);
    }

    // The AVX2 algorithm (see vec256_float.h) on 16 lanes
    Vectorized exp_u20() const {
        const __m512 vec_factorial_1 = _mm512_set1_ps(0.999999701f);  // 1/factorial(1)
        const __m512 vec_factorial_2 = _mm512_set1_ps(0.499991506f);  // 1/factorial(2)
        const __m512 vec_factorial_3 = _mm512_set1_ps(0.166676521f);  // 1/factorial(3)
        const __m512 vec_factorial_4 = _mm512_set1_ps(0.0418978221f);  // 1/factorial(4)
        const __m512 vec_factorial_5 = _mm512_set1_ps(0.00828929059f);  // 1/factorial(5)
        const __m512 vec_exp_log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));  // log2(e)
        const __m512 vec_half = _mm512_set1_ps(0.5f);
        const __m512 vec_one = _mm512_set1_ps(1.f);
        const __m512 vec_zero = _mm512_set1_ps(0.f);
        const __m512 vec_two = _mm512_set1_ps(2.f);
        const __m512 vec_ln2f = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));  // ln(2)
        const __m512 vec_ln_flt_min = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));
        const __m512 vec_ln_flt_max = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));
        const __m512i vec_127 = _mm512_set1_epi32(0x0000007f);
        const int n_mantissa_bits = 23;

        const __mmask16 less_ln_flt_min_mask = _mm512_cmp_ps_mask(values_, vec_ln_flt_min, _CMP_LT_OS);
        __m512 vec_src = _mm512_min_ps(values_, vec_ln_flt_max);
        vec_src = _mm512_max_ps(vec_src, vec_ln_flt_min);

        // fx = floor(x * log2(e) + 0.5), rounding toward -inf in the conversion
        __m512 vec_fx = _mm512_fmadd_ps(vec_src, vec_exp_log2ef, vec_half);
        const __m512i vec_fx_i = _mm512_cvt_roundps_epi32(vec_fx, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
        vec_fx = _mm512_cvtepi32_ps(vec_fx_i);

        // r = x - fx * ln2
        const __m512 vec_exp_poly = _mm512_fnmadd_ps(vec_fx, vec_ln2f, vec_src);

        __m512 vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_factorial_5, vec_factorial_4);
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_3);
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_2);
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_factorial_1);
        vec_res = _mm512_fmadd_ps(vec_exp_poly, vec_res, vec_one);

        // 2^(n - 1) and a final * 2, so that n = 128 does not overflow the exponent
        const __m512 vec_exp_number = _mm512_sub_ps(vec_fx, vec_one);
        const __m512i vec_exp_number_i = _mm512_cvtps_epi32(vec_exp_number);
        __m512i vec_two_pow_n_i = _mm512_add_epi32(vec_exp_number_i, vec_127);
        vec_two_pow_n_i = _mm512_slli_epi32(vec_two_pow_n_i, n_mantissa_bits);
        __m512 vec_two_pow_n = _mm512_castsi512_ps(vec_two_pow_n_i);
        vec_two_pow_n = _mm512_mask_blend_ps(less_ln_flt_min_mask, vec_two_pow_n, vec_zero);

        vec_res = _mm512_mul_ps(vec_res, vec_two_pow_n);
        return _mm512_mul_ps(vec_res, vec_two);
    }

    Vectorized expm1() const {
        return Sleef_expm1f16_u10(values_);
    }

    Vectorized log() const {
        return Sleef_logf16_u10(values_);
    }

    Vectorized log2() const {
        return Sleef_log2f16_u10(values_);
    }

    Vectorized log10() const {
        return Sleef_log10f16_u10(values_);
    }

    Vectorized log1p() const {
        return Sleef_log1pf16_u10(values_);
    }

    Vectorized sin() const {
        return Sleef_sinf16_u10(values_);
    }

    Vectorized cos() const {
        return Sleef_cosf16_u10(values_);
    }

    Vectorized tan() const {
        return Sleef_tanf16_u10(values_);
    }

    Vectorized asin() const {
        return Sleef_asinf16_u10(values_);
    }

    Vectorized acos() const {
        return Sleef_acosf16_u10(values_);
    }

    Vectorized atan() const {
        return Sleef_atanf16_u10(values_);
    }

    Vectorized sinh() const {
        return Sleef_sinhf16_u10(values_);
    }

    Vectorized cosh() const {
        return Sleef_coshf16_u10(values_);
    }

    Vectorized tanh() const {
        return Sleef_tanhf16_u10(values_);
    }

    // Abramowitz and Stegun 7.1.26, as the AVX2 version (see vec256_float.h)
    Vectorized erf() const {
        const __m512 neg_zero_vec = _mm512_set1_ps(-0.f);
        const __m512 one_vec = _mm512_set1_ps(1.0f);
        const __m512 p = _mm512_set1_ps(0.3275911f);
        const __m512 p1 = _mm512_set1_ps(0.254829592f);
        const __m512 p2 = _mm512_set1_ps(-0.284496736f);
        const __m512 p3 = _mm512_set1_ps(1.421413741f);
        const __m512 p4 = _mm512_set1_ps(-1.453152027f);
        const __m512 p5 = _mm512_set1_ps(1.061405429f);
        const __m512 sign_mask = _mm512_and_ps(neg_zero_vec, values_);
        const __m512 abs_vec = _mm512_xor_ps(sign_mask, values_);
        // t = 1 / (p * |x| + 1)
        const __m512 t = _mm512_div_ps(one_vec, _mm512_fmadd_ps(p, abs_vec, one_vec));
        // r = p5 t^4 + p4 t^3 + p3 t^2 + p2 t + p1, by Horner
        __m512 r = _mm512_fmadd_ps(p5, t, p4);
        r = _mm512_fmadd_ps(r, t, p3);
        r = _mm512_fmadd_ps(r, t, p2);
        r = _mm512_fmadd_ps(r, t, p1);
        // 1 - r * t * exp(-x^2)
        const __m512 neg_exp = _mm512_xor_ps(neg_zero_vec, Sleef_expf16_u10(_mm512_xor_ps(neg_zero_vec, _mm512_mul_ps(values_, values_))));
        const __m512 result = _mm512_fmadd_ps(_mm512_mul_ps(neg_exp, t), r, one_vec);
        return _mm512_xor_ps(sign_mask, result);
    }

    Vectorized erfc() const {
        return Sleef_erfcf16_u15(values_);
    }

    Vectorized pow(const Vectorized& exponent) const {
        return Sleef_powf16_u10(values_, exponent);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_ps_mask(values_, other.values_, _CMP_EQ_OQ));
    }

    Vectorized operator!=(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_ps_mask(values_, other.values_, _CMP_NEQ_UQ));
    }

    Vectorized operator<(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_ps_mask(values_, other.values_, _CMP_LT_OQ));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_ps_mask(values_, other.values_, _CMP_LE_OQ));
    }

    Vectorized operator>(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_ps_mask(values_, other.values_, _CMP_GT_OQ));
    }

    Vectorized operator>=(const Vectorized& other) const {
        return mask_to_vec(_mm512_cmp_ps_mask(values_, other.values_, _CMP_GE_OQ));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    // --- NaN / mask ---

    Vectorized isnan() const {
        return mask_to_vec(_mm512_cmp_ps_mask(values_, values_, _CMP_UNORD_Q));
    }

    int zero_mask() const {
        return _mm512_cmp_ps_mask(values_, _mm512_setzero_ps(), _CMP_EQ_OQ);
    }
};

// Free function specializations
// --- Arithmetic ---

template <>
inline Vectorized<float> operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm512_add_ps(a, b);
}

template <>
inline Vectorized<float> operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm512_sub_ps(a, b);
}

template <>
inline Vectorized<float> operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm512_mul_ps(a, b);
}

template <>
inline Vectorized<float> operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm512_div_ps(a, b);
}

inline Vectorized<float> Vectorized<float>::frac() const {
    return *this - this->trunc();
}

// --- Bitwise ---

template <>
inline Vectorized<float> operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm512_and_ps(a, b);
}

template <>
inline Vectorized<float> operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm512_or_ps(a, b);
}

template <>
inline Vectorized<float> operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
    return _mm512_xor_ps(a, b);
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

// max_ps returns the second operand when either is NaN; the NaN lanes are
// then or-ed with all-ones, which is a NaN
template <>
inline Vectorized<float> maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
    const __m512 max_val = _mm512_max_ps(a, b);
    const __mmask16 isnan = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
    return _mm512_or_ps(max_val, _mm512_castsi512_ps(_mm512_movm_epi32(isnan)));
}

template <>
inline Vectorized<float> minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
    const __m512 min_val = _mm512_min_ps(a, b);
    const __mmask16 isnan = _mm512_cmp_ps_mask(a, b, _CMP_UNORD_Q);
    return _mm512_or_ps(min_val, _mm512_castsi512_ps(_mm512_movm_epi32(isnan)));
}

template <>
inline Vectorized<float> clamp(
    const Vectorized<float>& a,
    const Vectorized<float>& min_val,
    const Vectorized<float>& max_val
) {
    return _mm512_min_ps(max_val, _mm512_max_ps(min_val, a));
}

template <>
inline Vectorized<float> clamp_min(const Vectorized<float>& a, const Vectorized<float>& min_val) {
    return _mm512_max_ps(min_val, a);
}

template <>
inline Vectorized<float> clamp_max(const Vectorized<float>& a, const Vectorized<float>& max_val) {
    return _mm512_min_ps(max_val, a);
}

// --- FMA ---

template <>
inline Vectorized<float> fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c
) {
    return _mm512_fmadd_ps(a, b, c);
}

template <>
inline Vectorized<float> fnmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c
) {
    return _mm512_fnmadd_ps(a, b, c);
}

} // namespace CPU_CAPABILITY
} // namespace raxis::vec
