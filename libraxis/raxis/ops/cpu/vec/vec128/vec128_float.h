#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!

#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <sleef.h>

#include <raxis/ops/cpu/vec/vec_base.h>


namespace raxis::vec {

inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<float> {
    using value_type = float;
    using size_type = int;
    static constexpr size_type kSize = 4;

private:
    float32x4_t values_;

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(vdupq_n_f32(0)) {}

    Vectorized(float32x4_t v) : values_(v) {}

    Vectorized(float val) : values_(vdupq_n_f32(val)) {}

    operator float32x4_t() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return vld1q_f32(static_cast<const float*>(ptr));
        }
        alignas(16) float tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(float));
        return vld1q_f32(tmp);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            vst1q_f32(static_cast<float*>(ptr), values_);
            return;
        }
        alignas(16) float tmp[kSize];
        vst1q_f32(tmp, values_);
        std::memcpy(ptr, tmp, count * sizeof(float));
    }

    float operator[](int idx) const {
        alignas(16) float tmp[kSize];
        vst1q_f32(tmp, values_);
        return tmp[idx];
    }

    Vectorized map(float (*const f)(float)) const {
        alignas(16) float tmp[kSize];
        store(tmp);
        for (int i = 0; i < kSize; i++) {
            tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
    }

    float reduce(float (*const f)(float, float)) const {
        alignas(16) float tmp[kSize];
        store(tmp);
        float acc = tmp[0];
        for (int i = 1; i < kSize; i++) {
            acc = f(acc, tmp[i]);
        }
        return acc;
    }

    float reduce_add() const {
        return vaddvq_f32(values_);
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        alignas(16) float a_tmp[kSize], b_tmp[kSize], result[kSize];
        vst1q_f32(a_tmp, a.values_);
        vst1q_f32(b_tmp, b.values_);
        result[0] = (mask & 0x01) ? b_tmp[0] : a_tmp[0];
        result[1] = (mask & 0x02) ? b_tmp[1] : a_tmp[1];
        result[2] = (mask & 0x04) ? b_tmp[2] : a_tmp[2];
        result[3] = (mask & 0x08) ? b_tmp[3] : a_tmp[3];
        return vld1q_f32(result);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return vbslq_f32(
            vreinterpretq_u32_f32(mask.values_), b.values_, a.values_);
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
    static Vectorized arange(float base = 0.f, step_t step = static_cast<step_t>(1)) {
        alignas(16) float tmp[kSize] = {
            base,
            static_cast<float>(base + step),
            static_cast<float>(base + 2 * step),
            static_cast<float>(base + 3 * step)
        };
        return vld1q_f32(tmp);
    }

    // --- Unary ---

    Vectorized neg() const {
        return vnegq_f32(values_);
    }

    Vectorized abs() const {
        return vabsq_f32(values_);
    }

    Vectorized sqrt() const {
        return vsqrtq_f32(values_);
    }

    Vectorized reciprocal() const {
        return vdivq_f32(vdupq_n_f32(1.0f), values_);
    }

    Vectorized rsqrt() const {
        return vdivq_f32(vdupq_n_f32(1.0f), vsqrtq_f32(values_));
    }

    Vectorized ceil() const {
        return vrndpq_f32(values_);
    }

    Vectorized floor() const {
        return vrndmq_f32(values_);
    }

    Vectorized round() const {
        return vrndnq_f32(values_);
    }

    Vectorized trunc() const {
        return vrndq_f32(values_);
    }

    Vectorized frac() const;

    // --- Transcendentals (Sleef) ---

    Vectorized exp() const {
        return Sleef_expf4_u10(values_);
    }

    // From: aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h (vexpq_f32_u20)
    // exp(x) = 2^n (1 + poly(r)) with x = n ln2 + r, r in [-ln2/2, ln2/2]:
    // ln2 split in a high and a low part so that r keeps its precision, and
    // a degree 5 polynomial evaluated as two halves (Estrin) to shorten the
    // dependency chain. 2^n is built in the exponent bits. Inputs beyond
    // |x| > 87.3, where 2^n would overflow (also -inf, which flash attention
    // relies on giving 0), take Sleef's exp for the whole vector
    Vectorized exp_u20() const {
        const float32x4_t special_bound = vdupq_n_f32(0x1.5d5e2ap+6f);
        const uint32x4_t special = vcagtq_f32(values_, special_bound);
        if (vmaxvq_u32(special) != 0) {
            return exp();
        }

        const float32x4_t inv_ln2 = vdupq_n_f32(0x1.715476p+0f);
        constexpr float ln2_hi = 0x1.62e4p-1f;
        constexpr float ln2_lo = 0x1.7f7d1cp-20f;
        constexpr float c0 = 0x1.0e4020p-7f;
        constexpr float c2 = 0x1.555e66p-3f;
        const float32x4_t ln2_c02 = {ln2_hi, ln2_lo, c0, c2};
        const uint32x4_t exponent_bias = vdupq_n_u32(0x3f800000);
        const float32x4_t c1 = vdupq_n_f32(0x1.573e2ep-5f);
        const float32x4_t c3 = vdupq_n_f32(0x1.fffdb6p-2f);
        const float32x4_t c4 = vdupq_n_f32(0x1.ffffecp-1f);

        const float32x4_t n = vrndaq_f32(vmulq_f32(values_, inv_ln2));
        float32x4_t r = vfmsq_laneq_f32(values_, n, ln2_c02, 0);
        r = vfmsq_laneq_f32(r, n, ln2_c02, 1);
        const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_s32(vcvtq_s32_f32(n)), 23);
        const float32x4_t scale = vreinterpretq_f32_u32(vaddq_u32(e, exponent_bias));

        const float32x4_t r2 = vmulq_f32(r, r);
        float32x4_t p = vfmaq_laneq_f32(c1, r, ln2_c02, 2);
        float32x4_t q = vfmaq_laneq_f32(c3, r, ln2_c02, 3);
        q = vfmaq_f32(q, p, r2);
        p = vmulq_f32(c4, r);
        const float32x4_t poly = vfmaq_f32(p, q, r2);
        return vfmaq_f32(scale, poly, scale);
    }

    Vectorized expm1() const {
        return Sleef_expm1f4_u10(values_);
    }

    Vectorized log() const {
        return Sleef_logf4_u10(values_);
    }

    Vectorized log2() const {
        return Sleef_log2f4_u10(values_);
    }

    Vectorized log10() const {
        return Sleef_log10f4_u10(values_);
    }

    Vectorized log1p() const {
        return Sleef_log1pf4_u10(values_);
    }

    Vectorized sin() const {
        return Sleef_sinf4_u10(values_);
    }

    Vectorized cos() const {
        return Sleef_cosf4_u10(values_);
    }

    Vectorized tan() const {
        return Sleef_tanf4_u10(values_);
    }

    Vectorized asin() const {
        return Sleef_asinf4_u10(values_);
    }

    Vectorized acos() const {
        return Sleef_acosf4_u10(values_);
    }

    Vectorized atan() const {
        return Sleef_atanf4_u10(values_);
    }

    Vectorized sinh() const {
        return Sleef_sinhf4_u10(values_);
    }

    Vectorized cosh() const {
        return Sleef_coshf4_u10(values_);
    }

    Vectorized tanh() const {
        return Sleef_tanhf4_u10(values_);
    }

    // From: aten/src/ATen/cpu/vec/vec128/vec128_float_neon.h (erf)
    // Abramowitz and Stegun 7.1.26, as the AVX2 version: erf(x) =
    // 1 - (a1 t + ... + a5 t^5) e^(-x^2) with t = 1 / (1 + p |x|), absolute
    // error below 1.5e-7, with exp_u20 for e^(-x^2)
    Vectorized erf() const {
        const float32x4_t neg_zero = vdupq_n_f32(-0.f);
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t p = vdupq_n_f32(0.3275911f);
        const float32x4_t p1 = vdupq_n_f32(0.254829592f);
        const float32x4_t p2 = vdupq_n_f32(-0.284496736f);
        const float32x4_t p3 = vdupq_n_f32(1.421413741f);
        const float32x4_t p4 = vdupq_n_f32(-1.453152027f);
        const float32x4_t p5 = vdupq_n_f32(1.061405429f);
        const uint32x4_t sign_mask = vandq_u32(
            vreinterpretq_u32_f32(neg_zero), vreinterpretq_u32_f32(values_));
        // t = 1 / (p * |x| + 1)
        const float32x4_t t = vdivq_f32(one, vfmaq_f32(one, p, vabsq_f32(values_)));
        // r = p5 t^4 + p4 t^3 + p3 t^2 + p2 t + p1, by Horner
        float32x4_t r = vfmaq_f32(p4, p5, t);
        r = vfmaq_f32(p3, r, t);
        r = vfmaq_f32(p2, r, t);
        r = vfmaq_f32(p1, r, t);
        // 1 - r * t * exp(-x^2)
        const float32x4_t exp_neg_x2 = Vectorized(vnegq_f32(vmulq_f32(values_, values_))).exp_u20();
        const float32x4_t result = vfmsq_f32(one, vmulq_f32(t, exp_neg_x2), r);
        return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(result), sign_mask));
    }

    Vectorized erfc() const {
        return Sleef_erfcf4_u15(values_);
    }

    Vectorized pow(const Vectorized& exponent) const {
        return Sleef_powf4_u10(values_, exponent);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return vreinterpretq_f32_u32(vceqq_f32(values_, other.values_));
    }

    Vectorized operator!=(const Vectorized& other) const {
        return vreinterpretq_f32_u32(
            vmvnq_u32(vceqq_f32(values_, other.values_)));
    }

    Vectorized operator<(const Vectorized& other) const {
        return vreinterpretq_f32_u32(vcltq_f32(values_, other.values_));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return vreinterpretq_f32_u32(vcleq_f32(values_, other.values_));
    }

    Vectorized operator>(const Vectorized& other) const {
        return vreinterpretq_f32_u32(vcgtq_f32(values_, other.values_));
    }

    Vectorized operator>=(const Vectorized& other) const {
        return vreinterpretq_f32_u32(vcgeq_f32(values_, other.values_));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    // --- NaN / mask ---

    Vectorized isnan() const {
        return vreinterpretq_f32_u32(
            vmvnq_u32(vceqq_f32(values_, values_)));
    }

    int zero_mask() const {
        uint32x4_t is_zero = vceqzq_f32(values_);
        alignas(16) uint32_t tmp[kSize];
        vst1q_u32(tmp, is_zero);
        int mask = 0;
        for (int i = 0; i < kSize; i++) {
            if (tmp[i]) mask |= (1 << i);
        }
        return mask;
    }
};

// --- Arithmetic ---

template <>
inline Vectorized<float> operator+(const Vectorized<float>& a, const Vectorized<float>& b) {
    return vaddq_f32(a, b);
}

template <>
inline Vectorized<float> operator-(const Vectorized<float>& a, const Vectorized<float>& b) {
    return vsubq_f32(a, b);
}

template <>
inline Vectorized<float> operator*(const Vectorized<float>& a, const Vectorized<float>& b) {
    return vmulq_f32(a, b);
}

template <>
inline Vectorized<float> operator/(const Vectorized<float>& a, const Vectorized<float>& b) {
    return vdivq_f32(a, b);
}

inline Vectorized<float> Vectorized<float>::frac() const {
    return *this - this->trunc();
}

// --- Bitwise ---

template <>
inline Vectorized<float> operator&(const Vectorized<float>& a, const Vectorized<float>& b) {
    return vreinterpretq_f32_u32(
        vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

template <>
inline Vectorized<float> operator|(const Vectorized<float>& a, const Vectorized<float>& b) {
    return vreinterpretq_f32_u32(
        vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

template <>
inline Vectorized<float> operator^(const Vectorized<float>& a, const Vectorized<float>& b) {
    return vreinterpretq_f32_u32(
        veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

inline Vectorized<float> Vectorized<float>::eq(const Vectorized<float>& other) const {
    return (*this == other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ne(const Vectorized<float>& other) const {
    return (*this != other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::gt(const Vectorized<float>& other) const {
    return (*this > other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::ge(const Vectorized<float>& other) const {
    return (*this >= other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::lt(const Vectorized<float>& other) const {
    return (*this < other) & Vectorized<float>(1.0f);
}

inline Vectorized<float> Vectorized<float>::le(const Vectorized<float>& other) const {
    return (*this <= other) & Vectorized<float>(1.0f);
}

// --- maximum/minimum ---

template <>
inline Vectorized<float> maximum(const Vectorized<float>& a, const Vectorized<float>& b) {
    return vmaxq_f32(a, b);
}

template <>
inline Vectorized<float> minimum(const Vectorized<float>& a, const Vectorized<float>& b) {
    return vminq_f32(a, b);
}

template <>
inline Vectorized<float> clamp(
    const Vectorized<float>& a,
    const Vectorized<float>& min_val,
    const Vectorized<float>& max_val
) {
    return minimum(max_val, maximum(min_val, a));
}

template <>
inline Vectorized<float> clamp_min(const Vectorized<float>& a, const Vectorized<float>& min_val) {
    return maximum(min_val, a);
}

template <>
inline Vectorized<float> clamp_max(const Vectorized<float>& a, const Vectorized<float>& max_val) {
    return minimum(max_val, a);
}

// --- FMA ---

template <>
inline Vectorized<float> fmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c
) {
    return vfmaq_f32(c, a, b);
}

template <>
inline Vectorized<float> fnmadd(
    const Vectorized<float>& a,
    const Vectorized<float>& b,
    const Vectorized<float>& c
) {
    return vfmsq_f32(c, a, b);
}

} // namespace CPU_CAPABILITY
} // namespace raxis::vec