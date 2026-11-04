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

// From: aten/src/ATen/cpu/vec/vec128/vec128_double_neon.h
// AArch64 NEON has full double precision lanes (2 per register): every op
// maps to one instruction, transcendentals go to Sleef's AdvSIMD versions
template <>
struct Vectorized<double> {
    using value_type = double;
    using size_type = int;
    static constexpr size_type kSize = 2;

private:
    float64x2_t values_;

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(vdupq_n_f64(0)) {}

    Vectorized(float64x2_t v) : values_(v) {}

    Vectorized(double val) : values_(vdupq_n_f64(val)) {}

    operator float64x2_t() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return vld1q_f64(static_cast<const double*>(ptr));
        }
        alignas(16) double tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(double));
        return vld1q_f64(tmp);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            vst1q_f64(static_cast<double*>(ptr), values_);
            return;
        }
        alignas(16) double tmp[kSize];
        vst1q_f64(tmp, values_);
        std::memcpy(ptr, tmp, count * sizeof(double));
    }

    double operator[](int idx) const {
        alignas(16) double tmp[kSize];
        vst1q_f64(tmp, values_);
        return tmp[idx];
    }

    Vectorized map(double (*const f)(double)) const {
        alignas(16) double tmp[kSize];
        store(tmp);
        for (int i = 0; i < kSize; i++) {
            tmp[i] = f(tmp[i]);
        }
        return loadu(tmp);
    }

    double reduce_add() const {
        return vaddvq_f64(values_);
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        const uint64x2_t select = {
            (mask & 0x01) ? ~uint64_t{0} : 0,
            (mask & 0x02) ? ~uint64_t{0} : 0
        };
        return vbslq_f64(select, b.values_, a.values_);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return vbslq_f64(vreinterpretq_u64_f64(mask.values_), b.values_, a.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        switch (count) {
            case 0: return a;
            case 1: return blend<1>(a, b);
        }
        return b;
    }

    template <typename step_t>
    static Vectorized arange(double base = 0., step_t step = static_cast<step_t>(1)) {
        const float64x2_t v = {base, static_cast<double>(base + step)};
        return v;
    }

    // --- Unary ---

    Vectorized neg() const {
        return vnegq_f64(values_);
    }

    Vectorized abs() const {
        return vabsq_f64(values_);
    }

    Vectorized sqrt() const {
        return vsqrtq_f64(values_);
    }

    Vectorized reciprocal() const {
        return vdivq_f64(vdupq_n_f64(1.0), values_);
    }

    Vectorized rsqrt() const {
        return vdivq_f64(vdupq_n_f64(1.0), vsqrtq_f64(values_));
    }

    Vectorized ceil() const {
        return vrndpq_f64(values_);
    }

    Vectorized floor() const {
        return vrndmq_f64(values_);
    }

    Vectorized round() const {
        return vrndnq_f64(values_);
    }

    Vectorized trunc() const {
        return vrndq_f64(values_);
    }

    Vectorized frac() const;

    // --- Transcendentals (Sleef) ---

    Vectorized exp() const {
        return Sleef_expd2_u10(values_);
    }

    Vectorized exp_u20() const {
        return exp();
    }

    Vectorized expm1() const {
        return Sleef_expm1d2_u10(values_);
    }

    Vectorized log() const {
        return Sleef_logd2_u10(values_);
    }

    Vectorized log2() const {
        return Sleef_log2d2_u10(values_);
    }

    Vectorized log10() const {
        return Sleef_log10d2_u10(values_);
    }

    Vectorized log1p() const {
        return Sleef_log1pd2_u10(values_);
    }

    Vectorized sin() const {
        return Sleef_sind2_u10(values_);
    }

    Vectorized cos() const {
        return Sleef_cosd2_u10(values_);
    }

    Vectorized tan() const {
        return Sleef_tand2_u10(values_);
    }

    Vectorized asin() const {
        return Sleef_asind2_u10(values_);
    }

    Vectorized acos() const {
        return Sleef_acosd2_u10(values_);
    }

    Vectorized atan() const {
        return Sleef_atand2_u10(values_);
    }

    Vectorized sinh() const {
        return Sleef_sinhd2_u10(values_);
    }

    Vectorized cosh() const {
        return Sleef_coshd2_u10(values_);
    }

    Vectorized tanh() const {
        return Sleef_tanhd2_u10(values_);
    }

    Vectorized erf() const {
        return Sleef_erfd2_u10(values_);
    }

    Vectorized erfc() const {
        return Sleef_erfcd2_u15(values_);
    }

    Vectorized pow(const Vectorized& exponent) const {
        return Sleef_powd2_u10(values_, exponent);
    }

    // Exact, as std::fmod
    Vectorized fmod(const Vectorized& divisor) const {
        return Sleef_fmodd2(values_, divisor);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return vreinterpretq_f64_u64(vceqq_f64(values_, other.values_));
    }

    Vectorized operator!=(const Vectorized& other) const {
        return vreinterpretq_f64_u32(
            vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(values_, other.values_))));
    }

    Vectorized operator<(const Vectorized& other) const {
        return vreinterpretq_f64_u64(vcltq_f64(values_, other.values_));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return vreinterpretq_f64_u64(vcleq_f64(values_, other.values_));
    }

    Vectorized operator>(const Vectorized& other) const {
        return vreinterpretq_f64_u64(vcgtq_f64(values_, other.values_));
    }

    Vectorized operator>=(const Vectorized& other) const {
        return vreinterpretq_f64_u64(vcgeq_f64(values_, other.values_));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    // --- NaN / mask ---

    Vectorized isnan() const {
        return vreinterpretq_f64_u32(
            vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(values_, values_))));
    }

    int zero_mask() const {
        const uint64x2_t is_zero = vceqzq_f64(values_);
        return static_cast<int>((vgetq_lane_u64(is_zero, 0) & 1) |
                                ((vgetq_lane_u64(is_zero, 1) & 1) << 1));
    }
};

// --- Arithmetic ---

template <>
inline Vectorized<double> operator+(const Vectorized<double>& a, const Vectorized<double>& b) {
    return vaddq_f64(a, b);
}

template <>
inline Vectorized<double> operator-(const Vectorized<double>& a, const Vectorized<double>& b) {
    return vsubq_f64(a, b);
}

template <>
inline Vectorized<double> operator*(const Vectorized<double>& a, const Vectorized<double>& b) {
    return vmulq_f64(a, b);
}

template <>
inline Vectorized<double> operator/(const Vectorized<double>& a, const Vectorized<double>& b) {
    return vdivq_f64(a, b);
}

inline Vectorized<double> Vectorized<double>::frac() const {
    return *this - this->trunc();
}

// --- Bitwise ---

template <>
inline Vectorized<double> operator&(const Vectorized<double>& a, const Vectorized<double>& b) {
    return vreinterpretq_f64_u64(
        vandq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
}

template <>
inline Vectorized<double> operator|(const Vectorized<double>& a, const Vectorized<double>& b) {
    return vreinterpretq_f64_u64(
        vorrq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
}

template <>
inline Vectorized<double> operator^(const Vectorized<double>& a, const Vectorized<double>& b) {
    return vreinterpretq_f64_u64(
        veorq_u64(vreinterpretq_u64_f64(a), vreinterpretq_u64_f64(b)));
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

// FMAX / FMIN return NaN when either input is NaN, as torch.maximum
template <>
inline Vectorized<double> maximum(const Vectorized<double>& a, const Vectorized<double>& b) {
    return vmaxq_f64(a, b);
}

template <>
inline Vectorized<double> minimum(const Vectorized<double>& a, const Vectorized<double>& b) {
    return vminq_f64(a, b);
}

template <>
inline Vectorized<double> clamp(
    const Vectorized<double>& a,
    const Vectorized<double>& min_val,
    const Vectorized<double>& max_val
) {
    return minimum(max_val, maximum(min_val, a));
}

template <>
inline Vectorized<double> clamp_min(const Vectorized<double>& a, const Vectorized<double>& min_val) {
    return maximum(min_val, a);
}

template <>
inline Vectorized<double> clamp_max(const Vectorized<double>& a, const Vectorized<double>& max_val) {
    return minimum(max_val, a);
}

// --- FMA ---

template <>
inline Vectorized<double> fmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c
) {
    return vfmaq_f64(c, a, b);
}

template <>
inline Vectorized<double> fnmadd(
    const Vectorized<double>& a,
    const Vectorized<double>& b,
    const Vectorized<double>& c
) {
    return vfmsq_f64(c, a, b);
}

} // namespace CPU_CAPABILITY
} // namespace raxis::vec
