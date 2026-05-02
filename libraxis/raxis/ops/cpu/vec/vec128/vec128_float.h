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

    Vectorized exp() const {
        return Sleef_expf4_u10(values_);
    }

    Vectorized log() const {
        return Sleef_logf4_u10(values_);
    }

    Vectorized log2() const {
        return Sleef_log2f4_u10(values_);
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