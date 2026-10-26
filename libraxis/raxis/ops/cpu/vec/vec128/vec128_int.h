#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!

#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <raxis/ops/cpu/vec/vec_base.h>


namespace raxis::vec {

inline namespace CPU_CAPABILITY {

// From: aten/src/ATen/cpu/vec/vec128/vec128_int_aarch64.h and
// vec128_uint_aarch64.h. The same types the AVX2 and AVX512 headers cover:
// int32, int64 and uint8 (bool kernels run on uint8). Unlike AVX2, NEON has
// 64-bit compares, min/max through compare + select, and an 8-bit multiply.
// Comparisons return all-ones lanes, which vbslq turns into a select.

template <>
struct Vectorized<int32_t> {
    using value_type = int32_t;
    using size_type = int;
    static constexpr size_type kSize = 4;

private:
    int32x4_t values_;

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(vdupq_n_s32(0)) {}

    Vectorized(int32x4_t v) : values_(v) {}

    Vectorized(int32_t val) : values_(vdupq_n_s32(val)) {}

    operator int32x4_t() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return vld1q_s32(static_cast<const int32_t*>(ptr));
        }
        alignas(16) int32_t tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(int32_t));
        return vld1q_s32(tmp);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            vst1q_s32(static_cast<int32_t*>(ptr), values_);
            return;
        }
        alignas(16) int32_t tmp[kSize];
        vst1q_s32(tmp, values_);
        std::memcpy(ptr, tmp, count * sizeof(int32_t));
    }

    int32_t operator[](int idx) const {
        alignas(16) int32_t tmp[kSize];
        vst1q_s32(tmp, values_);
        return tmp[idx];
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        const uint32x4_t select = {
            (mask & 0x01) ? ~uint32_t{0} : 0,
            (mask & 0x02) ? ~uint32_t{0} : 0,
            (mask & 0x04) ? ~uint32_t{0} : 0,
            (mask & 0x08) ? ~uint32_t{0} : 0
        };
        return vbslq_s32(select, b.values_, a.values_);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return vbslq_s32(vreinterpretq_u32_s32(mask.values_), b.values_, a.values_);
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
    static Vectorized arange(int32_t base = 0, step_t step = static_cast<step_t>(1)) {
        const int32x4_t v = {
            base,
            static_cast<int32_t>(base + step),
            static_cast<int32_t>(base + 2 * step),
            static_cast<int32_t>(base + 3 * step)
        };
        return v;
    }

    // --- Unary ---

    Vectorized neg() const {
        return vnegq_s32(values_);
    }

    Vectorized abs() const {
        return vabsq_s32(values_);
    }

    Vectorized sqrt() const {
        alignas(16) int32_t tmp[kSize];
        store(tmp);
        for (int i = 0; i < kSize; i++) {
            tmp[i] = static_cast<int32_t>(std::sqrt(static_cast<double>(tmp[i])));
        }
        return vld1q_s32(tmp);
    }

    Vectorized reciprocal() const {
        return Vectorized(0);
    }

    // --- Reductions ---

    int32_t reduce_add() const {
        return vaddvq_s32(values_);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return vreinterpretq_s32_u32(vceqq_s32(values_, other.values_));
    }

    Vectorized operator!=(const Vectorized& other) const {
        return vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(values_, other.values_)));
    }

    Vectorized operator>(const Vectorized& other) const {
        return vreinterpretq_s32_u32(vcgtq_s32(values_, other.values_));
    }

    Vectorized operator<(const Vectorized& other) const {
        return vreinterpretq_s32_u32(vcltq_s32(values_, other.values_));
    }

    Vectorized operator>=(const Vectorized& other) const {
        return vreinterpretq_s32_u32(vcgeq_s32(values_, other.values_));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return vreinterpretq_s32_u32(vcleq_s32(values_, other.values_));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    Vectorized isnan() const {
        return Vectorized(0);
    }

    int zero_mask() const {
        alignas(16) uint32_t tmp[kSize];
        vst1q_u32(tmp, vceqzq_s32(values_));
        int mask = 0;
        for (int i = 0; i < kSize; i++) {
            if (tmp[i]) mask |= (1 << i);
        }
        return mask;
    }
};

// --- Arithmetic ---

template <>
inline Vectorized<int32_t> operator+(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return vaddq_s32(a, b);
}

template <>
inline Vectorized<int32_t> operator-(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return vsubq_s32(a, b);
}

template <>
inline Vectorized<int32_t> operator*(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return vmulq_s32(a, b);
}

// No SIMD integer division
template <>
inline Vectorized<int32_t> operator/(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    alignas(16) int32_t a_arr[Vectorized<int32_t>::size()];
    alignas(16) int32_t b_arr[Vectorized<int32_t>::size()];
    a.store(a_arr);
    b.store(b_arr);
    for (int i = 0; i < Vectorized<int32_t>::size(); i++) {
        a_arr[i] = a_arr[i] / b_arr[i];
    }
    return Vectorized<int32_t>::loadu(a_arr);
}

// --- Bitwise ---

template <>
inline Vectorized<int32_t> operator&(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return vandq_s32(a, b);
}

template <>
inline Vectorized<int32_t> operator|(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return vorrq_s32(a, b);
}

template <>
inline Vectorized<int32_t> operator^(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return veorq_s32(a, b);
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
    return vmaxq_s32(a, b);
}

template <>
inline Vectorized<int32_t> minimum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return vminq_s32(a, b);
}

template <>
inline Vectorized<int32_t> clamp(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& min_val,
    const Vectorized<int32_t>& max_val
) {
    return vminq_s32(max_val, vmaxq_s32(a, min_val));
}

template <>
inline Vectorized<int32_t> clamp_min(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val) {
    return vmaxq_s32(min_val, a);
}

template <>
inline Vectorized<int32_t> clamp_max(const Vectorized<int32_t>& a, const Vectorized<int32_t>& max_val) {
    return vminq_s32(max_val, a);
}


template <>
struct Vectorized<int64_t> {
    using value_type = int64_t;
    using size_type = int;
    static constexpr size_type kSize = 2;

private:
    int64x2_t values_;

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(vdupq_n_s64(0)) {}

    Vectorized(int64x2_t v) : values_(v) {}

    Vectorized(int64_t val) : values_(vdupq_n_s64(val)) {}

    operator int64x2_t() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return vld1q_s64(static_cast<const int64_t*>(ptr));
        }
        alignas(16) int64_t tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(int64_t));
        return vld1q_s64(tmp);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            vst1q_s64(static_cast<int64_t*>(ptr), values_);
            return;
        }
        alignas(16) int64_t tmp[kSize];
        vst1q_s64(tmp, values_);
        std::memcpy(ptr, tmp, count * sizeof(int64_t));
    }

    int64_t operator[](int idx) const {
        alignas(16) int64_t tmp[kSize];
        vst1q_s64(tmp, values_);
        return tmp[idx];
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        const uint64x2_t select = {
            (mask & 0x01) ? ~uint64_t{0} : 0,
            (mask & 0x02) ? ~uint64_t{0} : 0
        };
        return vbslq_s64(select, b.values_, a.values_);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return vbslq_s64(vreinterpretq_u64_s64(mask.values_), b.values_, a.values_);
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
    static Vectorized arange(int64_t base = 0, step_t step = static_cast<step_t>(1)) {
        const int64x2_t v = {base, static_cast<int64_t>(base + step)};
        return v;
    }

    // --- Unary ---

    Vectorized neg() const {
        return vnegq_s64(values_);
    }

    Vectorized abs() const {
        return vabsq_s64(values_);
    }

    // --- Reductions ---

    int64_t reduce_add() const {
        return vaddvq_s64(values_);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return vreinterpretq_s64_u64(vceqq_s64(values_, other.values_));
    }

    Vectorized operator!=(const Vectorized& other) const {
        return vreinterpretq_s64_u32(
            vmvnq_u32(vreinterpretq_u32_u64(vceqq_s64(values_, other.values_))));
    }

    Vectorized operator>(const Vectorized& other) const {
        return vreinterpretq_s64_u64(vcgtq_s64(values_, other.values_));
    }

    Vectorized operator<(const Vectorized& other) const {
        return vreinterpretq_s64_u64(vcltq_s64(values_, other.values_));
    }

    Vectorized operator>=(const Vectorized& other) const {
        return vreinterpretq_s64_u64(vcgeq_s64(values_, other.values_));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return vreinterpretq_s64_u64(vcleq_s64(values_, other.values_));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    Vectorized isnan() const {
        return Vectorized(int64_t{0});
    }

    int zero_mask() const {
        const uint64x2_t is_zero = vceqzq_s64(values_);
        return static_cast<int>((vgetq_lane_u64(is_zero, 0) & 1) |
                                ((vgetq_lane_u64(is_zero, 1) & 1) << 1));
    }
};

// --- Arithmetic ---

template <>
inline Vectorized<int64_t> operator+(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return vaddq_s64(a, b);
}

template <>
inline Vectorized<int64_t> operator-(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return vsubq_s64(a, b);
}

// NEON has no 64-bit lane multiply: two scalar MULs
template <>
inline Vectorized<int64_t> operator*(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    const int64x2_t v = {
        vgetq_lane_s64(a, 0) * vgetq_lane_s64(b, 0),
        vgetq_lane_s64(a, 1) * vgetq_lane_s64(b, 1)
    };
    return v;
}

template <>
inline Vectorized<int64_t> operator/(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    const int64x2_t v = {
        vgetq_lane_s64(a, 0) / vgetq_lane_s64(b, 0),
        vgetq_lane_s64(a, 1) / vgetq_lane_s64(b, 1)
    };
    return v;
}

// --- Bitwise ---

template <>
inline Vectorized<int64_t> operator&(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return vandq_s64(a, b);
}

template <>
inline Vectorized<int64_t> operator|(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return vorrq_s64(a, b);
}

template <>
inline Vectorized<int64_t> operator^(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return veorq_s64(a, b);
}

inline Vectorized<int64_t> Vectorized<int64_t>::eq(const Vectorized<int64_t>& other) const {
    return (*this == other) & Vectorized<int64_t>(int64_t{1});
}

inline Vectorized<int64_t> Vectorized<int64_t>::ne(const Vectorized<int64_t>& other) const {
    return (*this != other) & Vectorized<int64_t>(int64_t{1});
}

inline Vectorized<int64_t> Vectorized<int64_t>::gt(const Vectorized<int64_t>& other) const {
    return (*this > other) & Vectorized<int64_t>(int64_t{1});
}

inline Vectorized<int64_t> Vectorized<int64_t>::ge(const Vectorized<int64_t>& other) const {
    return (*this >= other) & Vectorized<int64_t>(int64_t{1});
}

inline Vectorized<int64_t> Vectorized<int64_t>::lt(const Vectorized<int64_t>& other) const {
    return (*this < other) & Vectorized<int64_t>(int64_t{1});
}

inline Vectorized<int64_t> Vectorized<int64_t>::le(const Vectorized<int64_t>& other) const {
    return (*this <= other) & Vectorized<int64_t>(int64_t{1});
}

// --- maximum/minimum (no 64-bit SMAX / SMIN: compare + select) ---

template <>
inline Vectorized<int64_t> maximum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return vbslq_s64(vcgtq_s64(a, b), a, b);
}

template <>
inline Vectorized<int64_t> minimum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return vbslq_s64(vcltq_s64(a, b), a, b);
}

template <>
inline Vectorized<int64_t> clamp(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& min_val,
    const Vectorized<int64_t>& max_val
) {
    return minimum(maximum(a, min_val), max_val);
}

template <>
inline Vectorized<int64_t> clamp_min(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val) {
    return maximum(min_val, a);
}

template <>
inline Vectorized<int64_t> clamp_max(const Vectorized<int64_t>& a, const Vectorized<int64_t>& max_val) {
    return minimum(max_val, a);
}


template <>
struct Vectorized<uint8_t> {
    using value_type = uint8_t;
    using size_type = int;
    static constexpr size_type kSize = 16;

private:
    uint8x16_t values_;

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(vdupq_n_u8(0)) {}

    Vectorized(uint8x16_t v) : values_(v) {}

    Vectorized(uint8_t val) : values_(vdupq_n_u8(val)) {}

    operator uint8x16_t() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return vld1q_u8(static_cast<const uint8_t*>(ptr));
        }
        alignas(16) uint8_t tmp[kSize] = {};
        std::memcpy(tmp, ptr, count * sizeof(uint8_t));
        return vld1q_u8(tmp);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            vst1q_u8(static_cast<uint8_t*>(ptr), values_);
            return;
        }
        alignas(16) uint8_t tmp[kSize];
        vst1q_u8(tmp, values_);
        std::memcpy(ptr, tmp, count * sizeof(uint8_t));
    }

    uint8_t operator[](int idx) const {
        alignas(16) uint8_t tmp[kSize];
        vst1q_u8(tmp, values_);
        return tmp[idx];
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        alignas(16) uint8_t select[kSize];
        for (int i = 0; i < kSize; i++) {
            select[i] = ((mask >> i) & 1) ? 0xFF : 0;
        }
        return vbslq_u8(vld1q_u8(select), b.values_, a.values_);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return vbslq_u8(mask.values_, b.values_, a.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        alignas(16) uint8_t a_tmp[kSize], b_tmp[kSize];
        a.store(a_tmp);
        b.store(b_tmp);
        for (int64_t i = 0; i < count; i++) {
            a_tmp[i] = b_tmp[i];
        }
        return loadu(a_tmp);
    }

    template <typename step_t>
    static Vectorized arange(uint8_t base = 0, step_t step = static_cast<step_t>(1)) {
        alignas(16) uint8_t tmp[kSize];
        for (int i = 0; i < kSize; i++) {
            tmp[i] = static_cast<uint8_t>(base + i * step);
        }
        return loadu(tmp);
    }

    // --- Unary ---

    Vectorized neg() const {
        return vsubq_u8(vdupq_n_u8(0), values_);
    }

    Vectorized abs() const {
        return *this;
    }

    // --- Comparisons (unsigned) ---

    Vectorized operator==(const Vectorized& other) const {
        return vceqq_u8(values_, other.values_);
    }

    Vectorized operator!=(const Vectorized& other) const {
        return vmvnq_u8(vceqq_u8(values_, other.values_));
    }

    Vectorized operator<(const Vectorized& other) const {
        return vcltq_u8(values_, other.values_);
    }

    Vectorized operator<=(const Vectorized& other) const {
        return vcleq_u8(values_, other.values_);
    }

    Vectorized operator>(const Vectorized& other) const {
        return vcgtq_u8(values_, other.values_);
    }

    Vectorized operator>=(const Vectorized& other) const {
        return vcgeq_u8(values_, other.values_);
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    Vectorized isnan() const {
        return Vectorized(uint8_t{0});
    }

    int zero_mask() const {
        alignas(16) uint8_t tmp[kSize];
        vst1q_u8(tmp, vceqzq_u8(values_));
        int mask = 0;
        for (int i = 0; i < kSize; i++) {
            if (tmp[i]) mask |= (1 << i);
        }
        return mask;
    }
};

// --- Arithmetic ---

template <>
inline Vectorized<uint8_t> operator+(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return vaddq_u8(a, b);
}

template <>
inline Vectorized<uint8_t> operator-(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return vsubq_u8(a, b);
}

template <>
inline Vectorized<uint8_t> operator*(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return vmulq_u8(a, b);
}

template <>
inline Vectorized<uint8_t> operator/(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    alignas(16) uint8_t a_arr[Vectorized<uint8_t>::size()];
    alignas(16) uint8_t b_arr[Vectorized<uint8_t>::size()];
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
    return vandq_u8(a, b);
}

template <>
inline Vectorized<uint8_t> operator|(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return vorrq_u8(a, b);
}

template <>
inline Vectorized<uint8_t> operator^(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return veorq_u8(a, b);
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::eq(const Vectorized<uint8_t>& other) const {
    return (*this == other) & Vectorized<uint8_t>(uint8_t{1});
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::ne(const Vectorized<uint8_t>& other) const {
    return (*this != other) & Vectorized<uint8_t>(uint8_t{1});
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::gt(const Vectorized<uint8_t>& other) const {
    return (*this > other) & Vectorized<uint8_t>(uint8_t{1});
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::ge(const Vectorized<uint8_t>& other) const {
    return (*this >= other) & Vectorized<uint8_t>(uint8_t{1});
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::lt(const Vectorized<uint8_t>& other) const {
    return (*this < other) & Vectorized<uint8_t>(uint8_t{1});
}

inline Vectorized<uint8_t> Vectorized<uint8_t>::le(const Vectorized<uint8_t>& other) const {
    return (*this <= other) & Vectorized<uint8_t>(uint8_t{1});
}

// --- maximum/minimum (unsigned) ---

template <>
inline Vectorized<uint8_t> maximum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return vmaxq_u8(a, b);
}

template <>
inline Vectorized<uint8_t> minimum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return vminq_u8(a, b);
}

template <>
inline Vectorized<uint8_t> clamp(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& min_val,
    const Vectorized<uint8_t>& max_val
) {
    return vminq_u8(max_val, vmaxq_u8(a, min_val));
}

template <>
inline Vectorized<uint8_t> clamp_min(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& min_val) {
    return vmaxq_u8(min_val, a);
}

template <>
inline Vectorized<uint8_t> clamp_max(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& max_val) {
    return vminq_u8(max_val, a);
}

} // namespace CPU_CAPABILITY
} // namespace raxis::vec
