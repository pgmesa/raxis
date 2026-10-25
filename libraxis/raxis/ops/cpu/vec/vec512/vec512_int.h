#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstring>

#include <raxis/ops/cpu/vec/vec_base.h>


// From: aten/src/ATen/cpu/vec/vec512/vec512_int.h
// AVX512 fills the gaps of AVX2 for integers: 64-bit multiply (DQ), 64-bit
// min/max/abs (F) and unsigned byte comparisons (BW) are single instructions.
// As in vec512_float.h, comparisons produce a mask register that movm turns
// into all-ones lanes.

namespace raxis::vec {

inline namespace CPU_CAPABILITY {

template <>
struct Vectorized<int32_t> {
    using value_type = int32_t;
    using size_type = int;
    static constexpr size_type kSize = 16;

private:
    __m512i values_;

    static __mmask16 first_lanes(int64_t count) {
        return static_cast<__mmask16>((1u << count) - 1);
    }

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm512_setzero_si512()) {}

    Vectorized(__m512i v) : values_(v) {}

    Vectorized(int32_t val) : values_(_mm512_set1_epi32(val)) {}

    operator __m512i() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm512_loadu_si512(ptr);
        }
        return _mm512_maskz_loadu_epi32(first_lanes(count), ptr);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm512_storeu_si512(ptr, values_);
            return;
        }
        _mm512_mask_storeu_epi32(ptr, first_lanes(count), values_);
    }

    int32_t operator[](int idx) const {
        alignas(64) int32_t tmp[kSize];
        _mm512_store_si512(tmp, values_);
        return tmp[idx];
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        return _mm512_mask_blend_epi32(static_cast<__mmask16>(mask), a.values_, b.values_);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return _mm512_mask_blend_epi32(_mm512_movepi32_mask(mask.values_), a.values_, b.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        if (count >= kSize) {
            return b;
        }
        return _mm512_mask_blend_epi32(first_lanes(count), a.values_, b.values_);
    }

    template <typename step_t>
    static Vectorized arange(int32_t base = 0, step_t step = static_cast<step_t>(1)) {
        alignas(64) int32_t tmp[kSize];
        for (int i = 0; i < kSize; i++) {
            tmp[i] = static_cast<int32_t>(base + i * step);
        }
        return _mm512_load_si512(tmp);
    }

    // --- Unary ---

    Vectorized neg() const;

    Vectorized abs() const {
        return _mm512_abs_epi32(values_);
    }

    Vectorized sqrt() const {
        alignas(64) int32_t tmp[kSize];
        store(tmp);
        for (int i = 0; i < kSize; i++) {
            tmp[i] = static_cast<int32_t>(std::sqrt(static_cast<double>(tmp[i])));
        }
        return _mm512_load_si512(tmp);
    }

    Vectorized reciprocal() const {
        return Vectorized(0);
    }

    // --- Reductions ---

    int32_t reduce_add() const {
        return _mm512_reduce_add_epi32(values_);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return _mm512_movm_epi32(_mm512_cmpeq_epi32_mask(values_, other.values_));
    }

    Vectorized operator!=(const Vectorized& other) const {
        return _mm512_movm_epi32(_mm512_cmpneq_epi32_mask(values_, other.values_));
    }

    Vectorized operator>(const Vectorized& other) const {
        return _mm512_movm_epi32(_mm512_cmpgt_epi32_mask(values_, other.values_));
    }

    Vectorized operator<(const Vectorized& other) const {
        return _mm512_movm_epi32(_mm512_cmplt_epi32_mask(values_, other.values_));
    }

    Vectorized operator>=(const Vectorized& other) const {
        return _mm512_movm_epi32(_mm512_cmpge_epi32_mask(values_, other.values_));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return _mm512_movm_epi32(_mm512_cmple_epi32_mask(values_, other.values_));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    Vectorized isnan() const {
        return _mm512_setzero_si512();
    }

    int zero_mask() const {
        return _mm512_cmpeq_epi32_mask(values_, _mm512_setzero_si512());
    }
};

// Free function specializations
// --- Arithmetic ---

template <>
inline Vectorized<int32_t> operator+(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm512_add_epi32(a, b);
}

template <>
inline Vectorized<int32_t> operator-(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm512_sub_epi32(a, b);
}

template <>
inline Vectorized<int32_t> operator*(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm512_mullo_epi32(a, b);
}

// No native SIMD integer division
template <>
inline Vectorized<int32_t> operator/(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    alignas(64) int32_t a_arr[Vectorized<int32_t>::size()];
    alignas(64) int32_t b_arr[Vectorized<int32_t>::size()];
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
    return _mm512_and_si512(a, b);
}

template <>
inline Vectorized<int32_t> operator|(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm512_or_si512(a, b);
}

template <>
inline Vectorized<int32_t> operator^(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm512_xor_si512(a, b);
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
    return _mm512_max_epi32(a, b);
}

template <>
inline Vectorized<int32_t> minimum(const Vectorized<int32_t>& a, const Vectorized<int32_t>& b) {
    return _mm512_min_epi32(a, b);
}

// --- clamp ---

template <>
inline Vectorized<int32_t> clamp(
    const Vectorized<int32_t>& a,
    const Vectorized<int32_t>& min_val,
    const Vectorized<int32_t>& max_val
) {
    return _mm512_min_epi32(max_val, _mm512_max_epi32(a, min_val));
}

template <>
inline Vectorized<int32_t> clamp_min(const Vectorized<int32_t>& a, const Vectorized<int32_t>& min_val) {
    return _mm512_max_epi32(min_val, a);
}

template <>
inline Vectorized<int32_t> clamp_max(const Vectorized<int32_t>& a, const Vectorized<int32_t>& max_val) {
    return _mm512_min_epi32(max_val, a);
}


template <>
struct Vectorized<int64_t> {
    using value_type = int64_t;
    using size_type = int;
    static constexpr size_type kSize = 8;

private:
    __m512i values_;

    static __mmask8 first_lanes(int64_t count) {
        return static_cast<__mmask8>((1u << count) - 1);
    }

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm512_setzero_si512()) {}

    Vectorized(__m512i v) : values_(v) {}

    Vectorized(int64_t val) : values_(_mm512_set1_epi64(val)) {}

    operator __m512i() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm512_loadu_si512(ptr);
        }
        return _mm512_maskz_loadu_epi64(first_lanes(count), ptr);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm512_storeu_si512(ptr, values_);
            return;
        }
        _mm512_mask_storeu_epi64(ptr, first_lanes(count), values_);
    }

    int64_t operator[](int idx) const {
        alignas(64) int64_t tmp[kSize];
        _mm512_store_si512(tmp, values_);
        return tmp[idx];
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        return _mm512_mask_blend_epi64(static_cast<__mmask8>(mask), a.values_, b.values_);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return _mm512_mask_blend_epi64(_mm512_movepi64_mask(mask.values_), a.values_, b.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        if (count >= kSize) {
            return b;
        }
        return _mm512_mask_blend_epi64(first_lanes(count), a.values_, b.values_);
    }

    template <typename step_t>
    static Vectorized arange(int64_t base = 0, step_t step = static_cast<step_t>(1)) {
        return Vectorized(_mm512_setr_epi64(
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
        return _mm512_abs_epi64(values_);
    }

    // --- Reductions ---

    int64_t reduce_add() const {
        return _mm512_reduce_add_epi64(values_);
    }

    // --- Comparisons ---

    Vectorized operator==(const Vectorized& other) const {
        return _mm512_movm_epi64(_mm512_cmpeq_epi64_mask(values_, other.values_));
    }

    Vectorized operator!=(const Vectorized& other) const {
        return _mm512_movm_epi64(_mm512_cmpneq_epi64_mask(values_, other.values_));
    }

    Vectorized operator>(const Vectorized& other) const {
        return _mm512_movm_epi64(_mm512_cmpgt_epi64_mask(values_, other.values_));
    }

    Vectorized operator<(const Vectorized& other) const {
        return _mm512_movm_epi64(_mm512_cmplt_epi64_mask(values_, other.values_));
    }

    Vectorized operator>=(const Vectorized& other) const {
        return _mm512_movm_epi64(_mm512_cmpge_epi64_mask(values_, other.values_));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return _mm512_movm_epi64(_mm512_cmple_epi64_mask(values_, other.values_));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    Vectorized isnan() const {
        return _mm512_setzero_si512();
    }

    int zero_mask() const {
        return _mm512_cmpeq_epi64_mask(values_, _mm512_setzero_si512());
    }
};

// --- Arithmetic ---

template <>
inline Vectorized<int64_t> operator+(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm512_add_epi64(a, b);
}

template <>
inline Vectorized<int64_t> operator-(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm512_sub_epi64(a, b);
}

template <>
inline Vectorized<int64_t> operator*(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm512_mullo_epi64(a, b);
}

template <>
inline Vectorized<int64_t> operator/(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    alignas(64) int64_t a_arr[Vectorized<int64_t>::size()];
    alignas(64) int64_t b_arr[Vectorized<int64_t>::size()];
    a.store(a_arr);
    b.store(b_arr);
    for (int i = 0; i < Vectorized<int64_t>::size(); i++) {
        a_arr[i] = a_arr[i] / b_arr[i];
    }
    return Vectorized<int64_t>::loadu(a_arr);
}

inline Vectorized<int64_t> Vectorized<int64_t>::neg() const {
    return Vectorized<int64_t>(0) - *this;
}

// --- Bitwise ---

template <>
inline Vectorized<int64_t> operator&(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm512_and_si512(a, b);
}

template <>
inline Vectorized<int64_t> operator|(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm512_or_si512(a, b);
}

template <>
inline Vectorized<int64_t> operator^(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm512_xor_si512(a, b);
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
    return _mm512_max_epi64(a, b);
}

template <>
inline Vectorized<int64_t> minimum(const Vectorized<int64_t>& a, const Vectorized<int64_t>& b) {
    return _mm512_min_epi64(a, b);
}

// --- clamp ---

template <>
inline Vectorized<int64_t> clamp(
    const Vectorized<int64_t>& a,
    const Vectorized<int64_t>& min_val,
    const Vectorized<int64_t>& max_val
) {
    return _mm512_min_epi64(max_val, _mm512_max_epi64(a, min_val));
}

template <>
inline Vectorized<int64_t> clamp_min(const Vectorized<int64_t>& a, const Vectorized<int64_t>& min_val) {
    return _mm512_max_epi64(min_val, a);
}

template <>
inline Vectorized<int64_t> clamp_max(const Vectorized<int64_t>& a, const Vectorized<int64_t>& max_val) {
    return _mm512_min_epi64(max_val, a);
}


template <>
struct Vectorized<uint8_t> {
    using value_type = uint8_t;
    using size_type = int;
    static constexpr size_type kSize = 64;

private:
    __m512i values_;

    static __mmask64 first_lanes(int64_t count) {
        return (uint64_t{1} << count) - 1;
    }

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_(_mm512_setzero_si512()) {}

    Vectorized(__m512i v) : values_(v) {}

    Vectorized(uint8_t val) : values_(_mm512_set1_epi8(static_cast<char>(val))) {}

    operator __m512i() const {
        return values_;
    }

    static Vectorized loadu(const void* ptr, int64_t count = kSize) {
        if (count == kSize) {
            return _mm512_loadu_si512(ptr);
        }
        return _mm512_maskz_loadu_epi8(first_lanes(count), ptr);
    }

    void store(void* ptr, int count = kSize) const {
        if (count == kSize) {
            _mm512_storeu_si512(ptr, values_);
            return;
        }
        _mm512_mask_storeu_epi8(ptr, first_lanes(count), values_);
    }

    uint8_t operator[](int idx) const {
        alignas(64) uint8_t tmp[kSize];
        _mm512_store_si512(tmp, values_);
        return tmp[idx];
    }

    // --- Blend ---

    template <int64_t mask>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        return _mm512_mask_blend_epi8(static_cast<__mmask64>(mask), a.values_, b.values_);
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        return _mm512_mask_blend_epi8(_mm512_movepi8_mask(mask.values_), a.values_, b.values_);
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        if (count >= kSize) {
            return b;
        }
        return _mm512_mask_blend_epi8(first_lanes(count), a.values_, b.values_);
    }

    template <typename step_t>
    static Vectorized arange(uint8_t base = 0, step_t step = static_cast<step_t>(1)) {
        alignas(64) uint8_t tmp[kSize];
        for (int i = 0; i < kSize; i++) {
            tmp[i] = static_cast<uint8_t>(base + i * step);
        }
        return _mm512_load_si512(tmp);
    }

    // --- Unary ---

    Vectorized neg() const {
        return _mm512_sub_epi8(_mm512_setzero_si512(), values_);
    }

    Vectorized abs() const {
        return *this;
    }

    // --- Comparisons (unsigned) ---

    Vectorized operator==(const Vectorized& other) const {
        return _mm512_movm_epi8(_mm512_cmpeq_epu8_mask(values_, other.values_));
    }

    Vectorized operator!=(const Vectorized& other) const {
        return _mm512_movm_epi8(_mm512_cmpneq_epu8_mask(values_, other.values_));
    }

    Vectorized operator<(const Vectorized& other) const {
        return _mm512_movm_epi8(_mm512_cmplt_epu8_mask(values_, other.values_));
    }

    Vectorized operator<=(const Vectorized& other) const {
        return _mm512_movm_epi8(_mm512_cmple_epu8_mask(values_, other.values_));
    }

    Vectorized operator>(const Vectorized& other) const {
        return _mm512_movm_epi8(_mm512_cmpgt_epu8_mask(values_, other.values_));
    }

    Vectorized operator>=(const Vectorized& other) const {
        return _mm512_movm_epi8(_mm512_cmpge_epu8_mask(values_, other.values_));
    }

    Vectorized eq(const Vectorized& other) const;
    Vectorized ne(const Vectorized& other) const;
    Vectorized lt(const Vectorized& other) const;
    Vectorized le(const Vectorized& other) const;
    Vectorized gt(const Vectorized& other) const;
    Vectorized ge(const Vectorized& other) const;

    Vectorized isnan() const {
        return _mm512_setzero_si512();
    }

    // 64 lanes: the mask needs 64 bits
    int64_t zero_mask() const {
        return static_cast<int64_t>(_mm512_cmpeq_epu8_mask(values_, _mm512_setzero_si512()));
    }
};

// --- Arithmetic ---

template <>
inline Vectorized<uint8_t> operator+(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm512_add_epi8(a, b);
}

template <>
inline Vectorized<uint8_t> operator-(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm512_sub_epi8(a, b);
}

// No 8-bit multiply: the even and odd bytes are multiplied as 16-bit lanes
// and the low byte of each product is kept
template <>
inline Vectorized<uint8_t> operator*(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    const __m512i mask_00ff = _mm512_set1_epi16(0x00FF);
    const __m512i a_lo = _mm512_and_si512(a, mask_00ff);
    const __m512i b_lo = _mm512_and_si512(b, mask_00ff);
    const __m512i a_hi = _mm512_srli_epi16(a, 8);
    const __m512i b_hi = _mm512_srli_epi16(b, 8);
    const __m512i lo = _mm512_and_si512(_mm512_mullo_epi16(a_lo, b_lo), mask_00ff);
    const __m512i hi = _mm512_slli_epi16(_mm512_mullo_epi16(a_hi, b_hi), 8);
    return _mm512_or_si512(lo, hi);
}

template <>
inline Vectorized<uint8_t> operator/(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    alignas(64) uint8_t a_arr[Vectorized<uint8_t>::size()];
    alignas(64) uint8_t b_arr[Vectorized<uint8_t>::size()];
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
    return _mm512_and_si512(a, b);
}

template <>
inline Vectorized<uint8_t> operator|(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm512_or_si512(a, b);
}

template <>
inline Vectorized<uint8_t> operator^(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm512_xor_si512(a, b);
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
    return _mm512_max_epu8(a, b);
}

template <>
inline Vectorized<uint8_t> minimum(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& b) {
    return _mm512_min_epu8(a, b);
}

// --- clamp (unsigned) ---

template <>
inline Vectorized<uint8_t> clamp(
    const Vectorized<uint8_t>& a,
    const Vectorized<uint8_t>& min_val,
    const Vectorized<uint8_t>& max_val
) {
    return _mm512_min_epu8(max_val, _mm512_max_epu8(a, min_val));
}

template <>
inline Vectorized<uint8_t> clamp_min(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& min_val) {
    return _mm512_max_epu8(min_val, a);
}

template <>
inline Vectorized<uint8_t> clamp_max(const Vectorized<uint8_t>& a, const Vectorized<uint8_t>& max_val) {
    return _mm512_min_epu8(max_val, a);
}

} // namespace CPU_CAPABILITY
} // namespace raxis::vec
