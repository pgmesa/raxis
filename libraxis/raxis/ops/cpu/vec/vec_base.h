#pragma once

// DO NOT DEFINE STATIC DATA IN THIS HEADER!
//
// If you define a static initializer in this file, the initialization will use
// SIMD instructions because these object files are compiled with SIMD enabled.
// We need to avoid non-trivial global data in these architecture specific files
// because there's no way to guard the global initializers with CPU capability
// detection.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <type_traits>

#include <raxis/ops/numeric_utils.h>


// Alignment and vector width must be resolved at preprocessing time because
// the same source file may be compiled multiple times with different ISA flags
// (e.g. -mavx2, -mavx512f, -march=armv8-a). A constexpr value cannot vary
// per translation unit the way a #define can.
#if defined(__aarch64__)
#define VECTOR_WIDTH 16
#else
#define VECTOR_WIDTH 32
#endif

namespace raxis::vec {

inline namespace CPU_CAPABILITY {

template <typename T>
struct Vectorized {
    using value_type = T;
    using size_type = int;
    static constexpr size_type kSize = VECTOR_WIDTH / sizeof(T);

private:
    alignas(VECTOR_WIDTH) T values_[kSize];

    template <typename Op>
    inline Vectorized binary_pred(const Vectorized& other, Op op) const {
        Vectorized result;
        for (int i = 0; i < kSize; i++) {
            if (op(values_[i], other.values_[i])) {
                std::memset(static_cast<void*>(result.values_ + i), 0xFF, sizeof(T));
            } else {
                std::memset(static_cast<void*>(result.values_ + i), 0, sizeof(T));
            }
        }
        return result;
    }

    static bool mask_is_true(T val) {
        unsigned char bytes[sizeof(T)];
        std::memcpy(bytes, &val, sizeof(T));
        return bytes[sizeof(T) - 1] & 0x80;
    }

public:
    static constexpr size_type size() {
        return kSize;
    }

    Vectorized() : values_{static_cast<T>(0)} {}

    Vectorized(T val) {
        for (int i = 0; i < kSize; i++) {
            values_[i] = val;
        }
    }

    static Vectorized loadu(const void* ptr) {
        Vectorized v;
        std::memcpy(v.values_, ptr, sizeof(v.values_));
        return v;
    }

    static Vectorized loadu(const void* ptr, int64_t count) {
        Vectorized v;
        std::memcpy(v.values_, ptr, count * sizeof(T));
        return v;
    }

    void store(void* ptr, int count = kSize) const {
        std::memcpy(ptr, values_, count * sizeof(T));
    }

    const T* as_bytes() const {
        return reinterpret_cast<const T*>(values_);
    }

    T operator[](int idx) const {
        return values_[idx];
    }

    T& operator[](int idx) {
        return values_[idx];
    }

    // --- map / reduce ---

    Vectorized map(T (*const f)(T)) const {
        Vectorized result;
        for (int i = 0; i < kSize; i++) {
            result.values_[i] = f(values_[i]);
        }
        return result;
    }

    Vectorized map(T (*const f)(const T&)) const {
        Vectorized result;
        for (int i = 0; i < kSize; i++) {
            result.values_[i] = f(values_[i]);
        }
        return result;
    }

    T reduce(T (*const f)(T, T)) const {
        T acc = values_[0];
        for (int i = 1; i < kSize; i++) {
            acc = f(acc, values_[i]);
        }
        return acc;
    }

    T reduce_add() const {
        return reduce([](T x, T y) -> T { return x + y; });
    }

    // --- Blend ---

    template <int64_t mask_>
    static Vectorized blend(const Vectorized& a, const Vectorized& b) {
        int64_t mask = mask_;
        Vectorized result;
        for (int i = 0; i < kSize; i++) {
            result.values_[i] = (mask & 0x01) ? b.values_[i] : a.values_[i];
            mask >>= 1;
        }
        return result;
    }

    static Vectorized blendv(
        const Vectorized& a,
        const Vectorized& b,
        const Vectorized& mask
    ) {
        Vectorized result;
        for (int i = 0; i < kSize; i++) {
            result.values_[i] = mask_is_true(mask.values_[i])
                ? b.values_[i]
                : a.values_[i];
        }
        return result;
    }

    static Vectorized set(
        const Vectorized& a,
        const Vectorized& b,
        int64_t count = kSize
    ) {
        Vectorized result;
        for (int i = 0; i < kSize; i++) {
            result.values_[i] = (i < count) ? b.values_[i] : a.values_[i];
        }
        return result;
    }

    template <typename step_t>
    static Vectorized arange(T base = static_cast<T>(0), step_t step = static_cast<step_t>(1)) {
        Vectorized result;
        for (int i = 0; i < kSize; i++) {
            result.values_[i] = base + i * step;
        }
        return result;
    }

    // --- Unary ---

    Vectorized neg() const {
        return map([](T x) -> T { return -x; });
    }

    Vectorized operator-() const {
        return neg();
    }

    template <typename U = T,
        std::enable_if_t<std::is_floating_point_v<U>, int> = 0>
    Vectorized abs() const {
        return map([](T x) -> T { return std::abs(x); });
    }

    template <typename U = T,
        std::enable_if_t<!std::is_floating_point_v<U> && std::is_signed_v<U>, int> = 0>
    Vectorized abs() const {
        return map([](T x) -> T { return x < static_cast<T>(0) ? -x : x; });
    }

    template <typename U = T,
        std::enable_if_t<std::is_unsigned_v<U>, int> = 0>
    Vectorized abs() const {
        return *this;
    }

    Vectorized sqrt() const {
        return map([](T x) -> T { return static_cast<T>(std::sqrt(static_cast<double>(x))); });
    }

    Vectorized reciprocal() const {
        return map([](T x) -> T { return T(1) / x; });
    }

    Vectorized rsqrt() const {
        return map([](T x) -> T { return T(1) / static_cast<T>(std::sqrt(static_cast<double>(x))); });
    }

    Vectorized exp() const {
        return map([](T x) -> T { return static_cast<T>(std::exp(static_cast<double>(x))); });
    }

    Vectorized log() const {
        return map([](T x) -> T { return static_cast<T>(std::log(static_cast<double>(x))); });
    }

    Vectorized log2() const {
        return map([](T x) -> T { return static_cast<T>(std::log2(static_cast<double>(x))); });
    }

    Vectorized log10() const {
        return map([](T x) -> T { return static_cast<T>(std::log10(static_cast<double>(x))); });
    }

    Vectorized log1p() const {
        return map([](T x) -> T { return static_cast<T>(std::log1p(static_cast<double>(x))); });
    }

    Vectorized expm1() const {
        return map([](T x) -> T { return static_cast<T>(std::expm1(static_cast<double>(x))); });
    }

    Vectorized sin() const {
        return map([](T x) -> T { return static_cast<T>(std::sin(static_cast<double>(x))); });
    }

    Vectorized cos() const {
        return map([](T x) -> T { return static_cast<T>(std::cos(static_cast<double>(x))); });
    }

    Vectorized tan() const {
        return map([](T x) -> T { return static_cast<T>(std::tan(static_cast<double>(x))); });
    }

    Vectorized asin() const {
        return map([](T x) -> T { return static_cast<T>(std::asin(static_cast<double>(x))); });
    }

    Vectorized acos() const {
        return map([](T x) -> T { return static_cast<T>(std::acos(static_cast<double>(x))); });
    }

    Vectorized atan() const {
        return map([](T x) -> T { return static_cast<T>(std::atan(static_cast<double>(x))); });
    }

    Vectorized sinh() const {
        return map([](T x) -> T { return static_cast<T>(std::sinh(static_cast<double>(x))); });
    }

    Vectorized cosh() const {
        return map([](T x) -> T { return static_cast<T>(std::cosh(static_cast<double>(x))); });
    }

    Vectorized tanh() const {
        return map([](T x) -> T { return static_cast<T>(std::tanh(static_cast<double>(x))); });
    }

    Vectorized erf() const {
        return map([](T x) -> T { return static_cast<T>(std::erf(static_cast<double>(x))); });
    }

    Vectorized erfc() const {
        return map([](T x) -> T { return static_cast<T>(std::erfc(static_cast<double>(x))); });
    }

    Vectorized pow(const Vectorized& exponent) const {
        Vectorized result;
        for (int i = 0; i < kSize; i++) {
            result.values_[i] = static_cast<T>(std::pow(
                static_cast<double>(values_[i]),
                static_cast<double>(exponent.values_[i])
            ));
        }
        return result;
    }

    Vectorized ceil() const {
        return map([](T x) -> T { return static_cast<T>(std::ceil(static_cast<double>(x))); });
    }

    Vectorized floor() const {
        return map([](T x) -> T { return static_cast<T>(std::floor(static_cast<double>(x))); });
    }

    Vectorized round() const {
        return map([](T x) -> T { return static_cast<T>(std::nearbyint(static_cast<double>(x))); });
    }

    Vectorized trunc() const {
        return map([](T x) -> T { return static_cast<T>(std::trunc(static_cast<double>(x))); });
    }

    Vectorized frac() const {
        return *this - this->trunc();
    }

    // --- Comparisons (all-1-bits or all-0-bits per lane) ---

    Vectorized operator==(const Vectorized& other) const {
        return binary_pred(other, std::equal_to<T>());
    }

    Vectorized operator!=(const Vectorized& other) const {
        return binary_pred(other, std::not_equal_to<T>());
    }

    Vectorized operator<(const Vectorized& other) const {
        return binary_pred(other, std::less<T>());
    }

    Vectorized operator<=(const Vectorized& other) const {
        return binary_pred(other, std::less_equal<T>());
    }

    Vectorized operator>(const Vectorized& other) const {
        return binary_pred(other, std::greater<T>());
    }

    Vectorized operator>=(const Vectorized& other) const {
        return binary_pred(other, std::greater_equal<T>());
    }

    Vectorized eq(const Vectorized& other) const { return *this == other; }
    Vectorized ne(const Vectorized& other) const { return *this != other; }
    Vectorized lt(const Vectorized& other) const { return *this < other; }
    Vectorized le(const Vectorized& other) const { return *this <= other; }
    Vectorized gt(const Vectorized& other) const { return *this > other; }
    Vectorized ge(const Vectorized& other) const { return *this >= other; }

    // --- NaN detection ---

    Vectorized isnan() const {
        Vectorized result;
        for (int i = 0; i < kSize; i++) {
            if (_isnan(values_[i])) {
                std::memset(static_cast<void*>(result.values_ + i), 0xFF, sizeof(T));
            } else {
                std::memset(static_cast<void*>(result.values_ + i), 0, sizeof(T));
            }
        }
        return result;
    }

    int zero_mask() const {
        int mask = 0;
        for (int i = 0; i < kSize; i++) {
            if (values_[i] == static_cast<T>(0)) {
                mask |= (1 << i);
            }
        }
        return mask;
    }
};

// Free functions for specialization
// --- Arithmetic ---

template <typename T>
Vectorized<T> inline operator+(const Vectorized<T>& a, const Vectorized<T>& b) {
    Vectorized<T> result;
    for (int i = 0; i < Vectorized<T>::size(); i++) {
        result[i] = a[i] + b[i];
    }
    return result;
}

template <typename T>
Vectorized<T> inline operator-(const Vectorized<T>& a, const Vectorized<T>& b) {
    Vectorized<T> result;
    for (int i = 0; i < Vectorized<T>::size(); i++) {
        result[i] = a[i] - b[i];
    }
    return result;
}

template <typename T>
Vectorized<T> inline operator*(const Vectorized<T>& a, const Vectorized<T>& b) {
    Vectorized<T> result;
    for (int i = 0; i < Vectorized<T>::size(); i++) {
        result[i] = a[i] * b[i];
    }
    return result;
}

template <typename T>
Vectorized<T> inline operator/(const Vectorized<T>& a, const Vectorized<T>& b) {
    Vectorized<T> result;
    for (int i = 0; i < Vectorized<T>::size(); i++) {
        result[i] = a[i] / b[i];
    }
    return result;
}

// --- Compound assignment ---

template <typename T>
inline Vectorized<T>& operator+=(Vectorized<T>& a, const Vectorized<T>& b) {
    a = a + b;
    return a;
}

template <typename T>
inline Vectorized<T>& operator-=(Vectorized<T>& a, const Vectorized<T>& b) {
    a = a - b;
    return a;
}

template <typename T>
inline Vectorized<T>& operator*=(Vectorized<T>& a, const Vectorized<T>& b) {
    a = a * b;
    return a;
}

template <typename T>
inline Vectorized<T>& operator/=(Vectorized<T>& a, const Vectorized<T>& b) {
    a = a / b;
    return a;
}

// --- Unary negation ---

template <typename T>
Vectorized<T> inline operator-(const Vectorized<T>& a) {
    return a.neg();
}

// --- Bitwise ---

template <typename T, typename Op>
static inline Vectorized<T> bitwise_binary_op(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    Op op
) {
    static constexpr uint32_t num_chunks = VECTOR_WIDTH / sizeof(intmax_t);
    static_assert(
        VECTOR_WIDTH % sizeof(intmax_t) == 0,
        "VECTOR_WIDTH not a multiple of sizeof(intmax_t)"
    );

    alignas(VECTOR_WIDTH) intmax_t buffer[num_chunks];
    const auto* a_bytes = reinterpret_cast<const char*>(&a[0]);
    const auto* b_bytes = reinterpret_cast<const char*>(&b[0]);

    for (uint32_t i = 0; i < num_chunks; i++) {
        intmax_t a_chunk, b_chunk;
        std::memcpy(&a_chunk, a_bytes + i * sizeof(intmax_t), sizeof(intmax_t));
        std::memcpy(&b_chunk, b_bytes + i * sizeof(intmax_t), sizeof(intmax_t));
        buffer[i] = op(a_chunk, b_chunk);
    }
    return Vectorized<T>::loadu(buffer);
}

template <typename T>
inline Vectorized<T> operator&(const Vectorized<T>& a, const Vectorized<T>& b) {
    return bitwise_binary_op(a, b, std::bit_and<intmax_t>());
}

template <typename T>
inline Vectorized<T> operator|(const Vectorized<T>& a, const Vectorized<T>& b) {
    return bitwise_binary_op(a, b, std::bit_or<intmax_t>());
}

template <typename T>
inline Vectorized<T> operator^(const Vectorized<T>& a, const Vectorized<T>& b) {
    return bitwise_binary_op(a, b, std::bit_xor<intmax_t>());
}

template <typename T>
inline Vectorized<T> operator~(const Vectorized<T>& a) {
    Vectorized<T> ones;
    std::memset(&ones[0], 0xFF, sizeof(T) * Vectorized<T>::size());
    return a ^ ones;
}

template <typename T>
Vectorized<T> inline operator&&(const Vectorized<T>& a, const Vectorized<T>& b) {
    Vectorized<T> ret;
    for (int i = 0; i != Vectorized<T>::size(); i++) {
        ret[i] = a[i] && b[i];
    }
    return ret;
}

// --- maximum/minimum (propagate NaN) ---

template <typename T>
Vectorized<T> inline maximum(const Vectorized<T>& a, const Vectorized<T>& b) {
    Vectorized<T> result;
    for (int i = 0; i < Vectorized<T>::size(); i++) {
        result[i] = (a[i] > b[i]) ? a[i] : b[i];
        if (_isnan(a[i])) {
            result[i] = a[i];
        }
    }
    return result;
}

template <typename T>
Vectorized<T> inline minimum(const Vectorized<T>& a, const Vectorized<T>& b) {
    Vectorized<T> result;
    for (int i = 0; i < Vectorized<T>::size(); i++) {
        result[i] = (a[i] < b[i]) ? a[i] : b[i];
        if (_isnan(a[i])) {
            result[i] = a[i];
        }
    }
    return result;
}

template <typename T>
Vectorized<T> inline clamp(
    const Vectorized<T>& a,
    const Vectorized<T>& min_val,
    const Vectorized<T>& max_val
) {
    Vectorized<T> result;
    for (int i = 0; i < Vectorized<T>::size(); i++) {
        result[i] = std::min(std::max(a[i], min_val[i]), max_val[i]);
    }
    return result;
}

template <typename T>
Vectorized<T> inline clamp_min(const Vectorized<T>& a, const Vectorized<T>& min_val) {
    Vectorized<T> result;
    for (int i = 0; i < Vectorized<T>::size(); i++) {
        result[i] = a[i] < min_val[i] ? min_val[i] : a[i];
    }
    return result;
}

template <typename T>
Vectorized<T> inline clamp_max(const Vectorized<T>& a, const Vectorized<T>& max_val) {
    Vectorized<T> result;
    for (int i = 0; i < Vectorized<T>::size(); i++) {
        result[i] = a[i] > max_val[i] ? max_val[i] : a[i];
    }
    return result;
}

// --- Fused multiply-add ---

template <typename T>
Vectorized<T> inline fmadd(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c
) {
    return a * b + c;
}

template <typename T>
Vectorized<T> inline fnmadd(
    const Vectorized<T>& a,
    const Vectorized<T>& b,
    const Vectorized<T>& c
) {
    return -(a * b) + c;
}

} // namespace CPU_CAPABILITY
} // namespace raxis::vec