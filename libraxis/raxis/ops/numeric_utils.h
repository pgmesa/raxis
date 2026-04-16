#pragma once

#include <limits>
#include <type_traits>
#include <cmath>
#include <random>


namespace raxis {

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
inline bool _isnan(T /*val*/) {
    return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
inline bool _isnan(T val) {
    return std::isnan(val);
}

template <typename T>
inline T max_impl (T a, T b) {
    if (_isnan<T>(a) || _isnan<T>(b)) {
        return std::numeric_limits<T>::quiet_NaN();
    } else {
        return std::max(a, b);
    }
}

template <typename T>
inline T min_impl (T a, T b) {
    if (_isnan<T>(a) || _isnan<T>(b)) {
        return std::numeric_limits<T>::quiet_NaN();
    } else {
        return std::min(a, b);
    }
}

template <typename scalar_t>
constexpr scalar_t upper_bound() {
    using lim = std::numeric_limits<scalar_t>;
    if constexpr (lim::has_infinity) {
        return lim::infinity();
    } else {
        return lim::max();
    }
}

template <typename scalar_t>
constexpr scalar_t lower_bound() {
    using lim = std::numeric_limits<scalar_t>;
    if constexpr (lim::has_infinity) {
        return -lim::infinity();
    } else {
        return lim::lowest();
    }
}


/// Draw a uniform integer in [0, range) from a raw 64-bit engine.
/// Uses bitmask-with-rejection (Lemire-style) to avoid modulo bias.
template <typename T>
T uniform_int_from(std::mt19937_64& engine, uint64_t range) {
    if (range <= 1) return static_cast<T>(0);

    uint64_t mask = range - 1;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    
    uint64_t draw;
    do {
        draw = engine() & mask;
    } while (draw >= range);

    return static_cast<T>(draw);
}
    
} // namespace raxis

