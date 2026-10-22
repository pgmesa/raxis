#pragma once

#include <limits>
#include <type_traits>
#include <algorithm>
#include <cmath>


namespace raxis {

template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
bool _isnan(T /*val*/) {
    return false;
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
bool _isnan(T val) {
    return std::isnan(val);
}

template <typename T>
T max_impl (T a, T b) {
    if (_isnan<T>(a) || _isnan<T>(b)) {
        return std::numeric_limits<T>::quiet_NaN();
    } else {
        return std::max(a, b);
    }
}

template <typename T>
T min_impl (T a, T b) {
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

} // namespace raxis

