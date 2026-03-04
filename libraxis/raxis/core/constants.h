#pragma once

#include <limits>
#include <numbers>


namespace raxis {

constexpr double inf = std::numeric_limits<double>::infinity();
constexpr double nan = std::numeric_limits<double>::quiet_NaN();

constexpr double pi = std::numbers::pi;
constexpr double e  = std::numbers::e;
    
} // namespace raxis
