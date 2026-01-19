#pragma once

#include <fmt/format.h>

#include <raxis/array.hpp>


template <>
struct fmt::formatter<raxis::NDArray> : fmt::formatter<std::string_view> {
    auto format(const raxis::NDArray& a, fmt::format_context& ctx) const {
        auto s = a.to_string();
        return fmt::formatter<std::string_view>::format(s, ctx);
    }
};