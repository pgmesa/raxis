#pragma once

#include <vector>
#include <cstdint>
#include <format>


namespace raxis {

    using f64 = double;
    using f32 = float;
    using i64 = std::int64_t;
    using i32 = std::int32_t;
    using u8 = std::uint8_t;
    using u16 = std::uint16_t;

    using Int = i64;
    using Index = i64;
    using IntArray = std::vector<Int>;

}

template <>
struct std::formatter<raxis::IntArray> : std::formatter<std::string_view>
{
    auto format(const raxis::IntArray& v, format_context& ctx) const
    {
        auto out = ctx.out();
        *out++ = '[';

        for (size_t i = 0; i < v.size(); ++i) {
            out = std::format_to(out, "{}", v[i]);
            if (i + 1 < v.size()) {
                *out++ = ',';
                *out++ = ' ';
            }
        }

        *out++ = ']';
        return out;
    }
};