#pragma once

#include <vector>
#include <cstdint>
#include <format>


namespace raxis {

    using Dim = int64_t;
    using IntArray = std::vector<Dim>;

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