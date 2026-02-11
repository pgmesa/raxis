#pragma once

#include <fmt/format.h>

#include <raxis/tensor/tensor.h>
#include <raxis/core/types.h>


template <>
struct fmt::formatter<raxis::Tensor> : fmt::formatter<std::string_view> {
    auto format(const raxis::Tensor& a, fmt::format_context& ctx) const {
        auto s = a.to_string();
        return fmt::formatter<std::string_view>::format(s, ctx);
    }
};


template <>
struct fmt::formatter<raxis::IntArray> {
    constexpr auto parse(fmt::format_parse_context& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const raxis::IntArray& v, FormatContext& ctx) const {
        auto out = ctx.out();
        *out++ = '[';
        for (size_t i = 0; i < v.size(); ++i) {
            out = fmt::format_to(out, "{}", v[i]);
            if (i + 1 < v.size())
                out = fmt::format_to(out, ", ");
        }
        *out++ = ']';
        return out;
    }
};