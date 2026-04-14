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

template <typename Container>
struct fmt::formatter<Container,
    std::enable_if_t<raxis::is_printable_container<Container>::value, char>> {

    constexpr auto parse(fmt::format_parse_context& ctx) {
        return ctx.begin();
    }

    template <typename FormatContext>
    auto format(const Container& c, FormatContext& ctx) const {
        auto out = ctx.out();
        *out++ = '[';
        bool first = true;
        for (const auto& v : c) {
            if (!first) out = fmt::format_to(out, ", ");
            out = fmt::format_to(out, "{}", v);
            first = true;
        }
        *out++ = ']';
        return out;
    }
};