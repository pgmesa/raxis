#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <raxis/core/types.h>


namespace raxis::detail {

// ============================================================
// Message formatting
// ============================================================

template <typename... Args>
inline std::string format_message(Args&&... args) {
    if constexpr (sizeof...(Args) == 0) {
        return {};
    } else {
        std::ostringstream ss;
        (ss << ... << std::forward<Args>(args));
        return ss.str();
    }
}

// ============================================================
// Failure backends
// ============================================================
[[noreturn]] inline void throw_internal_error(
    const char* condition,
    const char* file,
    int line,
    std::string&& message)
{
    std::ostringstream ss;
    ss << "Internal error: ";

    if (!message.empty()) {
        ss << message << " ";
    }

    ss << "["
       << file
       << ":"
       << line
       << "] Assert failed: "
       << condition;

    throw std::runtime_error(ss.str());
}

} // namespace raxis::detail


// ============================================================
// Public Macros
// ============================================================

// ------------------------------------------------------------
// RAXIS_CHECK
// User-facing validation (always enabled)
// ------------------------------------------------------------

#define RAXIS_CHECK(cond, ...)                                                \
    do {                                                                      \
        if (!(cond)) [[unlikely]] {                                           \
            throw std::runtime_error(                                         \
                ::raxis::detail::format_message(__VA_ARGS__));                \
        }                                                                     \
    } while (0)


// ------------------------------------------------------------
// RAXIS_INTERNAL_ASSERT
// Invariants, always enabled
// Indicates a bug in raxis itself
// ------------------------------------------------------------

#define RAXIS_INTERNAL_ASSERT(cond, ...)                                      \
    do {                                                                      \
        if (!(cond)) [[unlikely]] {                                           \
            ::raxis::detail::throw_internal_error(                            \
                #cond,                                                        \
                __FILE__,                                                      \
                __LINE__,                                                      \
                ::raxis::detail::format_message(__VA_ARGS__));                \
        }                                                                     \
    } while (0)


// ------------------------------------------------------------
// RAXIS_DEBUG_ASSERT
// Compiled away in Release builds
// ------------------------------------------------------------

#ifdef NDEBUG
#define RAXIS_DEBUG_ASSERT(cond, ...) ((void)0)
#else
#define RAXIS_DEBUG_ASSERT(cond, ...) RAXIS_INTERNAL_ASSERT(cond, __VA_ARGS__)
#endif