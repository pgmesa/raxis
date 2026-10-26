#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <raxis/core/types.h>
#include <raxis/export.h>


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


namespace raxis {

// From: c10/util/Exception.h (IndexError)
// An index out of range: Python sees an IndexError, which also ends the
// sequence protocol (iterating a tensor through __getitem__)
class RAXIS_VISIBLE IndexError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// From: c10/util/Exception.h (NotImplementedError)
// A valid request raxis does not support: Python sees a NotImplementedError
class RAXIS_VISIBLE NotImplementedError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

} // namespace raxis


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

// Like RAXIS_CHECK, raising raxis::NotImplementedError (TORCH_CHECK_NOT_IMPLEMENTED)
#define RAXIS_CHECK_NOT_IMPLEMENTED(cond, ...)                                    do {                                                                              if (!(cond)) [[unlikely]] {                                                       throw ::raxis::NotImplementedError(                                               ::raxis::detail::format_message(__VA_ARGS__));                        }                                                                         } while (0)

// Like RAXIS_CHECK, raising raxis::IndexError (TORCH_CHECK_INDEX)
#define RAXIS_CHECK_INDEX(cond, ...)                                          \
    do {                                                                      \
        if (!(cond)) [[unlikely]] {                                           \
            throw ::raxis::IndexError(                                        \
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