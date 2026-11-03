#pragma once

#include <string>

#include <raxis/export.h>
#include <raxis/util/exception.h>


namespace raxis {

// From: c10/util/Exception.h (TORCH_WARN, WarningHandler)
// A warning for the user. The Python bindings route it to Python's warnings
// module (so warnings.simplefilter("error") turns it into an exception);
// C++ programs get it on stderr unless they set their own handler.
using WarningHandler = void (*)(const std::string& message);

RAXIS_API void set_warning_handler(WarningHandler handler);
RAXIS_API void warn(const std::string& message);

} // namespace raxis

#define RAXIS_WARN(...) ::raxis::warn(::raxis::detail::format_message(__VA_ARGS__))
