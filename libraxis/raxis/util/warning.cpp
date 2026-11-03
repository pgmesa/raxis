#include <raxis/util/warning.h>

#include <iostream>


namespace raxis {

namespace {

void print_warning(const std::string& message) {
    std::cerr << "Warning: " << message << "\n";
}

WarningHandler g_handler = print_warning;

} // anonymous namespace

void set_warning_handler(WarningHandler handler) {
    g_handler = handler ? handler : print_warning;
}

void warn(const std::string& message) {
    g_handler(message);
}

} // namespace raxis
