#include <raxis/core/context.h>

#include <atomic>


namespace raxis {

namespace {

std::atomic<bool> allow_tf32_matmul_flag{false};
std::atomic<bool> allow_tf32_conv_flag{true};

} // namespace

bool allow_tf32_matmul() {
    return allow_tf32_matmul_flag.load(std::memory_order_relaxed);
}

void set_allow_tf32_matmul(bool allow) {
    allow_tf32_matmul_flag.store(allow, std::memory_order_relaxed);
}

bool allow_tf32_conv() {
    return allow_tf32_conv_flag.load(std::memory_order_relaxed);
}

void set_allow_tf32_conv(bool allow) {
    allow_tf32_conv_flag.store(allow, std::memory_order_relaxed);
}

} // namespace raxis
