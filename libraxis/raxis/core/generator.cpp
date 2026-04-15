
#include <raxis/core/generator.h>


namespace raxis {

namespace detail {

uint64_t non_deterministic_seed() {
    std::random_device rd;
    // Combine two 32-bit draws into one 64-bit value
    return (static_cast<uint64_t>(rd()) << 32) | rd();
}

} // namespace detail

Generator& get_default_generator() {
    static Generator instance(detail::non_deterministic_seed());
    return instance;
}

void manual_seed(uint64_t seed) {
    get_default_generator().seed(seed);
}

} // namespace raxis