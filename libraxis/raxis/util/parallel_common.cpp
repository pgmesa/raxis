
#include <raxis/util/parallel.h>

#include <cstdlib>
#include <thread>

namespace raxis {

namespace {

size_t get_env_num_threads(const char* var_name, size_t fallback = 0) {
    const char* val = std::getenv(var_name);
    if (!val) return fallback;
    try {
        int n = std::stoi(val);
        if (n > 0) return static_cast<size_t>(n);
    } catch (...) {}
    return fallback;
}

} // anonymous namespace

int intraop_default_num_threads() {
    // Respect OMP_NUM_THREADS
    size_t n = get_env_num_threads("OMP_NUM_THREADS", 0);
    // Also allow our own variable to override
    n = get_env_num_threads("RAXIS_NUM_THREADS", n);

    if (n == 0) {
        n = std::thread::hardware_concurrency();
        if (n == 0) n = 1;
    }
    return static_cast<int>(n);
}

} // namespace raxis