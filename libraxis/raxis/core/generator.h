#pragma once

#include <cstdint>
#include <mutex>
#include <random>

#include <raxis/export.h>


namespace raxis {

// Default seed matching PyTorch's default_rng_seed_val (67280421310721)
inline constexpr uint64_t kDefaultSeed = 67280421310721ULL;


/// Lightweight RNG wrapper around std::mt19937_64.
class RAXIS_API Generator {
public:
    explicit Generator(uint64_t seed = kDefaultSeed)
        : seed_(seed), engine_(seed) {}

    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    Generator(Generator&&) = delete;
    Generator& operator=(Generator&&) = delete;

    void seed(uint64_t s) {
        std::lock_guard<std::mutex> lock(mutex_);
        seed_ = s;
        engine_.seed(s);
    }

    uint64_t current_seed() const {
        return seed_;
    }

    /// Access the raw engine. Caller MUST hold mutex() before calling this.
    std::mt19937_64& engine() { return engine_; }

    /// Mutex for external locking. The pattern is:
    ///   std::lock_guard<std::mutex> lock(gen.mutex());
    ///   auto& eng = gen.engine();
    ///   // ... draw samples ...
    std::mutex& mutex() { return mutex_; }

private:
    uint64_t seed_;
    std::mt19937_64 engine_;
    std::mutex mutex_;
};


/// Returns the process-wide default CPU generator.
RAXIS_API Generator& get_default_generator();

RAXIS_API void manual_seed(uint64_t seed);

} // namespace raxis