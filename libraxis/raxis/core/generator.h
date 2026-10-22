#pragma once

#include <cstdint>
#include <mutex>
#include <random>
#include <utility>

#include <raxis/export.h>


namespace raxis {

// Default seed matching PyTorch's default_rng_seed_val
inline constexpr uint64_t default_rng_seed_val = 67280421310721ULL;


/// Lightweight RNG wrapper around std::mt19937_64, plus the offset of a
/// Philox stream (see core/philox.h) for kernels that draw in parallel.
class RAXIS_API Generator {
public:
    explicit Generator(uint64_t seed = default_rng_seed_val)
        : seed_(seed), engine_(seed) {}

    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;
    Generator(Generator&&) = delete;
    Generator& operator=(Generator&&) = delete;

    void seed(uint64_t s) {
        std::lock_guard<std::mutex> lock(mutex_);
        seed_ = s;
        engine_.seed(s);
        philox_offset_ = 0;
    }

    uint64_t current_seed() const {
        return seed_;
    }

    /// Access the raw engine. Caller MUST hold mutex() before calling this.
    std::mt19937_64& engine() { 
        return engine_; 
    }

    /// Mutex for external locking. The pattern is:
    ///   std::lock_guard<std::mutex> lock(gen.mutex());
    ///   auto& eng = gen.engine();
    ///   // ... draw samples ...
    std::mutex& mutex() { 
        return mutex_; 
    }

    /// Reserves `increment` Philox blocks (4 x 32 bits each) and returns
    /// (seed, offset): the caller owns counters [offset, offset + increment)
    /// of the stream keyed by seed. Successive calls never overlap, so draws
    /// stay independent, and manual_seed makes them reproducible.
    /// From: aten/src/ATen/cuda/CUDAGeneratorImpl.cpp (philox_engine_inputs)
    std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint64_t offset = philox_offset_;
        philox_offset_ += increment;
        return {seed_, offset};
    }

private:
    uint64_t seed_;
    std::mt19937_64 engine_;
    uint64_t philox_offset_ = 0;
    std::mutex mutex_;
};

/// Returns the process-wide default CPU generator.
RAXIS_API Generator& get_default_generator();

RAXIS_API void manual_seed(uint64_t seed);

namespace detail {

uint64_t non_deterministic_seed();

} // namespace detail

} // namespace raxis