#pragma once

#include <cstdint>
#include <type_traits>


namespace raxis::cuda {

// atomicAdd for every dtype: native for 32 and 64-bit types, a compare and
// swap of the aligned 32-bit word for bytes (a sum of bools is their OR).
// Concurrent adds land in whatever order the threads run, so float sums
// may differ in the last bits between runs.
// From: aten/src/ATen/cuda/Atomic.cuh (AtomicFPOp, AtomicAddIntegerImpl)
template <typename T>
__device__ inline void atomic_add(T* address, T value) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int32_t>) {
        atomicAdd(address, value);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        // Two's complement: the unsigned sum is the signed one
        atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value));
    } else {
        static_assert(sizeof(T) == 1, "atomic_add: unsupported type");
        const auto byte = reinterpret_cast<uintptr_t>(address) % 4;
        auto* word = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(address) - byte);
        const uint32_t shift = byte * 8;
        uint32_t old = *word;
        uint32_t assumed;
        do {
            assumed = old;
            const auto current = static_cast<uint8_t>(assumed >> shift);
            uint8_t sum;
            if constexpr (std::is_same_v<T, bool>) {
                sum = current || value;
            } else {
                sum = static_cast<uint8_t>(current + value);
            }
            old = atomicCAS(word, assumed, (assumed & ~(0xffu << shift)) | (static_cast<uint32_t>(sum) << shift));
        } while (assumed != old);
    }
}

} // namespace raxis::cuda
