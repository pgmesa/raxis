#pragma once

#include <cassert>
#include <cstdint>


// Integer division by multiplication, for a divisor fixed at launch time.
// From: aten/src/ATen/cuda/detail/IntegerDivider.cuh
//
// Each GPU thread turns its linear index into an offset per operand by
// dividing by every dim size (OffsetCalculator). An integer division is
// tens of instructions on a GPU, and these divisors are known before the
// launch, so the host precomputes for each one a "magic number" m' and a
// shift s such that
//
//     n / d == (umulhi(n, m') + n) >> s     for 0 <= n <= INT32_MAX
//
// where umulhi is the high 32 bits of the 64-bit product, one instruction.
// With s = ceil(log2(d)) and m' = floor(2^32 * (2^s - d) / d) + 1, the
// quotient is exact for any n under 2^31 (the reason the kernels split
// iterators that need 64-bit offsets). Granlund and Montgomery, "Division by
// Invariant Integers Using Multiplication", 1994.
//
// The CPU loops avoid the division differently: each thread divides once to
// find where its chunk starts and then walks the dims as an odometer
// (DimCounter). A GPU thread handles only a few elements, so it pays the
// division per element, and makes it cheap.

namespace raxis::cuda {

template <typename Value>
struct DivMod {
    Value div;
    Value mod;

    __host__ __device__ DivMod(Value div, Value mod) : div(div), mod(mod) {}
};

struct IntDivider {
    IntDivider() = default;

    explicit IntDivider(uint32_t d) : divisor(d) {
        assert(divisor >= 1 && divisor <= INT32_MAX);
        for (shift = 0; shift < 32; shift++) {
            if ((1u << shift) >= divisor) break;
        }
        uint64_t one = 1;
        uint64_t magic = ((one << 32) * ((one << shift) - divisor)) / divisor + 1;
        m1 = static_cast<uint32_t>(magic);
        assert(m1 > 0 && m1 == magic);  // fits in 32 bits
    }

    __host__ __device__ inline uint32_t div(uint32_t n) const {
#ifdef __CUDA_ARCH__
        uint32_t t = __umulhi(n, m1);
#else
        uint32_t t = static_cast<uint32_t>((static_cast<uint64_t>(n) * m1) >> 32);
#endif
        return (t + n) >> shift;
    }

    __host__ __device__ inline DivMod<uint32_t> divmod(uint32_t n) const {
        uint32_t q = div(n);
        return DivMod<uint32_t>(q, n - q * divisor);
    }

    uint32_t divisor;
    uint32_t m1;     // magic number m'
    uint32_t shift;
};

} // namespace raxis::cuda
