// How long one GPU thread waits for memory, for working sets of growing
// size: each step in the numbers is one level of the GPU's memory (L1, L2,
// GDDR). The GPU version of memory/memory_levels.cpp.
//
// A chain of 128-byte cache lines, each holding the index of the next one,
// walked by a single thread: every load needs the result of the previous
// one, so the time per load is the latency of wherever the line lives.
// "random" visits the lines in a random cycle, "in order" one after another
// (a CPU's prefetcher would bring those early; the numbers show whether the
// GPU does anything alike). The kernel reads two clocks around the timed
// loads: clock64(), the SM's cycle counter, and %globaltimer, the GPU's
// nanosecond timer.
//
// Output, one line per measurement:
//   latency,<bytes>,<random|in order>,<cycles per load>,<ns per load>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>


#define CHECK(expr)                                                              \
    do {                                                                         \
        cudaError_t err_ = (expr);                                               \
        if (err_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(err_));   \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

constexpr int kLine = 128;                       // bytes per node: one cache line
constexpr int kStride = kLine / sizeof(uint32_t);

__device__ uint64_t global_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// A pass over (up to a million lines of) the chain to bring it into the
// caches it fits in, then `steps` timed loads. The result index is stored
// so the compiler keeps the loads. The pass is capped so that a launch
// stays well under Windows' 2-second limit on a kernel (TDR); past the L2
// it warms nothing anyway.
__global__ void chase(const uint32_t* next, int64_t n, int64_t steps, long long* result, uint32_t* sink) {
    uint32_t j = 0;
    const int64_t warm = n < 1'000'000 ? n : 1'000'000;
    for (int64_t s = 0; s < warm; s++) {
        j = next[j];
    }
    const uint64_t ns0 = global_ns();
    const long long start = clock64();
    for (int64_t s = 0; s < steps; s++) {
        j = next[j];
    }
    const long long cycles = clock64() - start;
    const uint64_t ns1 = global_ns();
    result[0] = cycles;
    result[1] = static_cast<long long>(ns1 - ns0);
    *sink = j;
}

int main() {
    std::mt19937_64 rng(0);
    long long* d_result;
    uint32_t* d_sink;
    CHECK(cudaMalloc(&d_result, 2 * sizeof(long long)));
    CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));

    for (int64_t kb : {16, 32, 64, 80, 96, 112, 128, 256, 1024, 4096, 16384, 20480, 24576, 28672, 32768,
                       36864, 40960, 49152, 65536, 131072, 524288}) {
        const int64_t bytes = kb * 1024;
        const int64_t n = bytes / kLine;
        std::vector<int64_t> order(n);
        std::iota(order.begin(), order.end(), int64_t{0});

        for (const char* kind : {"in order", "random"}) {
            if (kind[0] == 'r') {
                // Sattolo's shuffle: one random cycle through every line
                for (int64_t k = n - 1; k > 0; k--) {
                    std::uniform_int_distribution<int64_t> pick(0, k - 1);
                    std::swap(order[k], order[pick(rng)]);
                }
            }

            // Node order[k] points to node order[k + 1]; a node is the
            // first 4 bytes of its line, the index counts uint32 slots.
            // The walk starts at slot 0, which is in the cycle like every line
            std::vector<uint32_t> next(n * kStride, 0);
            for (int64_t k = 0; k < n; k++) {
                next[order[k] * kStride] = static_cast<uint32_t>(order[(k + 1) % n] * kStride);
            }

            uint32_t* d_next;
            CHECK(cudaMalloc(&d_next, bytes));
            CHECK(cudaMemcpy(d_next, next.data(), bytes, cudaMemcpyHostToDevice));

            const int64_t steps = 200'000;
            double best_cycles = 1e30;
            double best_ns = 1e30;
            for (int rep = 0; rep < 3; rep++) {
                chase<<<1, 1>>>(d_next, n, steps, d_result, d_sink);
                CHECK(cudaGetLastError());
                long long result[2];
                CHECK(cudaMemcpy(result, d_result, sizeof(result), cudaMemcpyDeviceToHost));
                best_cycles = std::min(best_cycles, static_cast<double>(result[0]) / steps);
                best_ns = std::min(best_ns, static_cast<double>(result[1]) / steps);
            }

            CHECK(cudaFree(d_next));
            std::printf("latency,%lld,%s,%.1f,%.1f\n", static_cast<long long>(bytes), kind, best_cycles, best_ns);
            std::fflush(stdout);
        }
    }
}
