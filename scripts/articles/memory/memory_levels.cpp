// How long the CPU waits for memory, and how fast a loop can stream it, for
// working sets of growing size: each step in the numbers is one cache level.
//
// latency: a chain of 64-byte cache lines, each holding a pointer to the
// next one. Every load needs the result of the previous one, so the time per
// load is the latency of wherever the line lives. "random" visits the lines
// in a random cycle (neither the order nor the prefetcher helps), "in order"
// visits them one after another (the prefetcher brings them early).
//
// add: out[i] = a[i] + b[i] over three float arrays of n elements, repeated
// over the same arrays, as a tensor op that runs many times would.
//
// Output, one line per measurement:
//   latency,<bytes>,<random|in order>,<ns per load>
//   add,<n>,<bytes of the three arrays>,<ns per element>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>


struct alignas(64) Line {
    Line* next;
    char pad[64 - sizeof(Line*)];
};

using Clock = std::chrono::steady_clock;

static Line* volatile sink_line;
static volatile float sink_float;

double chase_ns(std::vector<Line>& lines, const std::vector<size_t>& order, int64_t steps) {
    const size_t n = order.size();
    for (size_t k = 0; k < n; k++) {
        lines[order[k]].next = &lines[order[(k + 1) % n]];
    }
    Line* p = &lines[order[0]];
    for (size_t k = 0; k < n; k++) {
        p = p->next;
    }
    double best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        auto t0 = Clock::now();
        for (int64_t s = 0; s < steps; s++) {
            p = p->next;
        }
        auto t1 = Clock::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / steps);
    }
    sink_line = p;
    return best;
}

double add_ns(int64_t n) {
    std::vector<float> a(n, 1.0f), b(n, 2.0f), out(n);
    const int64_t reps = std::max<int64_t>(3, 200'000'000 / n);
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
        auto t0 = Clock::now();
        for (int64_t r = 0; r < reps; r++) {
            float* o = out.data();
            const float* x = a.data();
            const float* y = b.data();
            for (int64_t i = 0; i < n; i++) {
                o[i] = x[i] + y[i];
            }
            sink_float = o[r % n];
        }
        auto t1 = Clock::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / (reps * n));
    }
    return best;
}

int main() {
    std::mt19937_64 rng(0);
    for (int64_t kb : {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 262144}) {
        const size_t n = static_cast<size_t>(kb) * 1024 / sizeof(Line);
        std::vector<Line> lines(n);
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), size_t{0});
        const int64_t steps = 20'000'000;
        std::printf("latency,%lld,in order,%.2f\n", static_cast<long long>(kb) * 1024, chase_ns(lines, order, steps));
        // Sattolo's shuffle: one random cycle through every line
        for (size_t k = n - 1; k > 0; k--) {
            std::uniform_int_distribution<size_t> pick(0, k - 1);
            std::swap(order[k], order[pick(rng)]);
        }
        std::printf("latency,%lld,random,%.2f\n", static_cast<long long>(kb) * 1024, chase_ns(lines, order, steps));
        std::fflush(stdout);
    }
    for (int64_t n : {1000, 4000, 16000, 64000, 250000, 1000000, 4000000, 16000000}) {
        std::printf("add,%lld,%lld,%.3f\n", static_cast<long long>(n), static_cast<long long>(12 * n), add_ns(n));
        std::fflush(stdout);
    }
}
