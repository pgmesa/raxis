// Summing an array of float32 in C++, written five ways, for the article on
// reductions:
//
// one float:      float acc; acc += x[i]                (the obvious loop)
// four floats:    four float accumulators, one per i % 4
// one vector:     one AVX2 register of 8 accumulators
// four vectors:   four AVX2 registers, 32 accumulators (what PyTorch's
//                 row_sum does, before its cascade)
// one double:     double acc; acc += x[i]
//
// Speed: x holds 64Ki floats (256 KiB, in the L2 after the first run), so
// the memory is not the limit; each loop runs 2000 times and the minimum of
// 5 runs is kept.
// Error: x holds n floats from [0, 1), for n = 1M and 16M, and each sum is
// compared with the exact sum (accumulated in long double pairs, see
// exact_sum). The ones of torch.ones(2**25) show the worst case.
//
// Output, one line per measurement:
//   speed,<variant>,<ns per element>
//   error,<n>,<variant>,<sum>,<relative error>

#include <immintrin.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>


using Clock = std::chrono::steady_clock;

static volatile float sink;

#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

NOINLINE float one_float(const float* x, int64_t n) {
    float acc = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        acc += x[i];
    }
    return acc;
}

NOINLINE float four_floats(const float* x, int64_t n) {
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) {
        acc[0] += x[i];
        acc[1] += x[i + 1];
        acc[2] += x[i + 2];
        acc[3] += x[i + 3];
    }
    for (; i < n; i++) {
        acc[0] += x[i];
    }
    return (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

static float horizontal_sum(__m256 v) {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, v);
    float s = 0.0f;
    for (float f : lanes) {
        s += f;
    }
    return s;
}

NOINLINE float one_vector(const float* x, int64_t n) {
    __m256 acc = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(x + i));
    }
    float s = horizontal_sum(acc);
    for (; i < n; i++) {
        s += x[i];
    }
    return s;
}

NOINLINE float four_vectors(const float* x, int64_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(x + i));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(x + i + 8));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(x + i + 16));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(x + i + 24));
    }
    float s = horizontal_sum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < n; i++) {
        s += x[i];
    }
    return s;
}

NOINLINE float one_double(const float* x, int64_t n) {
    double acc = 0.0;
    for (int64_t i = 0; i < n; i++) {
        acc += x[i];
    }
    return static_cast<float>(acc);
}

// Exact enough for the reference: pairwise in double, error ~1e-16 relative
static double exact_sum(const float* x, int64_t n) {
    if (n <= 64) {
        double s = 0.0;
        for (int64_t i = 0; i < n; i++) {
            s += x[i];
        }
        return s;
    }
    return exact_sum(x, n / 2) + exact_sum(x + n / 2, n - n / 2);
}

using SumFn = float (*)(const float*, int64_t);

struct Variant {
    const char* name;
    SumFn f;
};

static const Variant variants[] = {
    {"one float", one_float},
    {"four floats", four_floats},
    {"one vector", one_vector},
    {"four vectors", four_vectors},
    {"one double", one_double},
};

int main() {
    std::mt19937 gen(0);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    const int64_t n_speed = 64 * 1024;
    std::vector<float> small(n_speed);
    for (auto& v : small) {
        v = dist(gen);
    }
    const int reps = 2000;
    for (const Variant& v : variants) {
        double best = 1e30;
        for (int run = 0; run < 5; run++) {
            auto t0 = Clock::now();
            for (int r = 0; r < reps; r++) {
                sink = v.f(small.data(), n_speed);
            }
            double ns = std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
            best = std::min(best, ns / (double(reps) * n_speed));
        }
        std::printf("speed,%s,%.4f\n", v.name, best);
    }

    for (int64_t n : {int64_t(1) << 20, int64_t(16) << 20}) {
        std::vector<float> x(n);
        for (auto& v : x) {
            v = dist(gen);
        }
        const double exact = exact_sum(x.data(), n);
        for (const Variant& v : variants) {
            float s = v.f(x.data(), n);
            std::printf("error,%lld,%s,%.1f,%.3e\n", static_cast<long long>(n), v.name, s,
                        std::abs(s - exact) / exact);
        }
    }

    // Ones: every partial sum is an integer, and float32 holds integers
    // exactly only up to 2^24 = 16,777,216
    const int64_t n_ones = int64_t(1) << 25;
    std::vector<float> ones(n_ones, 1.0f);
    for (const Variant& v : variants) {
        float s = v.f(ones.data(), n_ones);
        std::printf("ones,%lld,%s,%.1f\n", static_cast<long long>(n_ones), v.name, s);
    }
}
