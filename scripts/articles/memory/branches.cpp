// What a branch costs when the CPU guesses it right and when it guesses it
// wrong. The loop keeps the elements of x below a threshold (what x[x < t]
// does), written two ways:
//
// branch:    if (x[i] < t) out[k++] = x[i];
// no branch: out[k] = x[i]; k += (x[i] < t);
//
// x holds 1M int32 numbers from 0 to 99 (4 MB, read in order, so memory
// does not decide the time), and t picks how many are kept: t = 50 keeps
// half. "random" keeps them in a random order, "sorted" first sorts x, so
// the branch goes one way for the first half of the array and the other way
// for the rest.
//
// "repeat" runs the random half on smaller arrays, each one many times: the
// branch predictor remembers the history of the branch, and it can learn
// the whole sequence of a small array.
//
// Output, one line per measurement:
//   filter,<random|sorted>,<percent kept>,<branch|no branch>,<ns per element>
//   repeat,<n>,branch,<ns per element>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>


using Clock = std::chrono::steady_clock;

static volatile int64_t sink;

// The two loops are kept out of line, so the compiler sees them alone
#if defined(_MSC_VER)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

NOINLINE int64_t filter_branch(const int32_t* x, int32_t* out, int64_t n, int32_t t) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) {
        if (x[i] < t) {
            out[k++] = x[i];
        }
    }
    return k;
}

NOINLINE int64_t filter_no_branch(const int32_t* x, int32_t* out, int64_t n, int32_t t) {
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) {
        // Always store, and move the end of the output by 0 or 1: the
        // comparison becomes a number instead of a jump
        out[k] = x[i];
        k += x[i] < t;
    }
    return k;
}

template<typename F>
double ns_per_element(F filter, const std::vector<int32_t>& x, std::vector<int32_t>& out, int32_t t) {
    const int64_t n = static_cast<int64_t>(x.size());
    const int reps = static_cast<int>(std::max<int64_t>(1, 128'000'000 / n));
    double best = 1e30;
    for (int run = 0; run < 5; run++) {
        auto t0 = Clock::now();
        int64_t kept = 0;
        for (int r = 0; r < reps; r++) {
            kept += filter(x.data(), out.data(), n, t);
        }
        auto t1 = Clock::now();
        sink = kept;
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / (static_cast<double>(reps) * n));
    }
    return best;
}

int main() {
    std::mt19937 rng(0);
    auto random_numbers = [&](int64_t n) {
        std::vector<int32_t> x(n);
        for (auto& v : x) {
            v = static_cast<int32_t>(rng() % 100);
        }
        return x;
    };

    // 1M numbers: too many for the predictor to learn the sequence
    std::vector<int32_t> x = random_numbers(1 << 20);
    std::vector<int32_t> out(x.size());
    for (int32_t t : {0, 10, 25, 50, 75, 90, 100}) {
        std::printf("filter,random,%d,branch,%.3f\n", t, ns_per_element(filter_branch, x, out, t));
        std::printf("filter,random,%d,no branch,%.3f\n", t, ns_per_element(filter_no_branch, x, out, t));
        std::fflush(stdout);
    }
    std::sort(x.begin(), x.end());
    std::printf("filter,sorted,50,branch,%.3f\n", ns_per_element(filter_branch, x, out, 50));
    std::printf("filter,sorted,50,no branch,%.3f\n", ns_per_element(filter_no_branch, x, out, 50));

    // The same random half, from smaller arrays run again and again: the
    // predictor learns the sequence of a small one
    for (int64_t n : {4096, 16384, 65536, 262144}) {
        std::vector<int32_t> small = random_numbers(n);
        std::vector<int32_t> small_out(n);
        std::printf("repeat,%lld,branch,%.3f\n", static_cast<long long>(n), ns_per_element(filter_branch, small, small_out, 50));
        std::fflush(stdout);
    }
}
