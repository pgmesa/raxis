// a + b on float32 with four ways of walking the operands. Single threaded, the
// output is preallocated, so only the loop is timed.
//
// modulo:   unravel the flat index with a division per dimension and element.
// strided:  an odometer over the outer dims and an inner loop that advances the
//           pointers by strides read at runtime.
// recursive: one call per outer dim instead of the odometer, same inner loop.
// kernels:  the same odometer, but the inner loop is picked from the strides
//           (all contiguous, or one operand with stride 0).
// flat:     out[i] = a[i] + b[i], only valid for contiguous operands.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#define NO_VECTOR __pragma(loop(no_vector))
#else
#define NOINLINE __attribute__((noinline))
#define NO_VECTOR _Pragma("GCC novector")  // GCC 14+: older ones ignore it and vectorize
#endif


struct Operands {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides_a;  // in elements
    std::vector<int64_t> strides_b;
    std::vector<int64_t> strides_out;
    const float* a;
    const float* b;
    float* out;
    int64_t numel;
};


static std::vector<int64_t> row_major_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t d = static_cast<int64_t>(shape.size()) - 1; d >= 0; --d) {
        strides[d] = stride;
        stride *= shape[d];
    }
    return strides;
}


NOINLINE static void add_modulo(const Operands& op) {
    const int64_t ndim = static_cast<int64_t>(op.shape.size());
    for (int64_t i = 0; i < op.numel; ++i) {
        int64_t rest = i, offset_a = 0, offset_b = 0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            const int64_t coord = rest % op.shape[d];
            rest /= op.shape[d];
            offset_a += coord * op.strides_a[d];
            offset_b += coord * op.strides_b[d];
        }
        op.out[i] = op.a[offset_a] + op.b[offset_b];
    }
}


NOINLINE static void inner_strided(const float* a, const float* b, float* out, int64_t n,
                                   int64_t sa, int64_t sb, int64_t so) {
    for (int64_t i = 0; i < n; ++i) {
        *out = *a + *b;
        a += sa;
        b += sb;
        out += so;
    }
}


NOINLINE static void inner_contiguous(const float* a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}


NOINLINE static void inner_contiguous_no_vector(const float* a, const float* b, float* out, int64_t n) {
    NO_VECTOR
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}


NOINLINE static void inner_scalar_b(const float* a, const float b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = a[i] + b;
}


NOINLINE static void inner_scalar_a(const float a, const float* b, float* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) out[i] = a + b[i];
}


// Mixed-radix odometer: bumps the outer dims without dividing. The inner dim
// runs inside `inner`, so the carry logic is paid once per row, not per element.
template <typename Inner>
static void for_each_row(const Operands& op, Inner inner) {
    const int64_t ndim = static_cast<int64_t>(op.shape.size());
    const int64_t n = op.shape[ndim - 1];
    const int64_t rows = op.numel / n;
    std::vector<int64_t> counter(ndim - 1, 0);
    const float* a = op.a;
    const float* b = op.b;
    float* out = op.out;

    for (int64_t row = 0; row < rows; ++row) {
        inner(a, b, out, n);
        for (int64_t d = ndim - 2; d >= 0; --d) {
            a += op.strides_a[d];
            b += op.strides_b[d];
            out += op.strides_out[d];
            if (++counter[d] < op.shape[d]) break;
            counter[d] = 0;
            a -= op.strides_a[d] * op.shape[d];
            b -= op.strides_b[d] * op.shape[d];
            out -= op.strides_out[d] * op.shape[d];
        }
    }
}


NOINLINE static void add_strided(const Operands& op) {
    const size_t last = op.shape.size() - 1;
    const int64_t sa = op.strides_a[last], sb = op.strides_b[last], so = op.strides_out[last];
    for_each_row(op, [=](const float* a, const float* b, float* out, int64_t n) {
        inner_strided(a, b, out, n, sa, sb, so);
    });
}


// One nested loop per dimension, written as a function that calls itself for
// the next dimension. No division either; the innermost dim is the same loop.
static void recurse(const Operands& op, size_t d, const float* a, const float* b, float* out) {
    const size_t last = op.shape.size() - 1;
    if (d == last) {
        inner_strided(a, b, out, op.shape[d], op.strides_a[d], op.strides_b[d], op.strides_out[d]);
        return;
    }
    for (int64_t i = 0; i < op.shape[d]; ++i) {
        recurse(op, d + 1, a, b, out);
        a += op.strides_a[d];
        b += op.strides_b[d];
        out += op.strides_out[d];
    }
}


NOINLINE static void add_recursive(const Operands& op) {
    recurse(op, 0, op.a, op.b, op.out);
}


NOINLINE static void add_kernels(const Operands& op) {
    const size_t last = op.shape.size() - 1;
    const int64_t sa = op.strides_a[last], sb = op.strides_b[last], so = op.strides_out[last];
    for_each_row(op, [=](const float* a, const float* b, float* out, int64_t n) {
        if (sa == 1 && sb == 1 && so == 1) inner_contiguous(a, b, out, n);
        else if (sa == 1 && sb == 0 && so == 1) inner_scalar_b(a, *b, out, n);
        else if (sa == 0 && sb == 1 && so == 1) inner_scalar_a(*a, b, out, n);
        else inner_strided(a, b, out, n, sa, sb, so);
    });
}


NOINLINE static void add_flat(const Operands& op) {
    inner_contiguous(op.a, op.b, op.out, op.numel);
}


NOINLINE static void add_flat_no_vector(const Operands& op) {
    inner_contiguous_no_vector(op.a, op.b, op.out, op.numel);
}


template <typename Loop>
static void report(const char* shape, const char* loop_name, Loop loop, const Operands& op, int runs) {
    std::vector<double> samples;
    samples.reserve(runs);
    volatile float sink = 0.0f;
    for (int run = 0; run < runs; ++run) {
        const auto t0 = std::chrono::steady_clock::now();
        loop(op);
        const auto t1 = std::chrono::steady_clock::now();
        sink = op.out[run % op.numel];
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::nth_element(samples.begin(), samples.begin() + runs / 2, samples.end());
    std::printf("%s,%s,%.4f\n", shape, loop_name, samples[runs / 2]);
    (void)sink;
}


static std::vector<float> fill(int64_t n, float start) {
    std::vector<float> values(n);
    for (int64_t i = 0; i < n; ++i) values[i] = start + static_cast<float>(i % 1024) * 0.5f;
    return values;
}


static void run_case(const char* name, std::vector<int64_t> shape, std::vector<int64_t> shape_b) {
    int64_t numel = 1, numel_b = 1;
    for (int64_t s : shape) numel *= s;
    for (int64_t s : shape_b) numel_b *= s;

    std::vector<float> a = fill(numel, 1.0f), b = fill(numel_b, 2.0f), out(numel);

    std::vector<int64_t> strides_b = row_major_strides(shape_b);
    for (size_t d = 0; d < shape.size(); ++d) {
        if (shape_b[d] == 1 && shape[d] != 1) strides_b[d] = 0;
    }

    const Operands op {shape, row_major_strides(shape), strides_b, row_major_strides(shape),
                       a.data(), b.data(), out.data(), numel};

    add_modulo(op);
    report(name, "modulo", add_modulo, op, 21);
    report(name, "strided", add_strided, op, 101);
    report(name, "recursive", add_recursive, op, 101);
    report(name, "kernels", add_kernels, op, 101);
    if (numel == numel_b) {
        report(name, "flat", add_flat, op, 101);
        report(name, "flat_no_vector", add_flat_no_vector, op, 101);
    }
}


int main() {
    run_case("1000x1000 + 1000x1000", {1000, 1000}, {1000, 1000});
    run_case("1000x1000 + 1000x1", {1000, 1000}, {1000, 1});
    run_case("1000x1000 + 1x1000", {1000, 1000}, {1, 1000});
    run_case("100x100x100 + 100x1x100", {100, 100, 100}, {100, 1, 100});
    return 0;
}
