// Runs multitensor's own code: does Tensor::get read the right element of an expanded
// tensor, and how long does its add take with and without broadcasting?
// Tests with constant tensors (as its own) cannot see the wrong offsets.

#include "tensor.hpp"
#include "functional.hpp"
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <memory>
#include <vector>

static std::shared_ptr<float[]> iota(int n) {
    std::shared_ptr<float[]> d(new float[n]);
    for (int i = 0; i < n; i++) d[i] = (float)i;
    return d;
}

// Row-major strides of an expanded tensor, the right value at flat index i
static float expected(const std::vector<int>& src, const std::vector<int>& dst, int idx, const float* data) {
    int n = (int)dst.size(), off = 0, stride = 1;
    std::vector<int> coord(n);
    for (int d = n - 1; d >= 0; d--) { coord[d] = idx % dst[d]; idx /= dst[d]; }
    std::vector<int> pad(n, 1);
    for (size_t k = 0; k < src.size(); k++) pad[n - src.size() + k] = src[k];
    for (int d = n - 1; d >= 0; d--) { if (pad[d] != 1) off += coord[d] * stride; stride *= pad[d]; }
    return data[off];
}

static void check(std::vector<int> src, std::vector<int> dst) {
    int n = 1, m = 1;
    for (int s : src) n *= s;
    for (int s : dst) m *= s;
    Tensor<float> t(iota(n), src);
    Tensor<float> e = t.expand(dst);
    int wrong = 0;
    for (int i = 0; i < m; i++) if (e.get(i) != expected(src, dst, i, t.data.get())) wrong++;
    printf("expand [");
    for (int s : src) printf("%d ", s);
    printf("] -> [");
    for (int s : dst) printf("%d ", s);
    printf("]: %d/%d wrong | strides [", wrong, m);
    for (int s : e.strides) printf("%d ", s);
    printf("]\n");
}

static double time_add(std::vector<int> sa, std::vector<int> sb, int runs) {
    Tensor<float> a = Tensor<float>::ones(sa), b = Tensor<float>::ones(sb);
    std::vector<double> s;
    for (int r = 0; r < runs; r++) {
        auto t0 = std::chrono::steady_clock::now();
        Tensor<float> c = a + b;
        auto t1 = std::chrono::steady_clock::now();
        s.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(s.begin(), s.end());
    return s[runs / 2];
}

int main() {
    check({3, 1}, {3, 4});
    check({1, 4}, {3, 4});
    check({4}, {3, 4});
    check({2, 1, 3}, {2, 4, 3});
    check({3, 4}, {2, 3, 4});
    printf("add 1000x1000 + 1000x1000: %.3f ms\n", time_add({1000, 1000}, {1000, 1000}, 21));
    printf("add 1000x1000 + 1000x1: %.3f ms\n", time_add({1000, 1000}, {1000, 1}, 11));
    printf("add 1000x1000 + 1x1000: %.3f ms\n", time_add({1000, 1000}, {1, 1000}, 11));
    return 0;
}
