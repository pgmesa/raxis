#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <utility>

#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/util/parallel.h>


namespace raxis {

inline namespace CPU_CAPABILITY {

// From: aten/src/ATen/native/cpu/moments_utils.h
//
// Mean and variance of a contiguous row in one pass, with Welford's update
//   delta = x - m1;  m1 += delta / n;  m2 += delta * (x - m1)
// which does not lose precision like sum(x^2) - sum(x)^2 / n. Each vector
// lane keeps its own Welford state over chunks of kChunkSize vectors; the
// chunk results are merged pairwise (cascade sum), like a binary counter: a
// stack level k holds 2^k chunks and two equal levels merge into the next
// one. The rounding error then grows with log(n) instead of n.

constexpr int64_t kChunkSize = 16;

// Merges the moments of two groups: (m0_add, m1_add, m2_add) into (m0, m1, m2)
template <typename T>
void add_moments(int64_t m0_add, const T& m1_add, const T& m2_add, int64_t& m0, T& m1, T& m2) {
    const int64_t n = m0 + m0_add;
    const T c = n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
    const T delta = m1_add - m1;
    m1 += c * delta;
    m2 += m2_add + delta * delta * c * static_cast<T>(m0);
    m0 = n;
}

template <typename T>
inline void add_moments_vec(
    int64_t m0_add, const vec::Vectorized<T>& m1_add, const vec::Vectorized<T>& m2_add,
    int64_t& m0, vec::Vectorized<T>& m1, vec::Vectorized<T>& m2
) {
    using Vec = vec::Vectorized<T>;
    const int64_t n = m0 + m0_add;
    const T c = n == 0 ? static_cast<T>(0) : static_cast<T>(m0_add) / static_cast<T>(n);
    const Vec c_vec(c);
    const Vec delta = m1_add - m1;
    const Vec m2_tmp = m2 + m2_add;
    const Vec c_vec_delta = c_vec * delta;
    const Vec m0_delta = delta * Vec(static_cast<T>(m0));
    m1 = m1 + c_vec_delta;
    m2 = vec::fmadd(m0_delta, c_vec_delta, m2_tmp);
    m0 = n;
}

// Welford over m0 consecutive vectors starting at X, merged into the stack
// base. c_vecs[j] = 1 / (j + 1) avoids a division per vector.
template <typename T>
inline void update_moments_vec(
    int64_t m0, const T* X, const std::array<vec::Vectorized<T>, kChunkSize>& c_vecs,
    int64_t& m0_stk0, vec::Vectorized<T>& m1_stk0, vec::Vectorized<T>& m2_stk0
) {
    using Vec = vec::Vectorized<T>;
    Vec m1_vec(T(0));
    Vec m2_vec(T(0));
    for (int64_t j = 0; j < m0; ++j) {
        const Vec x_vec = Vec::loadu(X + j * Vec::size());
        const Vec delta_vec = x_vec - m1_vec;
        m1_vec = vec::fmadd(c_vecs[j], delta_vec, m1_vec);
        m2_vec = vec::fmadd(delta_vec, x_vec - m1_vec, m2_vec);
    }
    add_moments_vec(m0, m1_vec, m2_vec, m0_stk0, m1_stk0, m2_stk0);
}

inline int64_t ceil_log2(int64_t x) {
    return x <= 2 ? 1 : std::bit_width(static_cast<uint64_t>(x) - 1);
}

template <typename T, int64_t kMaxDepth>
std::pair<T, T> rowwise_moments_impl(const T* X, int64_t N, int64_t ddof) {
    using Vec = vec::Vectorized<T>;
    constexpr int64_t kVecSize = Vec::size();
    const int64_t n = N / kVecSize;
    const int64_t m = divup(n, kChunkSize);
    const int64_t depth = ceil_log2(m);

    static const std::array<Vec, kChunkSize> c_vecs = [] {
        std::array<Vec, kChunkSize> result;
        for (int64_t i = 0; i < kChunkSize; ++i) {
            result[i] = Vec(T(1) / static_cast<T>(i + 1));
        }
        return result;
    }();

    const Vec zero_vec(T(0));
    std::array<int64_t, kMaxDepth> m0_stk = {{0}};
    std::array<Vec, kMaxDepth> m1_stk;
    std::array<Vec, kMaxDepth> m2_stk;
    m1_stk.fill(zero_vec);
    m2_stk.fill(zero_vec);

    for (int64_t i = 0; i < m; ++i) {
        const T* X_ptr = X + i * kChunkSize * kVecSize;
        const int64_t m0 = std::min(kChunkSize, n - i * kChunkSize);
        update_moments_vec(m0, X_ptr, c_vecs, m0_stk[0], m1_stk[0], m2_stk[0]);

        // Carry: after chunk i + 1, merge one level up per trailing zero bit
        int64_t mask = i + 1;
        for (int64_t j = 1; j < depth && (mask & 1) == 0; ++j) {
            add_moments_vec(m0_stk[j - 1], m1_stk[j - 1], m2_stk[j - 1], m0_stk[j], m1_stk[j], m2_stk[j]);
            m0_stk[j - 1] = 0;
            m1_stk[j - 1] = zero_vec;
            m2_stk[j - 1] = zero_vec;
            mask >>= 1;
        }
    }
    for (int64_t i = 1; i < depth; ++i) {
        add_moments_vec(m0_stk[i], m1_stk[i], m2_stk[i], m0_stk[0], m1_stk[0], m2_stk[0]);
    }

    std::array<T, kVecSize> m1_arr{};
    std::array<T, kVecSize> m2_arr{};
    m1_stk[0].store(m1_arr.data());
    m2_stk[0].store(m2_arr.data());

    // The scalar tail, then the lanes merged into it
    int64_t m0 = 0;
    T m1 = 0;
    T m2 = 0;
    for (int64_t i = n * kVecSize; i < N; ++i) {
        const T x = X[i];
        const T delta = x - m1;
        ++m0;
        m1 += delta / static_cast<T>(m0);
        m2 += delta * (x - m1);
    }
    for (int64_t i = 0; i < kVecSize; ++i) {
        add_moments(n, m1_arr[i], m2_arr[i], m0, m1, m2);
    }
    return {m1, m2 / static_cast<T>(N - ddof)};
}

// (mean, variance) of X[0, N), the variance divided by N - ddof
template <typename T>
std::pair<T, T> rowwise_moments(const T* X, int64_t N, int64_t ddof = 0) {
    constexpr int64_t kVecSize = vec::Vectorized<T>::size();
    const int64_t depth = ceil_log2(divup(N / kVecSize, kChunkSize));
    if (depth <= 4) {
        return rowwise_moments_impl<T, 4>(X, N, ddof);
    } else if (depth <= 8) {
        return rowwise_moments_impl<T, 8>(X, N, ddof);
    } else if (depth <= 16) {
        return rowwise_moments_impl<T, 16>(X, N, ddof);
    } else if (depth <= 32) {
        return rowwise_moments_impl<T, 32>(X, N, ddof);
    }
    return rowwise_moments_impl<T, 64>(X, N, ddof);
}

} // namespace CPU_CAPABILITY

} // namespace raxis
