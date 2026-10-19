#pragma once

#include <cmath>
#include <cstdint>
#include <utility>

#include <raxis/ops/numeric_utils.h>


namespace raxis {

// From: aten/src/ATen/native/SharedReduceOps.h
// Accumulator structs for binary_kernel_reduce

namespace detail {

// a "wins" over b: NaN beats any number (it propagates), and ties go to the
// lower index, so the result does not depend on how threads split the work
template <typename scalar_t>
struct GreaterOrNan {
    bool operator()(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
        if (_isnan(a)) {
            if (_isnan(b)) {
                return idx_a < idx_b;
            }
            return true;
        }
        return (a == b) ? idx_a < idx_b : (a > b);
    }
};

template <typename scalar_t>
struct LessOrNan {
    bool operator()(scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
        if (_isnan(a)) {
            if (_isnan(b)) {
                return idx_a < idx_b;
            }
            return true;
        }
        return (a == b) ? idx_a < idx_b : (a < b);
    }
};

// Keeps (best value, its index) and returns the index
template <typename scalar_t, typename comp_t>
struct ArgReductionOps {
    using arg_t = std::pair<scalar_t, int64_t>;

    static arg_t reduce(arg_t acc, scalar_t value, int64_t idx) {
        return comp_t{}(acc.first, value, acc.second, idx) ? acc : arg_t(value, idx);
    }

    static arg_t combine(arg_t a, arg_t b) {
        return comp_t{}(a.first, b.first, a.second, b.second) ? a : b;
    }

    // Indices are relative to the sub-iterator; shift them by its offset
    static arg_t translate_idx(arg_t acc, int64_t base_idx) {
        return {acc.first, acc.second + base_idx};
    }

    static int64_t project(arg_t acc) {
        return acc.second;
    }
};

} // namespace detail

template <typename scalar_t>
struct ArgMaxOps : detail::ArgReductionOps<scalar_t, detail::GreaterOrNan<scalar_t>> {};

template <typename scalar_t>
struct ArgMinOps : detail::ArgReductionOps<scalar_t, detail::LessOrNan<scalar_t>> {};


// Running mean and M2 = sum((x - mean)^2) of the n elements seen so far
template <typename acc_scalar_t, typename index_t>
struct WelfordData {
    acc_scalar_t mean = 0;
    acc_scalar_t m2 = 0;
    index_t n = 0;
    // n as a float, which combine needs (n itself may not fit index_t there)
    acc_scalar_t nf = 0;
};

// Welford's algorithm: the variance in one pass, without the catastrophic
// cancellation of E[x^2] - E[x]^2. Each element updates the mean and M2
// incrementally; two partial results combine exactly (Chan et al.), which
// lets threads reduce chunks independently.
template <typename scalar_t, typename acc_scalar_t, typename index_t, typename res_t>
struct WelfordOps {
    using acc_t = WelfordData<acc_scalar_t, index_t>;

    acc_scalar_t correction;
    bool take_sqrt;

    acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
        index_t new_n = acc.n + 1;
        acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
        acc_scalar_t delta = data - acc.mean;
        acc_scalar_t new_mean = acc.mean + delta / new_nf;
        acc_scalar_t new_delta = data - new_mean;
        return {new_mean, acc.m2 + delta * new_delta, new_n, new_nf};
    }

    acc_t combine(acc_t a, acc_t b) const {
        if (a.nf == 0) return b;
        if (b.nf == 0) return a;
        acc_scalar_t delta = b.mean - a.mean;
        acc_scalar_t new_count = a.nf + b.nf;
        acc_scalar_t nb_over_n = b.nf / new_count;
        // n is no longer meaningful after combining: only nf is used
        return {a.mean + delta * nb_over_n,
                a.m2 + b.m2 + delta * delta * a.nf * nb_over_n,
                -1,
                new_count};
    }

    // var = M2 / (n - correction): correction 1 is the unbiased estimator.
    // With n <= correction the divisor is 0 (inf, or NaN for M2 = 0).
    res_t project(acc_t acc) const {
        acc_scalar_t divisor = acc.nf > correction ? acc.nf - correction : 0;
        acc_scalar_t var = acc.m2 / divisor;
        return static_cast<res_t>(take_sqrt ? std::sqrt(var) : var);
    }

    static acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
        return acc;
    }
};

} // namespace raxis
