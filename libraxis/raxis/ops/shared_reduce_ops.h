#pragma once

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

} // namespace raxis
