
#include <raxis/macros.h>

#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/cpu/reduce.h>

#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>


namespace raxis {

namespace {

inline int64_t divup(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

inline int64_t ceil_log2(int64_t n) {
    if (n <= 1) return 1;
    int64_t r = 0;
    n -= 1;
    while (n > 0) {
        n >>= 1;
        ++r;
    }
    return r;
}


// ============================================================================
// Load Policies
// ============================================================================
//
// The LoadPolicy abstraction decouples "how to read one element" from the
// summation algorithm. This lets multi_row_sum / row_sum work identically
// for normal sum, nansum, and SIMD paths (only the load changes).
//
// Each policy provides:
//   static scalar_t load(const char* data, int64_t stride, int64_t index)
//   static constexpr int64_t memsize()

/// Normal scalar load
template <typename scalar_t>
struct LoadPolicy {
    static constexpr int64_t memsize() { return sizeof(scalar_t); }

    static scalar_t load(const char* RAXIS_RESTRICT data,
                         int64_t stride, int64_t index) {
        return *reinterpret_cast<const scalar_t*>(data + index * stride);
    }
};

/// NaN-aware scalar load: replaces NaN with 0 before accumulation
template <typename scalar_t>
struct NanLoadPolicy {
    static constexpr int64_t memsize() { return sizeof(scalar_t); }

    static scalar_t load(const char* RAXIS_RESTRICT data,
                         int64_t stride, int64_t index) {
        auto val = LoadPolicy<scalar_t>::load(data, stride, index);
        if constexpr (std::is_floating_point_v<scalar_t>) {
            return std::isnan(val) ? scalar_t(0) : val;
        } else {
            return val;
        }
    }
};

/// Select the right load policy based on ignore_nan
template <typename scalar_t, bool ignore_nan>
using SumLoadPolicy = std::conditional_t<
    ignore_nan, NanLoadPolicy<scalar_t>, LoadPolicy<scalar_t>>;


// ============================================================================
// Store policy
// ============================================================================
//
// Accumulates into the output (+=) rather than assigning. This is critical
// because parallel_reduce's two-pass approach may call the kernel multiple
// times for the same output element. First each thread reduces its chunk,
// then the partial results are combined.

template <typename scalar_t>
struct StoreAccumulate {
    static void store(char* RAXIS_RESTRICT data, int64_t stride,
                      int64_t index, scalar_t value) {
        auto* ptr = reinterpret_cast<scalar_t*>(data + index * stride);
        *ptr += value;
    }
};

/// Store an array of values (for multi_row_sum with nrows > 1)
template <typename scalar_t, int64_t nrows>
void store_array(char* RAXIS_RESTRICT data, int64_t stride, int64_t index,
                 const std::array<scalar_t, nrows>& values) {
    char* base = data + stride * index;
    for (int64_t k = 0; k < nrows; ++k) {
        StoreAccumulate<scalar_t>::store(base, stride, k, values[k]);
    }
}


// ============================================================================
// Cascade (pairwise) summation
// ============================================================================
//
// This algorithm sums elements with O(log n) rounding error instead of O(n).
// It works by maintaining multiple levels of partial sums. Elements are first
// accumulated into level 0. Every `level_step` elements, level 0 is flushed
// into level 1, which flushes into level 2, etc., like a carry chain in
// binary addition.
//
// The key insight: at each level, we're only adding numbers of similar
// magnitude, which minimizes the rounding error from each individual addition.
//
// This is done in a single linear pass with O(1) extra storage (4 levels
// of accumulators). The `nrows` template parameter allows simultaneous
// processing of multiple independent rows for instruction-level parallelism.
//
// PyTorch's `multi_row_sum` in SumKernel.cpp is the reference implementation.

template <typename scalar_t, int64_t nrows, typename Policy>
std::array<scalar_t, nrows> multi_row_sum(
    const char* RAXIS_RESTRICT in_data,
    const int64_t row_stride,    // stride between consecutive reduction elements
    const int64_t col_stride,    // stride between consecutive rows (output elements)
    const int64_t size)          // number of reduction elements
{
    constexpr int64_t num_levels = 4;

    // Compute the step size for each level.
    // level_power = max(4, ceil_log2(size) / num_levels)
    // This means each level accumulates at least 16 elements before flushing,
    // and the total number of levels stays at 4 regardless of input size.
    const int64_t level_power =
        std::max(int64_t(4), ceil_log2(size) / num_levels);
    const int64_t level_step = (int64_t(1) << level_power);
    const int64_t level_mask = level_step - 1;

    // Accumulators: acc[level][row]
    std::array<std::array<scalar_t, nrows>, num_levels> acc{};
    for (auto& row : acc) {
        row.fill(scalar_t(0));
    }

    // Main loop: process full chunks of level_step elements
    int64_t i = 0;
    for (; i + level_step <= size;) {
        // Accumulate level_step elements into level 0
        for (int64_t j = 0; j < level_step; ++j, ++i) {
            const char* sum_base = in_data + i * row_stride;
            for (int64_t k = 0; k < nrows; ++k) {
                acc[0][k] += Policy::load(sum_base, col_stride, k);
            }
        }

        // Cascade: flush lower levels into higher levels when appropriate.
        // This is like carry propagation in binary addition.
        // After processing i elements:
        //   - flush level 0 → level 1 always (every level_step elements)
        //   - flush level 1 → level 2 every level_step^2 elements
        //   - flush level 2 → level 3 every level_step^3 elements
        for (int64_t lev = 1; lev < num_levels; ++lev) {
            for (int64_t k = 0; k < nrows; ++k) {
                acc[lev][k] += acc[lev - 1][k];
                acc[lev - 1][k] = scalar_t(0);
            }

            // Check if this level should propagate further.
            // If bit (lev * level_power) of i is set, stop here.
            const auto mask = (level_mask << (lev * level_power));
            if ((i & mask) != 0) {
                break;
            }
        }
    }

    // Tail: remaining elements that don't fill a complete chunk
    for (; i < size; ++i) {
        const char* sum_base = in_data + i * row_stride;
        for (int64_t k = 0; k < nrows; ++k) {
            acc[0][k] += Policy::load(sum_base, col_stride, k);
        }
    }

    // Final reduction: collapse all levels into level 0
    for (int64_t lev = 1; lev < num_levels; ++lev) {
        for (int64_t k = 0; k < nrows; ++k) {
            acc[0][k] += acc[lev][k];
        }
    }

    return acc[0];
}

/// Single-row cascade sum with ILP (instruction-level parallelism).
/// Treats the input as a 2D array of shape (size/ilp_factor, ilp_factor)
/// and sums the "columns" independently, then combines them.
/// This gives the CPU's out-of-order engine ilp_factor independent
/// dependency chains instead of one.
template <typename scalar_t, typename Policy>
scalar_t row_sum(
    const char* RAXIS_RESTRICT in_data,
    const int64_t in_stride,
    const int64_t size)
{
    constexpr int64_t ilp_factor = 4;

    // Reshape as (size/ilp_factor, ilp_factor) and cascade-sum the "columns"
    const int64_t size_ilp = size / ilp_factor;
    auto partial_sums = multi_row_sum<scalar_t, ilp_factor, Policy>(
        in_data,
        in_stride * ilp_factor,   // row_stride: jump ilp_factor elements per step
        in_stride,                // col_stride: consecutive elements are the "rows"
        size_ilp);

    // Handle remaining elements (size % ilp_factor)
    for (int64_t i = size_ilp * ilp_factor; i < size; ++i) {
        partial_sums[0] += Policy::load(in_data, in_stride, i);
    }

    // Combine the ilp_factor partial sums
    for (int64_t k = 1; k < ilp_factor; ++k) {
        partial_sums[0] += partial_sums[k];
    }

    return partial_sums[0];
}


template <typename F>
void outer_loop(char* data[2], const int64_t outer_strides[2],
                int64_t size1, F&& f) {
    for (int64_t j = 0; j < size1; ++j) {
        f();
        data[0] += outer_strides[0];
        data[1] += outer_strides[1];
    }
}

template <typename scalar_t, bool ignore_nan>
void scalar_inner_sum(
    char* RAXIS_RESTRICT data[2],
    int64_t in_strides[2],
    int64_t out_stride,
    int64_t size0,
    int64_t size1)
{
    using Policy = SumLoadPolicy<scalar_t, ignore_nan>;

    for (int64_t j = 0; j < size1; ++j) {
        const char* row_in = data[1] + j * in_strides[1];
        scalar_t acc = row_sum<scalar_t, Policy>(row_in, in_strides[0], size0);
        StoreAccumulate<scalar_t>::store(data[0], out_stride, j, acc);
    }
}

template <typename scalar_t, bool ignore_nan>
void scalar_outer_sum(
    char* RAXIS_RESTRICT data[2],
    int64_t in_strides[2],
    int64_t out_stride,
    int64_t size0,
    int64_t size1)
{
    using Policy = SumLoadPolicy<scalar_t, ignore_nan>;

    constexpr int64_t nrows = 4;
    int64_t j = 0;

    // Process nrows output elements at a time
    for (; j + nrows <= size1; j += nrows) {
        const char* row_in = data[1] + j * in_strides[1];
        auto sums = multi_row_sum<scalar_t, nrows, Policy>(
            row_in,
            in_strides[0],    // row_stride: step along reduction dim
            in_strides[1],    // col_stride: step between output elements
            size0);

        store_array<scalar_t, nrows>(data[0], out_stride, j, sums);
    }

    // Tail
    for (; j < size1; ++j) {
        const char* row_in = data[1] + j * in_strides[1];
        scalar_t acc = row_sum<scalar_t, Policy>(row_in, in_strides[0], size0);
        StoreAccumulate<scalar_t>::store(data[0], out_stride, j, acc);
    }
}

/// Elementwise sum: not a true reduction (both strides nonzero).
/// Used by parallel_reduce's combine pass.
template <typename scalar_t, bool ignore_nan>
void elementwise_sum(
    char* data[2],
    const int64_t* strides,
    int64_t size0,
    int64_t size1)
{
    using Policy = SumLoadPolicy<scalar_t, ignore_nan>;

    int64_t outer_strides[] = { strides[2], strides[3] };
    char* data_copy[2] = { data[0], data[1] };

    outer_loop(data_copy, outer_strides, size1, [&] {
        char* out_ptr = data_copy[0];
        const char* in_ptr = data_copy[1];
        for (int64_t i = 0; i < size0; ++i) {
            scalar_t val = Policy::load(in_ptr, strides[1], 0);
            auto* out = reinterpret_cast<scalar_t*>(out_ptr);
            *out += val;
            out_ptr += strides[0];
            in_ptr += strides[1];
        }
    });
}

template <typename scalar_t, bool ignore_nan>
void sum_kernel_impl(TensorIterator& iter) {
    iter.get_output(0).zero_();
    iter.parallel_reduce(
        [](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
            int64_t in_strides[]  = { strides[1], strides[3] };
            int64_t out_strides[] = { strides[0], strides[2] };

            // Case 1: not a true reduction (parallel_reduce combine pass)
            if (out_strides[0] != 0 && out_strides[1] != 0) {
                elementwise_sum<scalar_t, ignore_nan>(data, strides, size0, size1);
                return;
            }

            // Normalize: ensure dim 0 is the reduced dim (out_stride == 0)
            if (out_strides[0] != 0 && out_strides[1] == 0) {
                std::swap(in_strides[0], in_strides[1]);
                std::swap(out_strides[0], out_strides[1]);
                std::swap(size0, size1);
            }

            int64_t out_stride = out_strides[1];
            RAXIS_INTERNAL_ASSERT(out_strides[0] == 0);

            bool inner_contiguous =
                (in_strides[0] == static_cast<int64_t>(sizeof(scalar_t)));
            bool outer_contiguous =
                (in_strides[1] == static_cast<int64_t>(sizeof(scalar_t)));

            if (inner_contiguous) {
                // Future: vectorized_inner_sum with SIMD LoadPolicy
                scalar_inner_sum<scalar_t, ignore_nan>(
                    data, in_strides, out_stride, size0, size1);
            } else if (outer_contiguous) {
                // Future: vectorized_outer_sum with SIMD LoadPolicy
                scalar_outer_sum<scalar_t, ignore_nan>(
                    data, in_strides, out_stride, size0, size1);
            } else if (in_strides[0] < in_strides[1]) {
                scalar_inner_sum<scalar_t, ignore_nan>(
                    data, in_strides, out_stride, size0, size1);
            } else {
                scalar_outer_sum<scalar_t, ignore_nan>(
                    data, in_strides, out_stride, size0, size1);
            }
        }
    );
}

void sum_kernel(TensorIterator& iter) {
    if (iter.common_dtype().is_integral(/*include_bool=*/true)) {
        iter.common_dtype().dispatch_integral_types(
            [&]<typename scalar_t>(std::type_identity<scalar_t>) {
                cpu_reduce_kernel<scalar_t>(
                    iter,
                    [](scalar_t a, scalar_t b) -> scalar_t { return a + b; },
                    /*ident=*/0);
            },
            /*include_bool=*/true
        );
        return;
    }

    iter.common_dtype().dispatch_floating_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) {
            sum_kernel_impl<scalar_t, /*ignore_nan=*/false>(iter);
        }
    );
}

void nansum_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) {
            sum_kernel_impl<scalar_t, /*ignore_nan=*/true>(iter);
        }
    );
}

} // anonymous namespace

REGISTER_CPU_DISPATCH(sum_stub, &sum_kernel);
REGISTER_CPU_DISPATCH(nansum_stub, &nansum_kernel);

} // namespace raxis