
#include <raxis/macros.h>

#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/cpu/reduce.h>

#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>


namespace raxis {

namespace {

template <typename F>
void outer_loop(char* data[2], const int64_t outer_strides[2], int64_t size1, F&& f) {
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
    int64_t size1
) {
    for (int64_t j = 0; j < size1; ++j) {
        const char* row_in = data[1] + j * in_strides[1];
        scalar_t acc = 0;
        for (int64_t i = 0; i < size0; ++i) {
            scalar_t val = *reinterpret_cast<const scalar_t*>(
                row_in + i * in_strides[0]);
            if constexpr (ignore_nan) {
                if constexpr (std::is_floating_point_v<scalar_t>) {
                    acc += std::isnan(val) ? scalar_t(0) : val;
                } else {
                    acc += val;
                }
            } else {
                acc += val;
            }
        }
        // Accumulate into output (not assign, output may already contain 
        // partial sums from parallel_reduce's two-pass approach)
        auto* out = reinterpret_cast<scalar_t*>(data[0] + j * out_stride);
        *out += acc;
    }
}

template <typename scalar_t, bool ignore_nan>
void scalar_outer_sum(
    char* RAXIS_RESTRICT data[2],
    int64_t in_strides[2],
    int64_t out_stride,
    int64_t size0,
    int64_t size1
) {
    // Process nrows output elements at a time for ILP
    constexpr int64_t nrows = 4;
    int64_t j = 0;

    for (; j + nrows <= size1; j += nrows) {
        scalar_t accs[nrows] = {};

        for (int64_t i = 0; i < size0; ++i) {
            for (int64_t k = 0; k < nrows; ++k) {
                const char* ptr = data[1] + (j + k) * in_strides[1] 
                                           + i * in_strides[0];
                scalar_t val = *reinterpret_cast<const scalar_t*>(ptr);
                if constexpr (ignore_nan && std::is_floating_point_v<scalar_t>) {
                    accs[k] += std::isnan(val) ? scalar_t(0) : val;
                } else {
                    accs[k] += val;
                }
            }
        }

        for (int64_t k = 0; k < nrows; ++k) {
            auto* out = reinterpret_cast<scalar_t*>(
                data[0] + (j + k) * out_stride);
            *out += accs[k];
        }
    }

    // Tail: remaining output elements
    for (; j < size1; ++j) {
        scalar_t acc = 0;
        for (int64_t i = 0; i < size0; ++i) {
            const char* ptr = data[1] + j * in_strides[1] 
                                       + i * in_strides[0];
            scalar_t val = *reinterpret_cast<const scalar_t*>(ptr);
            if constexpr (ignore_nan && std::is_floating_point_v<scalar_t>) {
                acc += std::isnan(val) ? scalar_t(0) : val;
            } else {
                acc += val;
            }
        }
        auto* out = reinterpret_cast<scalar_t*>(data[0] + j * out_stride);
        *out += acc;
    }
}

template <typename scalar_t, bool ignore_nan>
void elementwise_sum(
    char* data[2],
    const int64_t* strides,
    int64_t size0,
    int64_t size1
) {
    int64_t outer_strides[] = { strides[2], strides[3] };
    char* data_copy[2] = { data[0], data[1] };

    outer_loop(data_copy, outer_strides, size1, [&] {
        char* out_ptr = data_copy[0];
        char* in_ptr = data_copy[1];
        for (int64_t i = 0; i < size0; ++i) {
            auto* out = reinterpret_cast<scalar_t*>(out_ptr);
            scalar_t val = *reinterpret_cast<const scalar_t*>(in_ptr);
            if constexpr (ignore_nan && std::is_floating_point_v<scalar_t>) {
                *out += std::isnan(val) ? scalar_t(0) : val;
            } else {
                *out += val;
            }
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

            // Case 2: contiguous inner reduction
            // Input is contiguous along the reduced dim (dim 0)
            // This is the most common case for row-major data
            bool inner_contiguous = 
                (in_strides[0] == static_cast<int64_t>(sizeof(scalar_t)));
            
            // Case 3: contiguous outer reduction
            // Input is contiguous along the output dim (dim 1) 
            bool outer_contiguous = 
                (in_strides[1] == static_cast<int64_t>(sizeof(scalar_t)));

            if (inner_contiguous) {
                scalar_inner_sum<scalar_t, ignore_nan>(data, in_strides, out_stride, size0, size1);
            } else if (outer_contiguous) {
                scalar_outer_sum<scalar_t, ignore_nan>(data, in_strides, out_stride, size0, size1);
            } else if (in_strides[0] < in_strides[1]) {
                // Non-contiguous: pick the variant that walks the smaller stride
                // in the inner loop (better cache behavior)
                scalar_inner_sum<scalar_t, ignore_nan>(data, in_strides, out_stride, size0, size1);
            } else {
                scalar_outer_sum<scalar_t, ignore_nan>(data, in_strides, out_stride, size0, size1);
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
                    /*ident=*/0
                );
            }, /*include_bool=*/true
        );
        return;
    }

    // Floats: custom sum kernel (accuracy-aware)
    iter.common_dtype().dispatch_floating_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) {
            sum_kernel_impl<scalar_t, /*ignore_nan=*/false>(iter);
        }
    );
}

void nansum_kernel(TensorIterator& iter) {
    // Only floats can be NaN. The front-end redirects integers to sum.
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