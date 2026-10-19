#include <raxis/ops/softmax.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/vec/functional_base.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/parallel.h>

#include <algorithm>
#include <cmath>
#include <memory>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

// The contiguous input is viewed as [outer, dim_size, inner]. Every
// (outer, inner) pair is an independent row of dim_size elements, each
// computed with the max trick for numerical stability:
//
//   m = max(x)
//   softmax(x)     = exp(x - m) / sum(exp(x - m))
//   log_softmax(x) = x - m - log(sum(exp(x - m)))
//
// From: aten/src/ATen/native/cpu/SoftMaxKernel.cpp

// inner == 1 (softmax over the last dim, the common case): each row is
// contiguous, so every pass is a vectorized sweep over dim_size elements.
template <typename scalar_t, bool LogSoftmax>
void softmax_lastdim(scalar_t* out, const scalar_t* in, int64_t outer, int64_t dim_size) {
    using Vec = Vectorized<scalar_t>;
    // Rows per task: aim for ~32K elements so tiny rows are not one task each
    int64_t grain = std::max<int64_t>(1, 32768 / dim_size);

    parallel_for(0, outer, grain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const scalar_t* x = in + i * dim_size;
            scalar_t* y = out + i * dim_size;

            scalar_t max = reduce_all<scalar_t>(
                [](Vec a, Vec b) { return vec::maximum(a, b); }, x, dim_size);

            // y = exp(x - max), accumulating the sum in the same pass
            Vec max_vec(max);
            Vec sum_vec(static_cast<scalar_t>(0));
            int64_t d = 0;
            for (; d < dim_size - (dim_size % Vec::size()); d += Vec::size()) {
                Vec e = (Vec::loadu(x + d) - max_vec).exp();
                if constexpr (!LogSoftmax) e.store(y + d);
                sum_vec = sum_vec + e;
            }
            scalar_t sum = vec_reduce_all<scalar_t>(
                [](Vec a, Vec b) { return a + b; }, sum_vec);
            for (; d < dim_size; ++d) {
                scalar_t e = std::exp(x[d] - max);
                if constexpr (!LogSoftmax) y[d] = e;
                sum += e;
            }

            if constexpr (LogSoftmax) {
                scalar_t shift = max + std::log(sum);
                map([shift](Vec v) { return v - Vec(shift); }, y, x, dim_size);
            } else {
                scalar_t inv_sum = static_cast<scalar_t>(1) / sum;
                map([inv_sum](Vec v) { return v * Vec(inv_sum); }, y, y, dim_size);
            }
        }
    });
}

// inner > 1: elements of one row are `inner` apart, but the `inner`
// neighbours of each element belong to other rows. Vectorizing across inner
// processes Vec::size() rows at once with contiguous loads, instead of
// striding through a single row.
template <typename scalar_t, bool LogSoftmax>
void softmax_innerdim(
    scalar_t* out, const scalar_t* in,
    int64_t outer, int64_t dim_size, int64_t inner
) {
    using Vec = Vectorized<scalar_t>;
    int64_t grain = std::max<int64_t>(1, 32768 / (dim_size * inner));

    parallel_for(0, outer, grain, [&](int64_t begin, int64_t end) {
        // Per-row max and sum, one slot per inner position
        auto buffer = std::make_unique<scalar_t[]>(2 * inner);
        scalar_t* max = buffer.get();
        scalar_t* sum = buffer.get() + inner;

        for (int64_t i = begin; i < end; ++i) {
            const scalar_t* x = in + i * dim_size * inner;
            scalar_t* y = out + i * dim_size * inner;

            std::copy(x, x + inner, max);
            for (int64_t d = 1; d < dim_size; ++d) {
                map2([](Vec a, Vec b) { return vec::maximum(a, b); },
                    max, max, x + d * inner, inner);
            }

            std::fill(sum, sum + inner, static_cast<scalar_t>(0));
            for (int64_t d = 0; d < dim_size; ++d) {
                const scalar_t* x_d = x + d * inner;
                scalar_t* y_d = y + d * inner;
                map2([](Vec a, Vec m) { return (a - m).exp(); }, y_d, x_d, max, inner);
                map2([](Vec s, Vec e) { return s + e; }, sum, sum, y_d, inner);
            }

            if constexpr (LogSoftmax) {
                // shift = max + log(sum), then y = x - shift
                for (int64_t k = 0; k < inner; ++k) {
                    max[k] += std::log(sum[k]);
                }
                for (int64_t d = 0; d < dim_size; ++d) {
                    map2([](Vec a, Vec shift) { return a - shift; },
                        y + d * inner, x + d * inner, max, inner);
                }
            } else {
                for (int64_t d = 0; d < dim_size; ++d) {
                    scalar_t* y_d = y + d * inner;
                    map2([](Vec e, Vec s) { return e / s; }, y_d, y_d, sum, inner);
                }
            }
        }
    });
}

template <bool LogSoftmax>
void softmax_kernel_impl(const Tensor& result, const Tensor& self, int64_t dim) {
    int64_t dim_size = self.shape()[dim];
    int64_t inner = 1;
    for (int64_t d = dim + 1; d < self.ndim(); ++d) {
        inner *= self.shape()[d];
    }
    int64_t outer = self.numel() / (dim_size * inner);

    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* in = self.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();
        if (inner == 1) {
            softmax_lastdim<scalar_t, LogSoftmax>(out, in, outer, dim_size);
        } else {
            softmax_innerdim<scalar_t, LogSoftmax>(out, in, outer, dim_size, inner);
        }
    });
}

// Backward, with g the incoming gradient and y the forward output:
//
//   softmax:     grad_input = y * (g - sum(g * y))
//   log_softmax: grad_input = g - exp(y) * sum(g)
//
// Each row needs one reduction and then one elementwise pass.
//
// From: aten/src/ATen/native/cpu/SoftMaxKernel.cpp (_vec_host_softmax_backward_lastdim)
template <typename scalar_t, bool LogSoftmax>
void softmax_backward_lastdim(
    scalar_t* grad_input, const scalar_t* grad, const scalar_t* output,
    int64_t outer, int64_t dim_size
) {
    using Vec = Vectorized<scalar_t>;
    // Same grain as the forward (PyTorch splits evenly among threads, which
    // costs more than it saves for the small batches of a training step)
    int64_t grain = std::max<int64_t>(1, 32768 / dim_size);

    parallel_for(0, outer, grain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            scalar_t* gi = grad_input + i * dim_size;
            const scalar_t* g = grad + i * dim_size;
            const scalar_t* y = output + i * dim_size;
            if constexpr (LogSoftmax) {
                scalar_t sum = reduce_all<scalar_t>([](Vec a, Vec b) { return a + b; }, g, dim_size);
                map2([sum](Vec gv, Vec yv) { return gv - yv.exp() * Vec(sum); }, gi, g, y, dim_size);
            } else {
                scalar_t sum = map2_reduce_all<scalar_t>(
                    [](Vec gv, Vec yv) { return gv * yv; },
                    [](Vec a, Vec b) { return a + b; },
                    g, y, dim_size);
                map2([sum](Vec gv, Vec yv) { return (gv - Vec(sum)) * yv; }, gi, g, y, dim_size);
            }
        }
    });
}

// From: aten/src/ATen/native/cpu/SoftMaxKernel.cpp (_vec_softmax_backward)
// inner > 1: like the forward, Vec::size() rows are handled at once by
// vectorizing across inner. The inner axis is cut into chunks so that the
// per-row sums of a chunk times dim_size rows (~128 KB) stay in cache
// between the reduction pass and the elementwise pass.
template <typename scalar_t, bool LogSoftmax>
void softmax_backward_innerdim(
    scalar_t* grad_input, const scalar_t* grad, const scalar_t* output,
    int64_t outer, int64_t dim_size, int64_t inner
) {
    using Vec = Vectorized<scalar_t>;
    int64_t outer_stride = dim_size * inner;
    constexpr int64_t BLOCK_SIZE = 128 * 1024;
    int64_t max_chunk = std::max<int64_t>(
        BLOCK_SIZE / dim_size / static_cast<int64_t>(sizeof(scalar_t)), Vec::size());
    max_chunk = max_chunk / Vec::size() * Vec::size();
    int64_t chunk_size = std::min<int64_t>(max_chunk, inner);
    int64_t num_chunks = (inner + chunk_size - 1) / chunk_size;
    int64_t grain = std::max<int64_t>(1, 32768 / (dim_size * chunk_size));

    parallel_for(0, outer * num_chunks, grain, [&](int64_t begin, int64_t end) {
        auto buffer = std::make_unique<scalar_t[]>(chunk_size);
        scalar_t* sum = buffer.get();

        for (int64_t i = begin; i < end; i++) {
            int64_t outer_idx = i / num_chunks;
            int64_t inner_begin = (i % num_chunks) * chunk_size;
            int64_t size = std::min(chunk_size, inner - inner_begin);
            int64_t base = outer_idx * outer_stride + inner_begin;

            std::fill(sum, sum + size, static_cast<scalar_t>(0));
            for (int64_t d = 0; d < dim_size; d++) {
                const scalar_t* g = grad + base + d * inner;
                if constexpr (LogSoftmax) {
                    map2([](Vec s, Vec gv) { return s + gv; }, sum, sum, g, size);
                } else {
                    const scalar_t* y = output + base + d * inner;
                    int64_t k = 0;
                    for (; k < size - (size % Vec::size()); k += Vec::size()) {
                        Vec s = Vec::loadu(sum + k) + Vec::loadu(g + k) * Vec::loadu(y + k);
                        s.store(sum + k);
                    }
                    for (; k < size; k++) {
                        sum[k] += g[k] * y[k];
                    }
                }
            }

            for (int64_t d = 0; d < dim_size; d++) {
                int64_t offset = base + d * inner;
                const scalar_t* g = grad + offset;
                const scalar_t* y = output + offset;
                scalar_t* gi = grad_input + offset;
                int64_t k = 0;
                for (; k < size - (size % Vec::size()); k += Vec::size()) {
                    Vec gv = Vec::loadu(g + k);
                    Vec yv = Vec::loadu(y + k);
                    Vec sv = Vec::loadu(sum + k);
                    Vec out = LogSoftmax ? gv - yv.exp() * sv : yv * (gv - sv);
                    out.store(gi + k);
                }
                for (; k < size; k++) {
                    gi[k] = LogSoftmax ? g[k] - std::exp(y[k]) * sum[k] : y[k] * (g[k] - sum[k]);
                }
            }
        }
    });
}

template <bool LogSoftmax>
void softmax_backward_kernel_impl(
    const Tensor& grad_input, const Tensor& grad, const Tensor& output, int64_t dim
) {
    int64_t dim_size = output.shape()[dim];
    int64_t inner = 1;
    for (int64_t d = dim + 1; d < output.ndim(); ++d) {
        inner *= output.shape()[d];
    }
    int64_t outer = output.numel() / (dim_size * inner);

    output.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        scalar_t* gi = grad_input.mutable_data_ptr<scalar_t>();
        const scalar_t* g = grad.data_ptr<scalar_t>();
        const scalar_t* y = output.data_ptr<scalar_t>();
        if (inner == 1) {
            softmax_backward_lastdim<scalar_t, LogSoftmax>(gi, g, y, outer, dim_size);
        } else {
            softmax_backward_innerdim<scalar_t, LogSoftmax>(gi, g, y, outer, dim_size, inner);
        }
    });
}

void softmax_backward_kernel(const Tensor& grad_input, const Tensor& grad, const Tensor& output, int64_t dim) {
    softmax_backward_kernel_impl</*LogSoftmax=*/false>(grad_input, grad, output, dim);
}

void log_softmax_backward_kernel(const Tensor& grad_input, const Tensor& grad, const Tensor& output, int64_t dim) {
    softmax_backward_kernel_impl</*LogSoftmax=*/true>(grad_input, grad, output, dim);
}

void softmax_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
    softmax_kernel_impl</*LogSoftmax=*/false>(result, self, dim);
}

void log_softmax_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
    softmax_kernel_impl</*LogSoftmax=*/true>(result, self, dim);
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(softmax_stub, &softmax_kernel);
REGISTER_DISPATCH(log_softmax_stub, &log_softmax_kernel);
REGISTER_DISPATCH(softmax_backward_stub, &softmax_backward_kernel);
REGISTER_DISPATCH(log_softmax_backward_stub, &log_softmax_backward_kernel);

} // namespace raxis
