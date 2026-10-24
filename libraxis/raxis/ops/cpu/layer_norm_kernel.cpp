#include <raxis/ops/normalization.h>
#include <raxis/ops/cpu/moments_utils.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/vec/functional_base.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>

#include <cmath>
#include <cstdint>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

// From: aten/src/ATen/native/cpu/layer_norm_kernel.cpp
//
// The input is a contiguous [M, N] matrix: M rows (the leading dims) of N
// elements (normalized_shape), each normalized on its own. Unlike batch
// norm, the statistics reduce along a contiguous row, so the rows are the
// parallel dim and each row is one pass of rowwise_moments plus one
// elementwise pass. Accumulation stays in scalar_t, as in PyTorch.

// y = (x - mean) * rstd * gamma + beta, the same expression order as
// LayerNormSecondPass. A missing gamma or beta is skipped, not replaced by
// 1 and 0, so every case stays vectorized.
template <typename T>
void layer_norm_second_pass(
    const T* X, const T* gamma, const T* beta, T* Y, int64_t N, T scale, T bias
) {
    using Vec = Vectorized<T>;
    const Vec bias_vec(bias);
    const Vec scale_vec(scale);
    if (gamma && beta) {
        map3([=](Vec x, Vec g, Vec b) { return (x + bias_vec) * scale_vec * g + b; }, Y, X, gamma, beta, N);
    } else if (gamma) {
        map2([=](Vec x, Vec g) { return (x + bias_vec) * scale_vec * g; }, Y, X, gamma, N);
    } else if (beta) {
        map2([=](Vec x, Vec b) { return (x + bias_vec) * scale_vec + b; }, Y, X, beta, N);
    } else {
        map([=](Vec x) { return (x + bias_vec) * scale_vec; }, Y, X, N);
    }
}

template <typename T>
void layer_norm_impl(
    const Tensor& output, const Tensor& mean, const Tensor& rstd, const Tensor& input,
    const Tensor& weight, const Tensor& bias, int64_t M, int64_t N, double eps
) {
    const T* X_data = input.data_ptr<T>();
    const T* gamma_data = weight.defined() ? weight.data_ptr<T>() : nullptr;
    const T* beta_data = bias.defined() ? bias.data_ptr<T>() : nullptr;
    T* Y_data = output.mutable_data_ptr<T>();
    T* mean_data = mean.mutable_data_ptr<T>();
    T* rstd_data = rstd.mutable_data_ptr<T>();
    const T eps_t = static_cast<T>(eps);

    parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            auto [mean_val, var_val] = rowwise_moments(X_data + i * N, N);
            const T rstd_val = T(1) / std::sqrt(var_val + eps_t);
            layer_norm_second_pass<T>(X_data + i * N, gamma_data, beta_data, Y_data + i * N, N,
                                      rstd_val, -mean_val);
            mean_data[i] = mean_val;
            rstd_data[i] = rstd_val;
        }
    });
}

// From: layer_norm_backward_frame
//
// With x_hat = (x - mean) * rstd and g = dy * gamma, per row:
//   dx = rstd * (g - mean(g) - x_hat * mean(g * x_hat))
// expanded so that each element is one multiply-add chain:
//   dx = a * dy * gamma + b * x + c,  a = rstd,
//   b = (db * mean - ds) * rstd^3 / N,  c = -b * mean - db * rstd / N
// where ds = sum(dy * x * gamma) and db = sum(dy * gamma) are the only
// reductions. dgamma = sum over rows of dy * x_hat and dbeta = sum of dy are
// added into this thread's row of the buffer while the row is in cache.
template <typename T>
void layer_norm_backward_row(
    const T* dY_data, const T* X_data, const T* mean_data, const T* rstd_data, const T* gamma_data,
    T* dX_data, T* dgamma_acc, T* dbeta_acc, T scale, int64_t N, int64_t i
) {
    using Vec = Vectorized<T>;
    const T* dY = dY_data + i * N;
    const T* X = X_data + i * N;
    const T rstd = rstd_data[i];
    const T mean = mean_data[i];
    if (dgamma_acc) {
        const T b = -rstd * mean;
        map3([=](Vec dgamma, Vec dy, Vec x) { return dgamma + dy * (Vec(rstd) * x + Vec(b)); },
             dgamma_acc, dgamma_acc, dY, X, N);
    }
    if (dbeta_acc) {
        map2([](Vec dbeta, Vec dy) { return dbeta + dy; }, dbeta_acc, dbeta_acc, dY, N);
    }
    if (!dX_data) {
        return;
    }
    const auto add = [](Vec x, Vec y) { return x + y; };
    T ds, db;
    if (gamma_data) {
        ds = map3_reduce_all<T>([](Vec x, Vec y, Vec z) { return x * y * z; }, add, dY, X, gamma_data, N);
        db = map2_reduce_all<T>([](Vec x, Vec y) { return x * y; }, add, dY, gamma_data, N);
    } else {
        ds = map2_reduce_all<T>([](Vec x, Vec y) { return x * y; }, add, dY, X, N);
        db = reduce_all<T>(add, dY, N);
    }
    const T a = rstd;
    const T b = (db * mean - ds) * a * a * a * scale;
    const T c = -b * mean - db * a * scale;
    T* dX = dX_data + i * N;
    if (gamma_data) {
        map3([=](Vec dy, Vec gamma, Vec x) { return Vec(a) * dy * gamma + Vec(b) * x + Vec(c); },
             dX, dY, gamma_data, X, N);
    } else {
        map2([=](Vec dy, Vec x) { return Vec(a) * dy + Vec(b) * x + Vec(c); }, dX, dY, X, N);
    }
}

// dgamma and dbeta are column sums over the M rows: each thread adds its rows
// into its own row of a [2, threads, N] buffer, then the buffer rows are
// added up in parallel along N (two-pass reduction, no atomics).
template <typename T>
void layer_norm_backward_impl(
    const Tensor& grad_input, const Tensor& grad_weight, const Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& rstd,
    const Tensor& weight, int64_t M, int64_t N
) {
    const T* dY_data = grad_output.data_ptr<T>();
    const T* X_data = input.data_ptr<T>();
    const T* mean_data = mean.data_ptr<T>();
    const T* rstd_data = rstd.data_ptr<T>();
    const T* gamma_data = weight.defined() ? weight.data_ptr<T>() : nullptr;
    T* dX_data = grad_input.defined() ? grad_input.mutable_data_ptr<T>() : nullptr;
    T* dgamma_data = grad_weight.defined() ? grad_weight.mutable_data_ptr<T>() : nullptr;
    T* dbeta_data = grad_bias.defined() ? grad_bias.mutable_data_ptr<T>() : nullptr;
    const T scale = T(1) / static_cast<T>(N);

    const int num_threads = get_num_threads();
    Tensor buffer;
    T* buffer_data = nullptr;
    if (dgamma_data || dbeta_data) {
        buffer = zeros({2, num_threads, N}, input.options());
        buffer_data = buffer.mutable_data_ptr<T>();
    }

    parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        const int tid = get_thread_num();
        RAXIS_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
        T* dgamma_acc = dgamma_data ? buffer_data + tid * N : nullptr;
        T* dbeta_acc = dbeta_data ? buffer_data + (num_threads + tid) * N : nullptr;
        for (int64_t i = begin; i < end; ++i) {
            layer_norm_backward_row<T>(dY_data, X_data, mean_data, rstd_data, gamma_data,
                                       dX_data, dgamma_acc, dbeta_acc, scale, N, i);
        }
    });

    if (buffer_data) {
        parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
            for (int64_t j = begin; j < end; ++j) {
                T dgamma_v = 0;
                T dbeta_v = 0;
                for (int64_t t = 0; t < num_threads; ++t) {
                    dgamma_v += buffer_data[t * N + j];
                    dbeta_v += buffer_data[(num_threads + t) * N + j];
                }
                if (dgamma_data) {
                    dgamma_data[j] = dgamma_v;
                }
                if (dbeta_data) {
                    dbeta_data[j] = dbeta_v;
                }
            }
        });
    }
}

void layer_norm_kernel(
    const Tensor& output, const Tensor& mean, const Tensor& rstd, const Tensor& input,
    const Tensor& weight, const Tensor& bias, int64_t M, int64_t N, double eps
) {
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        layer_norm_impl<scalar_t>(output, mean, rstd, input, weight, bias, M, N, eps);
    });
}

void layer_norm_backward_kernel(
    const Tensor& grad_input, const Tensor& grad_weight, const Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& rstd,
    const Tensor& weight, int64_t M, int64_t N
) {
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        layer_norm_backward_impl<scalar_t>(
            grad_input, grad_weight, grad_bias, grad_output, input, mean, rstd, weight, M, N);
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(layer_norm_stub, &layer_norm_kernel);
REGISTER_DISPATCH(layer_norm_backward_stub, &layer_norm_backward_kernel);

} // namespace raxis
