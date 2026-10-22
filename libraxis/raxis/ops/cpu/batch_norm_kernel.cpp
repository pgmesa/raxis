#include <raxis/ops/normalization.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/vec/functional_base.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/parallel.h>

#include <cmath>
#include <cstdint>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

// From: aten/src/ATen/native/cpu/batch_norm_kernel.cpp (contiguous NCHW paths)
//
// The input is [N, C, image_size]: channel c is N separate runs of
// image_size elements. Statistics and gradients reduce over all of them, so
// the channels are the parallel dim; the elementwise forward parallelizes
// over the N * C runs.
//
// Sums accumulate in double, as in PyTorch (acc_type<float> on CPU), so
// large batches do not lose precision.

struct BatchNormShape {
    int64_t batch;
    int64_t channels;
    int64_t image_size;
};

inline BatchNormShape batch_norm_shape(const Tensor& input) {
    const int64_t batch = input.shape()[0];
    const int64_t channels = input.shape()[1];
    return {batch, channels, input.numel() / (batch * channels)};
}

// mean and invstd of channel c: batch statistics when training, running ones
// otherwise. PyTorch computes the eval invstd in scalar_t in the forward and
// in double in the backward; acc_t keeps both bitwise equal to it.
template <typename scalar_t, typename acc_t>
inline void channel_mean_invstd(
    int64_t c, bool train, double eps,
    const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var,
    scalar_t& mean, scalar_t& invstd
) {
    if (train) {
        mean = save_mean.data_ptr<scalar_t>()[c];
        invstd = save_invstd.data_ptr<scalar_t>()[c];
    } else {
        mean = running_mean.data_ptr<scalar_t>()[c];
        invstd = static_cast<scalar_t>(
            1 / std::sqrt(running_var.data_ptr<scalar_t>()[c] + static_cast<acc_t>(eps)));
    }
}

// From: batch_norm_cpu_collect_linear_and_constant_terms, batch_norm_cpu_contiguous_impl
//
//   output = (x - mean) * invstd * weight + bias
//          = x * alpha + beta,  alpha = invstd * weight,  beta = bias - mean * alpha
//
// so each element is one multiply-add with per-channel constants.
template <typename scalar_t>
void batch_norm_impl(
    const Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps
) {
    using Vec = Vectorized<scalar_t>;
    const auto [batch, channels, image_size] = batch_norm_shape(input);

    Tensor alpha = empty({channels}, input.options());
    Tensor beta = empty({channels}, input.options());
    scalar_t* alpha_data = alpha.mutable_data_ptr<scalar_t>();
    scalar_t* beta_data = beta.mutable_data_ptr<scalar_t>();
    const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
    const scalar_t* bias_data = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;
    for (int64_t c = 0; c < channels; ++c) {
        scalar_t mean, invstd;
        channel_mean_invstd<scalar_t, scalar_t>(
            c, train, eps, save_mean, save_invstd, running_mean, running_var, mean, invstd);
        const scalar_t w = weight_data ? weight_data[c] : scalar_t(1);
        const scalar_t b = bias_data ? bias_data[c] : scalar_t(0);
        alpha_data[c] = invstd * w;
        beta_data[c] = b - mean * alpha_data[c];
    }

    const scalar_t* in = input.data_ptr<scalar_t>();
    scalar_t* out = output.mutable_data_ptr<scalar_t>();
    if (image_size == 1) {
        // [N, C] (BatchNorm1d after a Linear): a row holds one value per
        // channel, so vectorize along it with per-lane alpha and beta
        // From: batch_norm_cpu_channels_last_impl
        parallel_for(0, batch, 1, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                map3([](Vec x, Vec a, Vec b) { return x * a + b; },
                     out + i * channels, in + i * channels, alpha_data, beta_data, channels);
            }
        });
        return;
    }
    parallel_for(0, batch * channels, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t c = i % channels;
            const Vec alpha_vec(alpha_data[c]);
            const Vec beta_vec(beta_data[c]);
            map([&](Vec x) { return x * alpha_vec + beta_vec; },
                out + i * image_size, in + i * image_size, image_size);
        }
    });
}

// From: batch_norm_cpu_collect_stats_contiguous_impl. Two passes per channel
// (mean, then sum of squared deviations) instead of sum(x^2) - n * mean^2,
// which cancels catastrophically when the mean is large.
//
// For [N, C] inputs with more rows than threads it follows
// batch_norm_cpu_collect_stats_channels_last_impl instead: the column sums of
// the matrix are a vertical reduction, so each thread adds its rows into its
// own row of a [threads, C] buffer (vectorized along the channels), and the
// buffer rows are then added up in double.
template <typename scalar_t>
void batch_norm_collect_stats_rows(
    scalar_t* mean_data, scalar_t* var_sum_data, const Tensor& input, int64_t rows, int64_t channels
) {
    using Vec = Vectorized<scalar_t>;
    const int num_threads = get_num_threads();
    Tensor buffer = zeros({num_threads, channels}, input.options());
    scalar_t* buffer_data = buffer.mutable_data_ptr<scalar_t>();
    const scalar_t* in = input.data_ptr<scalar_t>();

    auto reduce_buffer = [&](scalar_t* out, auto finish) {
        parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
            for (int64_t c = begin; c < end; ++c) {
                double sum = 0;
                for (int t = 0; t < num_threads; ++t) {
                    sum += buffer_data[t * channels + c];
                }
                out[c] = finish(sum);
            }
        });
    };

    parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
        scalar_t* acc = buffer_data + get_thread_num() * channels;
        for (int64_t i = begin; i < end; ++i) {
            map2([](Vec x, Vec y) { return x + y; }, acc, in + i * channels, acc, channels);
        }
    });
    reduce_buffer(mean_data, [&](double sum) { return static_cast<scalar_t>(sum / rows); });

    std::fill_n(buffer_data, num_threads * channels, scalar_t(0));
    parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
        scalar_t* acc = buffer_data + get_thread_num() * channels;
        for (int64_t i = begin; i < end; ++i) {
            map3([](Vec x, Vec y, Vec mean) { return y + (x - mean) * (x - mean); },
                 acc, in + i * channels, acc, mean_data, channels);
        }
    });
    reduce_buffer(var_sum_data, [](double sum) { return static_cast<scalar_t>(sum); });
}

template <typename scalar_t>
void batch_norm_collect_stats_impl(const Tensor& mean, const Tensor& var_sum, const Tensor& input) {
    const auto [batch, channels, image_size] = batch_norm_shape(input);
    const int64_t n = batch * image_size;
    const scalar_t* in = input.data_ptr<scalar_t>();
    scalar_t* mean_data = mean.mutable_data_ptr<scalar_t>();
    scalar_t* var_sum_data = var_sum.mutable_data_ptr<scalar_t>();

    if (image_size == 1 && batch > get_num_threads()) {
        batch_norm_collect_stats_rows(mean_data, var_sum_data, input, batch, channels);
        return;
    }
    parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            double sum = 0;
            for (int64_t b = 0; b < batch; ++b) {
                const scalar_t* x = in + (b * channels + c) * image_size;
                for (int64_t i = 0; i < image_size; ++i) {
                    sum += x[i];
                }
            }
            const scalar_t channel_mean = sum / n;
            mean_data[c] = channel_mean;

            double channel_var_sum = 0;
            for (int64_t b = 0; b < batch; ++b) {
                const scalar_t* x = in + (b * channels + c) * image_size;
                for (int64_t i = 0; i < image_size; ++i) {
                    channel_var_sum += (x[i] - channel_mean) * (x[i] - channel_mean);
                }
            }
            var_sum_data[c] = channel_var_sum;
        }
    });
}

// From: batch_norm_cpu_backward_contiguous_impl
//
// With xhat = (x - mean) * invstd and n elements per channel:
//
//   grad_bias   = sum(dy)
//   grad_weight = sum(dy * xhat) = dotp * invstd,  dotp = sum((x - mean) * dy)
//   grad_input  = (dy - sum(dy) / n - (x - mean) * invstd^2 * dotp / n) * invstd * weight
//
// The two sums come from one fused pass over x and dy. In eval mode mean and
// invstd are constants, so grad_input is just dy * invstd * weight.

// [N, C] inputs. From: batch_norm_cpu_backward_channels_last_impl. The sums
// are column sums, computed as in batch_norm_collect_stats_rows, and every
// per-channel term becomes a vector lane.
template <typename scalar_t>
void batch_norm_backward_rows(
    scalar_t* dx_data, scalar_t* grad_weight_data, scalar_t* grad_bias_data,
    const scalar_t* dy_data, const scalar_t* in, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd, bool train, double eps,
    int64_t rows, int64_t channels
) {
    using Vec = Vectorized<scalar_t>;
    const Tensor weight_ones = weight.defined() ? weight : ones({channels}, input.options());
    const scalar_t* weight_data = weight_ones.data_ptr<scalar_t>();
    const scalar_t* mean_data = train ? save_mean.data_ptr<scalar_t>() : running_mean.data_ptr<scalar_t>();
    Tensor invstd = train ? save_invstd : empty({channels}, input.options());
    if (!train) {
        const scalar_t* var = running_var.data_ptr<scalar_t>();
        scalar_t* inv = invstd.mutable_data_ptr<scalar_t>();
        for (int64_t c = 0; c < channels; ++c) {
            inv[c] = static_cast<scalar_t>(1 / std::sqrt(var[c] + eps));
        }
    }
    const scalar_t* invstd_data = invstd.data_ptr<scalar_t>();

    // [2, threads, C]: per-thread sum(dy) and dotp; after the reduction the
    // first row of each half holds the totals
    const int num_threads = get_num_threads();
    Tensor buffer = zeros({2, num_threads, channels}, input.options());
    scalar_t* sum_data = buffer.mutable_data_ptr<scalar_t>();
    scalar_t* dotp_data = sum_data + num_threads * channels;
    parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
        const int tid = get_thread_num();
        scalar_t* sum_acc = sum_data + tid * channels;
        scalar_t* dotp_acc = dotp_data + tid * channels;
        for (int64_t i = begin; i < end; ++i) {
            const scalar_t* x = in + i * channels;
            const scalar_t* dy = dy_data + i * channels;
            map2([](Vec sum, Vec dy) { return sum + dy; }, sum_acc, sum_acc, dy, channels);
            map4([](Vec dotp, Vec x, Vec mean, Vec dy) { return dotp + (x - mean) * dy; },
                 dotp_acc, dotp_acc, x, mean_data, dy, channels);
        }
    });
    parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            double sum = 0;
            double dotp = 0;
            for (int t = 0; t < num_threads; ++t) {
                sum += sum_data[t * channels + c];
                dotp += dotp_data[t * channels + c];
            }
            sum_data[c] = static_cast<scalar_t>(sum);
            dotp_data[c] = static_cast<scalar_t>(dotp);
        }
    });

    if (dx_data) {
        const Vec n(static_cast<scalar_t>(rows));
        parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                scalar_t* dx = dx_data + i * channels;
                const scalar_t* x = in + i * channels;
                const scalar_t* dy = dy_data + i * channels;
                if (train) {
                    // Lane-wise version of the formula above; the per-channel
                    // terms are loaded as vectors
                    int64_t d = 0;
                    // count is a constant at each call, so the branch folds
                    auto load = [&](const scalar_t* p, int64_t count) {
                        return count == Vec::size() ? Vec::loadu(p + d) : Vec::loadu(p + d, count);
                    };
                    auto grad = [&](int64_t count) {
                        const Vec inv = load(invstd_data, count);
                        const Vec k = load(dotp_data, count) * inv * inv / n;
                        const Vec centered = (load(x, count) - load(mean_data, count)) * k;
                        const Vec grad_mean = load(sum_data, count) / n;
                        return (load(dy, count) - grad_mean - centered) * inv * load(weight_data, count);
                    };
                    for (; d < channels - (channels % Vec::size()); d += Vec::size()) {
                        grad(Vec::size()).store(dx + d);
                    }
                    if (channels - d > 0) {
                        grad(channels - d).store(dx + d, channels - d);
                    }
                } else {
                    map3([](Vec dy, Vec inv, Vec w) { return dy * inv * w; },
                         dx, dy, invstd_data, weight_data, channels);
                }
            }
        });
    }
    if (grad_weight_data) {
        map2([](Vec dotp, Vec inv) { return dotp * inv; }, grad_weight_data, dotp_data, invstd_data, channels);
    }
    if (grad_bias_data) {
        std::copy_n(sum_data, channels, grad_bias_data);
    }
}

template <typename scalar_t>
void batch_norm_backward_impl(
    const Tensor& grad_input, const Tensor& grad_weight, const Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd, bool train, double eps
) {
    using Vec = Vectorized<scalar_t>;
    const auto [batch, channels, image_size] = batch_norm_shape(input);
    const int64_t n = batch * image_size;
    const scalar_t* in = input.data_ptr<scalar_t>();
    const scalar_t* dy_data = grad_output.data_ptr<scalar_t>();
    const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
    scalar_t* dx_data = grad_input.defined() ? grad_input.mutable_data_ptr<scalar_t>() : nullptr;
    scalar_t* grad_weight_data = grad_weight.defined() ? grad_weight.mutable_data_ptr<scalar_t>() : nullptr;
    scalar_t* grad_bias_data = grad_bias.defined() ? grad_bias.mutable_data_ptr<scalar_t>() : nullptr;

    if (image_size == 1) {
        batch_norm_backward_rows(
            dx_data, grad_weight_data, grad_bias_data, dy_data, in, input, weight,
            running_mean, running_var, save_mean, save_invstd, train, eps, batch, channels);
        return;
    }
    parallel_for(0, channels, 1, [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            const scalar_t w = weight_data ? weight_data[c] : scalar_t(1);
            scalar_t mean, invstd;
            channel_mean_invstd<scalar_t, double>(
                c, train, eps, save_mean, save_invstd, running_mean, running_var, mean, invstd);

            double sum = 0;
            double dotp = 0;
            for (int64_t b = 0; b < batch; ++b) {
                const int64_t offset = (b * channels + c) * image_size;
                sum += reduce_all<scalar_t>(
                    [](Vec x, Vec y) { return x + y; }, dy_data + offset, image_size);
                dotp += map2_reduce_all<scalar_t>(
                    [mean](Vec x, Vec dy) { return (x - Vec(mean)) * dy; },
                    [](Vec x, Vec y) { return x + y; },
                    in + offset, dy_data + offset, image_size);
            }

            if (dx_data) {
                for (int64_t b = 0; b < batch; ++b) {
                    const int64_t offset = (b * channels + c) * image_size;
                    if (train) {
                        const scalar_t k = static_cast<scalar_t>(dotp) * invstd * invstd / n;
                        const scalar_t grad_mean = sum / n;
                        map2([=](Vec x, Vec dy) {
                                 Vec dx = (x - Vec(mean)) * Vec(k);
                                 return (dy - Vec(grad_mean) - dx) * Vec(invstd) * Vec(w);
                             },
                             dx_data + offset, in + offset, dy_data + offset, image_size);
                    } else {
                        map([=](Vec dy) { return dy * Vec(invstd) * Vec(w); },
                            dx_data + offset, dy_data + offset, image_size);
                    }
                }
            }
            if (grad_weight_data) {
                grad_weight_data[c] = dotp * invstd;
            }
            if (grad_bias_data) {
                grad_bias_data[c] = sum;
            }
        }
    });
}

void batch_norm_kernel(
    const Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps
) {
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        batch_norm_impl<scalar_t>(output, input, weight, bias, save_mean, save_invstd,
                                  running_mean, running_var, train, eps);
    });
}

void batch_norm_collect_stats_kernel(const Tensor& mean, const Tensor& var_sum, const Tensor& input) {
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        batch_norm_collect_stats_impl<scalar_t>(mean, var_sum, input);
    });
}

void batch_norm_backward_kernel(
    const Tensor& grad_input, const Tensor& grad_weight, const Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd, bool train, double eps
) {
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        batch_norm_backward_impl<scalar_t>(
            grad_input, grad_weight, grad_bias, grad_output, input, weight,
            running_mean, running_var, save_mean, save_invstd, train, eps);
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(batch_norm_stub, &batch_norm_kernel);
REGISTER_DISPATCH(batch_norm_collect_stats_stub, &batch_norm_collect_stats_kernel);
REGISTER_DISPATCH(batch_norm_backward_stub, &batch_norm_backward_kernel);

} // namespace raxis
