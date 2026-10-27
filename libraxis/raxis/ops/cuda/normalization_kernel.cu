#include <algorithm>
#include <cstdint>
#include <limits>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/block_reduce.cuh>
#include <raxis/ops/cuda/integer_divider.cuh>
#include <raxis/ops/normalization.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


// batch_norm and layer_norm on the GPU, same formulas as the CPU kernels
// (ops/cpu/batch_norm_kernel.cpp, layer_norm_kernel.cpp).
// From: aten/src/ATen/native/cuda/Normalization.cuh (batch_norm_collect_statistics_kernel,
// batch_norm_backward_reduce_kernel, batch_norm_transform_input_kernel) and
// layer_norm_kernel.cu (RowwiseMomentsCUDAKernel, LayerNormForwardCUDAKernel,
// ComputeInternalGradientsCUDAKernel, GammaBetaBackwardCUDAKernel)
//
// Both are a reduction per group followed by an elementwise pass. The groups
// differ: batch_norm reduces a channel, spread over the batch ([N, C, HW]:
// C groups of N * HW strided elements), layer_norm a row (M contiguous
// groups of N). Each group gets one block, whose threads accumulate their
// share with Welford (mean and m2 in one pass, no catastrophic cancellation
// of sum(x^2) - n * mean^2) and merge it with block_welford.

namespace raxis {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxRowThreads = 512;

unsigned blocks_for(int64_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

void check_32bit(const Tensor& t, const char* op) {
    RAXIS_CHECK(t.numel() <= std::numeric_limits<int32_t>::max(),
        op, ": tensors over 2^31 elements are not supported on CUDA yet");
}

// ---------------------------------------------------------------------------
// batch_norm. The input is contiguous [N, C, HW]; channel c of image b is
// the run of HW elements at (b * C + c) * HW. A block walks the N * HW
// elements of its channel with consecutive threads on consecutive elements
// of a run.
// ---------------------------------------------------------------------------

struct ChannelLayout {
    int32_t batch, channels, image_size;
    cuda::IntDivider image_div;     // flat channel index -> (image, position)
    cuda::IntDivider channels_div;  // plane (b * C + c) -> channel
};

ChannelLayout channel_layout(const Tensor& input) {
    ChannelLayout l;
    l.batch = static_cast<int32_t>(input.shape()[0]);
    l.channels = static_cast<int32_t>(input.shape()[1]);
    l.image_size = static_cast<int32_t>(input.numel() / std::max<int64_t>(1, l.batch * l.channels));
    l.image_div = cuda::IntDivider(std::max(l.image_size, 1));
    l.channels_div = cuda::IntDivider(l.channels);
    return l;
}

// Element i of channel c, for i in [0, N * HW)
__device__ inline int64_t channel_offset(const ChannelLayout& l, int32_t c, uint32_t i) {
    const auto bi = l.image_div.divmod(i);
    return (static_cast<int64_t>(bi.div) * l.channels + c) * l.image_size + bi.mod;
}

template <typename scalar_t>
__global__ void batch_norm_collect_stats_kernel(
    scalar_t* mean, scalar_t* var_sum, const scalar_t* input, ChannelLayout l
) {
    __shared__ cuda::WelfordData<scalar_t> shared[cuda::kWarpSize];
    const int32_t c = blockIdx.x;
    const uint32_t n = static_cast<uint32_t>(l.batch) * l.image_size;
    cuda::WelfordData<scalar_t> acc{0, 0, 0};
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        acc = cuda::welford_update(acc, input[channel_offset(l, c, i)]);
    }
    acc = cuda::block_welford(acc, shared);
    if (threadIdx.x == 0) {
        mean[c] = acc.mean;
        var_sum[c] = acc.m2;
    }
}

template <typename scalar_t>
__global__ void batch_norm_update_stats_kernel(
    scalar_t* save_invstd, scalar_t* running_mean, scalar_t* running_var,
    const scalar_t* mean, const scalar_t* var_sum, int32_t channels, int64_t n, scalar_t momentum, double eps
) {
    const int32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    const double var = static_cast<double>(var_sum[c]) / n;
    save_invstd[c] = (var != 0 || eps != 0) ? static_cast<scalar_t>(1 / ::sqrt(var + eps)) : scalar_t(0);
    if (running_mean != nullptr) {
        running_mean[c] = momentum * mean[c] + (1 - momentum) * running_mean[c];
        const double unbiased_var = static_cast<double>(var_sum[c]) / (n - 1);
        running_var[c] = momentum * static_cast<scalar_t>(unbiased_var) + (1 - momentum) * running_var[c];
    }
}

// mean and invstd of a channel: the batch's when training, else the running
// statistics
template <typename scalar_t>
__device__ inline void channel_mean_invstd(
    int32_t c, const scalar_t* mean, const scalar_t* invstd, const scalar_t* running_var, double eps,
    scalar_t& mean_out, scalar_t& invstd_out
) {
    mean_out = mean[c];
    invstd_out = running_var != nullptr ? static_cast<scalar_t>(1 / ::sqrt(static_cast<double>(running_var[c]) + eps))
                                        : invstd[c];
}

// output = (x - mean) * invstd * weight + bias, one thread per element.
// mean is save_mean or running_mean; running_var is non-null in eval mode
template <typename scalar_t>
__global__ void batch_norm_transform_kernel(
    scalar_t* output, const scalar_t* input, const scalar_t* weight, const scalar_t* bias,
    const scalar_t* mean, const scalar_t* invstd, const scalar_t* running_var, double eps,
    uint32_t numel, ChannelLayout l
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }
    const int32_t c = l.channels_div.divmod(l.image_div.div(index)).mod;
    scalar_t m, inv;
    channel_mean_invstd(c, mean, invstd, running_var, eps, m, inv);
    const scalar_t alpha = inv * (weight != nullptr ? weight[c] : scalar_t(1));
    const scalar_t beta = (bias != nullptr ? bias[c] : scalar_t(0)) - m * alpha;
    output[index] = input[index] * alpha + beta;
}

// Per channel: sum(dy) and dotp = sum((x - mean) * dy), and from them
// grad_weight = dotp * invstd and grad_bias = sum(dy)
template <typename scalar_t>
__global__ void batch_norm_backward_reduce_kernel(
    scalar_t* sum_dy, scalar_t* dotp, scalar_t* grad_weight, scalar_t* grad_bias,
    const scalar_t* grad_output, const scalar_t* input,
    const scalar_t* mean, const scalar_t* invstd, const scalar_t* running_var, double eps, ChannelLayout l
) {
    __shared__ scalar_t shared[cuda::kWarpSize];
    const int32_t c = blockIdx.x;
    const uint32_t n = static_cast<uint32_t>(l.batch) * l.image_size;
    scalar_t m, inv;
    channel_mean_invstd(c, mean, invstd, running_var, eps, m, inv);
    scalar_t sum = 0;
    scalar_t dot = 0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int64_t offset = channel_offset(l, c, i);
        const scalar_t dy = grad_output[offset];
        sum += dy;
        dot += (input[offset] - m) * dy;
    }
    sum = cuda::block_sum(sum, shared);
    dot = cuda::block_sum(dot, shared);
    if (threadIdx.x == 0) {
        sum_dy[c] = sum;
        dotp[c] = dot;
        if (grad_weight != nullptr) grad_weight[c] = dot * inv;
        if (grad_bias != nullptr) grad_bias[c] = sum;
    }
}

// grad_input = (dy - sum(dy) / n - (x - mean) * invstd^2 * dotp / n) * invstd * weight
// when training; dy * invstd * weight in eval mode (constant statistics)
template <typename scalar_t>
__global__ void batch_norm_backward_input_kernel(
    scalar_t* grad_input, const scalar_t* grad_output, const scalar_t* input, const scalar_t* weight,
    const scalar_t* mean, const scalar_t* invstd, const scalar_t* running_var, double eps,
    const scalar_t* sum_dy, const scalar_t* dotp, uint32_t numel, ChannelLayout l
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }
    const int32_t c = l.channels_div.divmod(l.image_div.div(index)).mod;
    scalar_t m, inv;
    channel_mean_invstd(c, mean, invstd, running_var, eps, m, inv);
    const scalar_t w = weight != nullptr ? weight[c] : scalar_t(1);
    if (running_var != nullptr) {
        grad_input[index] = grad_output[index] * inv * w;
        return;
    }
    const scalar_t n = static_cast<scalar_t>(l.batch) * static_cast<scalar_t>(l.image_size);
    const scalar_t k = dotp[c] * inv * inv / n;
    const scalar_t grad_mean = sum_dy[c] / n;
    grad_input[index] = (grad_output[index] - grad_mean - (input[index] - m) * k) * inv * w;
}

template <typename scalar_t>
const scalar_t* ptr_or_null(const Tensor& t) {
    return t.defined() ? t.data_ptr<scalar_t>() : nullptr;
}

template <typename scalar_t>
scalar_t* mutable_ptr_or_null(const Tensor& t) {
    return t.defined() ? t.mutable_data_ptr<scalar_t>() : nullptr;
}

void batch_norm_collect_stats_cuda(const Tensor& mean, const Tensor& var_sum, const Tensor& input) {
    DeviceGuard guard(input.device());
    check_32bit(input, "batch_norm");
    const ChannelLayout l = channel_layout(input);
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        batch_norm_collect_stats_kernel<scalar_t><<<l.channels, kThreads, 0, cuda::current_stream()>>>(
            mean.mutable_data_ptr<scalar_t>(), var_sum.mutable_data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), l);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void batch_norm_update_stats_cuda(
    const Tensor& save_invstd, const Tensor& running_mean, const Tensor& running_var,
    const Tensor& mean, const Tensor& var_sum, int64_t n, double momentum, double eps
) {
    DeviceGuard guard(mean.device());
    const auto channels = static_cast<int32_t>(mean.numel());
    mean.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        batch_norm_update_stats_kernel<scalar_t><<<blocks_for(channels), kThreads, 0, cuda::current_stream()>>>(
            save_invstd.mutable_data_ptr<scalar_t>(), mutable_ptr_or_null<scalar_t>(running_mean),
            mutable_ptr_or_null<scalar_t>(running_var), mean.data_ptr<scalar_t>(), var_sum.data_ptr<scalar_t>(),
            channels, n, static_cast<scalar_t>(momentum), eps);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void batch_norm_cuda(
    const Tensor& output, const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& save_mean, const Tensor& save_invstd,
    const Tensor& running_mean, const Tensor& running_var, bool train, double eps
) {
    DeviceGuard guard(input.device());
    check_32bit(input, "batch_norm");
    const ChannelLayout l = channel_layout(input);
    const int64_t numel = input.numel();
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        batch_norm_transform_kernel<scalar_t><<<blocks_for(numel), kThreads, 0, cuda::current_stream()>>>(
            output.mutable_data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
            ptr_or_null<scalar_t>(weight), ptr_or_null<scalar_t>(bias),
            (train ? save_mean : running_mean).data_ptr<scalar_t>(),
            train ? save_invstd.data_ptr<scalar_t>() : nullptr,
            train ? nullptr : running_var.data_ptr<scalar_t>(), eps, static_cast<uint32_t>(numel), l);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void batch_norm_backward_cuda(
    const Tensor& grad_input, const Tensor& grad_weight, const Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_invstd, bool train, double eps
) {
    DeviceGuard guard(input.device());
    check_32bit(input, "batch_norm");
    const ChannelLayout l = channel_layout(input);
    const int64_t numel = input.numel();
    Tensor sums = empty({2, l.channels}, input.options());
    cudaStream_t stream = cuda::current_stream();
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* mean = (train ? save_mean : running_mean).data_ptr<scalar_t>();
        const scalar_t* invstd = train ? save_invstd.data_ptr<scalar_t>() : nullptr;
        const scalar_t* var = train ? nullptr : running_var.data_ptr<scalar_t>();
        scalar_t* sum_dy = sums.mutable_data_ptr<scalar_t>();
        scalar_t* dotp = sum_dy + l.channels;
        // Eval mode needs no sums for grad_input
        if (train || grad_weight.defined() || grad_bias.defined()) {
            batch_norm_backward_reduce_kernel<scalar_t><<<l.channels, kThreads, 0, stream>>>(
                sum_dy, dotp, mutable_ptr_or_null<scalar_t>(grad_weight), mutable_ptr_or_null<scalar_t>(grad_bias),
                grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), mean, invstd, var, eps, l);
            RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (grad_input.defined()) {
            batch_norm_backward_input_kernel<scalar_t><<<blocks_for(numel), kThreads, 0, stream>>>(
                grad_input.mutable_data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                ptr_or_null<scalar_t>(weight), mean, invstd, var, eps, sum_dy, dotp,
                static_cast<uint32_t>(numel), l);
            RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });
}

// ---------------------------------------------------------------------------
// layer_norm. Contiguous [M, N]: one block per row for the moments and the
// row-wise backward terms; the weight and bias gradients are column sums
// over the M rows, reduced by tiles.
// ---------------------------------------------------------------------------

int row_threads(int64_t N) {
    const int64_t rounded = (N + cuda::kWarpSize - 1) / cuda::kWarpSize * cuda::kWarpSize;
    return static_cast<int>(std::clamp<int64_t>(rounded, cuda::kWarpSize, kMaxRowThreads));
}

// Moments of the row, then y = (x - mean) * rstd * gamma + beta (the CPU
// kernel's expression order). The row is read twice; at the sizes of a
// transformer's hidden dim it stays in L1/L2 between the passes
template <typename scalar_t>
__global__ void layer_norm_forward_kernel(
    scalar_t* output, scalar_t* mean, scalar_t* rstd, const scalar_t* input,
    const scalar_t* weight, const scalar_t* bias, int32_t N, scalar_t eps
) {
    __shared__ cuda::WelfordData<scalar_t> shared[cuda::kWarpSize];
    __shared__ scalar_t row_stats[2];
    const int64_t row = blockIdx.x;
    const scalar_t* x = input + row * N;
    scalar_t* y = output + row * N;

    cuda::WelfordData<scalar_t> acc{0, 0, 0};
    for (int32_t j = threadIdx.x; j < N; j += blockDim.x) {
        acc = cuda::welford_update(acc, x[j]);
    }
    acc = cuda::block_welford(acc, shared);
    if (threadIdx.x == 0) {
        const scalar_t r = scalar_t(1) / ::sqrt(acc.m2 / static_cast<scalar_t>(N) + eps);
        row_stats[0] = acc.mean;
        row_stats[1] = r;
        mean[row] = acc.mean;
        rstd[row] = r;
    }
    __syncthreads();
    const scalar_t m = row_stats[0];
    const scalar_t r = row_stats[1];
    for (int32_t j = threadIdx.x; j < N; j += blockDim.x) {
        const scalar_t gamma = weight != nullptr ? weight[j] : scalar_t(1);
        const scalar_t beta = bias != nullptr ? bias[j] : scalar_t(0);
        y[j] = (x[j] - m) * r * gamma + beta;
    }
}

// With g = dy * gamma, ds = sum(g * x) and db = sum(g) over the row:
//   dx = rstd * g + c1 * x + c2,
//   c1 = (db * mean - ds) * rstd^3 / N,  c2 = -c1 * mean - db * rstd / N
// which is rstd * (g - mean(g) - xhat * mean(g * xhat)) expanded.
// From: layer_norm_kernel.cu (ComputeInternalGradientsCUDAKernel,
// ComputeGradientFusedParamsCUDAKernel, LayerNormBackwardCUDAKernel)
template <typename scalar_t>
__global__ void layer_norm_backward_input_kernel(
    scalar_t* grad_input, const scalar_t* grad_output, const scalar_t* input,
    const scalar_t* mean, const scalar_t* rstd, const scalar_t* weight, int32_t N
) {
    __shared__ scalar_t shared[cuda::kWarpSize];
    __shared__ scalar_t coefficients[2];
    const int64_t row = blockIdx.x;
    const scalar_t* x = input + row * N;
    const scalar_t* dy = grad_output + row * N;
    scalar_t* dx = grad_input + row * N;

    scalar_t ds = 0;
    scalar_t db = 0;
    for (int32_t j = threadIdx.x; j < N; j += blockDim.x) {
        const scalar_t g = dy[j] * (weight != nullptr ? weight[j] : scalar_t(1));
        ds += g * x[j];
        db += g;
    }
    ds = cuda::block_sum(ds, shared);
    db = cuda::block_sum(db, shared);
    const scalar_t m = mean[row];
    const scalar_t r = rstd[row];
    if (threadIdx.x == 0) {
        const scalar_t scale = scalar_t(1) / static_cast<scalar_t>(N);
        const scalar_t c1 = (db * m - ds) * r * r * r * scale;
        coefficients[0] = c1;
        coefficients[1] = -c1 * m - db * r * scale;
    }
    __syncthreads();
    const scalar_t c1 = coefficients[0];
    const scalar_t c2 = coefficients[1];
    for (int32_t j = threadIdx.x; j < N; j += blockDim.x) {
        const scalar_t g = dy[j] * (weight != nullptr ? weight[j] : scalar_t(1));
        dx[j] = r * g + c1 * x[j] + c2;
    }
}

// Column sums dgamma[j] = sum_i dy[i, j] * (x[i, j] - mean[i]) * rstd[i] and
// dbeta[j] = sum_i dy[i, j]. A block is a tile of 32 columns by a chunk of
// rows: threadIdx.x walks the columns (coalesced reads of each row),
// threadIdx.y strides the rows of the chunk; the kTileRows partial sums of a
// column meet in shared memory. Each chunk writes one row of partial sums,
// added up afterwards by a sum over dim 0, so no atomics.
// From: layer_norm_kernel.cu (GammaBetaBackwardCUDAKernel), simplified
constexpr int kTileRows = 16;

template <typename scalar_t>
__global__ void layer_norm_gamma_beta_kernel(
    scalar_t* dgamma_partial, scalar_t* dbeta_partial, const scalar_t* grad_output, const scalar_t* input,
    const scalar_t* mean, const scalar_t* rstd, int32_t M, int32_t N, int32_t rows_per_chunk
) {
    __shared__ scalar_t gamma_tile[kTileRows][cuda::kWarpSize + 1];
    __shared__ scalar_t beta_tile[kTileRows][cuda::kWarpSize + 1];
    const int32_t col = blockIdx.x * cuda::kWarpSize + threadIdx.x;
    const int32_t row_begin = blockIdx.y * rows_per_chunk;
    const int32_t row_end = ::min(row_begin + rows_per_chunk, M);
    scalar_t dgamma = 0;
    scalar_t dbeta = 0;
    if (col < N) {
        for (int32_t i = row_begin + threadIdx.y; i < row_end; i += kTileRows) {
            const int64_t offset = static_cast<int64_t>(i) * N + col;
            const scalar_t dy = grad_output[offset];
            dgamma += dy * (input[offset] - mean[i]) * rstd[i];
            dbeta += dy;
        }
    }
    gamma_tile[threadIdx.y][threadIdx.x] = dgamma;
    beta_tile[threadIdx.y][threadIdx.x] = dbeta;
    __syncthreads();
    if (threadIdx.y == 0 && col < N) {
        for (int k = 1; k < kTileRows; ++k) {
            dgamma += gamma_tile[k][threadIdx.x];
            dbeta += beta_tile[k][threadIdx.x];
        }
        const int64_t out = static_cast<int64_t>(blockIdx.y) * N + col;
        if (dgamma_partial != nullptr) dgamma_partial[out] = dgamma;
        if (dbeta_partial != nullptr) dbeta_partial[out] = dbeta;
    }
}

void layer_norm_cuda(
    const Tensor& output, const Tensor& mean, const Tensor& rstd, const Tensor& input,
    const Tensor& weight, const Tensor& bias, int64_t M, int64_t N, double eps
) {
    DeviceGuard guard(input.device());
    check_32bit(input, "layer_norm");
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        layer_norm_forward_kernel<scalar_t><<<static_cast<unsigned>(M), row_threads(N), 0, cuda::current_stream()>>>(
            output.mutable_data_ptr<scalar_t>(), mean.mutable_data_ptr<scalar_t>(), rstd.mutable_data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(), ptr_or_null<scalar_t>(weight), ptr_or_null<scalar_t>(bias),
            static_cast<int32_t>(N), static_cast<scalar_t>(eps));
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void layer_norm_backward_cuda(
    const Tensor& grad_input, const Tensor& grad_weight, const Tensor& grad_bias,
    const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& rstd,
    const Tensor& weight, int64_t M, int64_t N
) {
    DeviceGuard guard(input.device());
    check_32bit(input, "layer_norm");
    cudaStream_t stream = cuda::current_stream();
    // About 128 rows per chunk, up to 1024 chunks
    const int64_t chunks = std::clamp<int64_t>((M + 127) / 128, 1, 1024);
    const auto rows_per_chunk = static_cast<int32_t>((M + chunks - 1) / chunks);
    const bool need_params = grad_weight.defined() || grad_bias.defined();
    // One chunk writes the gradients directly
    Tensor gamma_partial, beta_partial;
    if (grad_weight.defined()) {
        gamma_partial = chunks == 1 ? grad_weight : empty({chunks, N}, input.options());
    }
    if (grad_bias.defined()) {
        beta_partial = chunks == 1 ? grad_bias : empty({chunks, N}, input.options());
    }

    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if (grad_input.defined()) {
            layer_norm_backward_input_kernel<scalar_t><<<static_cast<unsigned>(M), row_threads(N), 0, stream>>>(
                grad_input.mutable_data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                mean.data_ptr<scalar_t>(), rstd.data_ptr<scalar_t>(), ptr_or_null<scalar_t>(weight),
                static_cast<int32_t>(N));
            RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
        }
        if (need_params) {
            const dim3 grid(static_cast<unsigned>((N + cuda::kWarpSize - 1) / cuda::kWarpSize),
                            static_cast<unsigned>(chunks));
            const dim3 block(cuda::kWarpSize, kTileRows);
            layer_norm_gamma_beta_kernel<scalar_t><<<grid, block, 0, stream>>>(
                mutable_ptr_or_null<scalar_t>(gamma_partial), mutable_ptr_or_null<scalar_t>(beta_partial),
                grad_output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                mean.data_ptr<scalar_t>(), rstd.data_ptr<scalar_t>(),
                static_cast<int32_t>(M), static_cast<int32_t>(N), rows_per_chunk);
            RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
        }
    });
    if (chunks > 1) {
        if (grad_weight.defined()) {
            grad_weight.view({N}).copy_(ops::sum(gamma_partial, IntArrayRef{0}));
        }
        if (grad_bias.defined()) {
            grad_bias.view({N}).copy_(ops::sum(beta_partial, IntArrayRef{0}));
        }
    }
}

} // anonymous namespace

REGISTER_DISPATCH(batch_norm_collect_stats_stub, &batch_norm_collect_stats_cuda);
REGISTER_DISPATCH(batch_norm_update_stats_stub, &batch_norm_update_stats_cuda);
REGISTER_DISPATCH(batch_norm_stub, &batch_norm_cuda);
REGISTER_DISPATCH(batch_norm_backward_stub, &batch_norm_backward_cuda);
REGISTER_DISPATCH(layer_norm_stub, &layer_norm_cuda);
REGISTER_DISPATCH(layer_norm_backward_stub, &layer_norm_backward_cuda);

} // namespace raxis
