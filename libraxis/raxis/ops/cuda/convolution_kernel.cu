#include <algorithm>
#include <cstdint>
#include <limits>

#include <raxis/core/context.h>
#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_blas.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/convolution.h>
#include <raxis/ops/cuda/integer_divider.cuh>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


// Convolution on the GPU as unfold (im2col) + cuBLAS gemm, PyTorch's
// slow_conv2d CUDA backend. PyTorch itself sends float convolutions to cuDNN
// (implicit gemm and Winograd / FFT kernels picked per shape); this is the
// path it falls back to without cuDNN, kept because it is the same algorithm
// as the CPU one and readable.
// From: aten/src/ATen/native/cuda/ConvolutionMM2d.cu, im2col.cuh
//
// PyTorch loops over the images: one im2col and one gemm per image, so a
// batch of 128 is 256 launches per convolution. Here a chunk of images is
// unfolded by one launch into one buffer, and each group is one strided
// batched gemm over the chunk (the weight is the same matrix for every
// image: batch stride 0). Chunks keep the column buffer under kColumnsBudget.
//
// Column layout. Row r = (c, i, j) of the unfolded image is kernel tap (i, j)
// of channel c; column p is output position (oh, ow). Forward and input
// gradient keep each image's matrix whole, [N, K, P]. The weight gradient
// sums over images and positions, so it wants them adjacent, [K, N, P]:
// then grad_weight [O, K] = grad_output [O, N * P] @ columns^T [N * P, K] is
// one gemm over the whole chunk instead of one accumulating gemm per image.
// The kernels take the two strides of the layout (image, row).

namespace raxis {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kColumnsBudget = int64_t{128} << 20;  // bytes

unsigned blocks_for(int64_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

// Geometry as 32-bit values with magic-number dividers: every launch is
// bounded by kColumnsBudget, far below 2^31 elements
struct Conv2dGeometry {
    int32_t channels, height, width;
    int32_t kernel_height, kernel_width;
    int32_t pad_height, pad_width;
    int32_t stride_height, stride_width;
    int32_t dilation_height, dilation_width;
    int32_t out_height, out_width;
    cuda::IntDivider width_div, height_div, channels_div;
    cuda::IntDivider out_width_div, out_height_div;
};

Conv2dGeometry make_geometry(const Unfold2dParams& p) {
    auto narrow = [](int64_t v) {
        RAXIS_CHECK(v <= std::numeric_limits<int32_t>::max(), "convolution: size ", v, " does not fit in 32 bits");
        return static_cast<int32_t>(v);
    };
    Conv2dGeometry g;
    g.channels = narrow(p.channels);
    g.height = narrow(p.input_height);
    g.width = narrow(p.input_width);
    g.kernel_height = narrow(p.kernel_height);
    g.kernel_width = narrow(p.kernel_width);
    g.pad_height = narrow(p.pad_height);
    g.pad_width = narrow(p.pad_width);
    g.stride_height = narrow(p.stride_height);
    g.stride_width = narrow(p.stride_width);
    g.dilation_height = narrow(p.dilation_height);
    g.dilation_width = narrow(p.dilation_width);
    g.out_height = narrow(p.output_height);
    g.out_width = narrow(p.output_width);
    g.width_div = cuda::IntDivider(g.width);
    g.height_div = cuda::IntDivider(g.height);
    g.channels_div = cuda::IntDivider(g.channels);
    g.out_width_div = cuda::IntDivider(g.out_width);
    g.out_height_div = cuda::IntDivider(g.out_height);
    return g;
}

// One thread per (image, channel, output position): it copies the
// kh * kw input values that position reads from that channel, zero where
// they fall in the padding. Consecutive threads write consecutive columns.
// From: aten/src/ATen/native/cuda/im2col.cuh (im2col_kernel)
template <typename scalar_t>
__global__ void im2col_kernel(
    scalar_t* columns, const scalar_t* images, uint32_t n, Conv2dGeometry g,
    int64_t col_image_stride, int64_t col_row_stride
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const auto w = g.out_width_div.divmod(index);
    const auto h = g.out_height_div.divmod(w.div);
    const auto c = g.channels_div.divmod(h.div);
    const int32_t image = c.div;
    const int32_t channel = c.mod;
    const int32_t h_in = h.mod * g.stride_height - g.pad_height;
    const int32_t w_in = w.mod * g.stride_width - g.pad_width;

    const scalar_t* im = images + (static_cast<int64_t>(image) * g.channels + channel) * g.height * g.width;
    scalar_t* col = columns + image * col_image_stride
        + static_cast<int64_t>(channel) * g.kernel_height * g.kernel_width * col_row_stride
        + h.mod * g.out_width + w.mod;
    for (int32_t i = 0; i < g.kernel_height; ++i) {
        const int32_t hh = h_in + i * g.dilation_height;
        for (int32_t j = 0; j < g.kernel_width; ++j) {
            const int32_t ww = w_in + j * g.dilation_width;
            *col = (hh >= 0 && ww >= 0 && hh < g.height && ww < g.width) ? im[hh * g.width + ww] : scalar_t(0);
            col += col_row_stride;
        }
    }
}

// The adjoint, as a gather: one thread per input element sums every column
// entry that was copied from it, so there are no atomics and the sum order
// is fixed. The output positions whose window covers (h, w) form a range in
// each dim; with dilation only the taps landing exactly on it count.
// From: aten/src/ATen/native/cuda/im2col.cuh (col2im_kernel)
template <typename scalar_t>
__global__ void col2im_kernel(
    scalar_t* images, const scalar_t* columns, uint32_t n, Conv2dGeometry g,
    int64_t col_image_stride, int64_t col_row_stride
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const auto w = g.width_div.divmod(index);
    const auto h = g.height_div.divmod(w.div);
    const auto c = g.channels_div.divmod(h.div);
    const int32_t w_im = w.mod + g.pad_width;
    const int32_t h_im = h.mod + g.pad_height;
    const int32_t extent_w = (g.kernel_width - 1) * g.dilation_width + 1;
    const int32_t extent_h = (g.kernel_height - 1) * g.dilation_height + 1;
    const int32_t w_col_start = w_im < extent_w ? 0 : (w_im - extent_w) / g.stride_width + 1;
    const int32_t w_col_end = ::min(w_im / g.stride_width + 1, g.out_width);
    const int32_t h_col_start = h_im < extent_h ? 0 : (h_im - extent_h) / g.stride_height + 1;
    const int32_t h_col_end = ::min(h_im / g.stride_height + 1, g.out_height);

    const scalar_t* col = columns + c.div * col_image_stride;
    scalar_t sum = 0;
    for (int32_t h_col = h_col_start; h_col < h_col_end; ++h_col) {
        for (int32_t w_col = w_col_start; w_col < w_col_end; ++w_col) {
            int32_t h_k = h_im - h_col * g.stride_height;
            int32_t w_k = w_im - w_col * g.stride_width;
            if (h_k % g.dilation_height == 0 && w_k % g.dilation_width == 0) {
                h_k /= g.dilation_height;
                w_k /= g.dilation_width;
                const int64_t row = (static_cast<int64_t>(c.mod) * g.kernel_height + h_k) * g.kernel_width + w_k;
                sum += col[row * col_row_stride + h_col * g.out_width + w_col];
            }
        }
    }
    images[index] = sum;
}

// Images per chunk: as many as fit in the column budget, at least one
int64_t images_per_chunk(const Conv2dProblem& problem, int64_t itemsize) {
    const int64_t image_columns = problem.params.channels * problem.params.kernel_height
        * problem.params.kernel_width * problem.output_plane();
    RAXIS_CHECK(std::max(image_columns, problem.input_numel()) <= std::numeric_limits<int32_t>::max(),
        "convolution: one image of ", image_columns, " columns does not fit in 32-bit indexing");
    const int64_t fit = std::max<int64_t>(1, kColumnsBudget / std::max<int64_t>(1, image_columns * itemsize));
    return std::min(fit, problem.batch_size);
}

template <typename scalar_t>
void im2col(scalar_t* columns, const scalar_t* images, int64_t num_images, const Conv2dGeometry& g,
            int64_t col_image_stride, int64_t col_row_stride) {
    const int64_t n = num_images * g.channels * g.out_height * g.out_width;
    im2col_kernel<scalar_t><<<blocks_for(n), kThreads, 0, cuda::current_stream()>>>(
        columns, images, static_cast<uint32_t>(n), g, col_image_stride, col_row_stride);
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void col2im(scalar_t* images, const scalar_t* columns, int64_t num_images, const Conv2dGeometry& g,
            int64_t col_image_stride, int64_t col_row_stride) {
    const int64_t n = num_images * g.channels * g.height * g.width;
    col2im_kernel<scalar_t><<<blocks_for(n), kThreads, 0, cuda::current_stream()>>>(
        images, columns, static_cast<uint32_t>(n), g, col_image_stride, col_row_stride);
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

// Row-major output[n][g] = weight[g] @ columns[n][g] as the column-major
// output^T = columns^T @ weight^T (see ops/cuda/blas.cpp), batched over the
// chunk's images. The bias is copied into output first and kept with beta = 1.
template <typename scalar_t>
void conv2d_forward_impl(const Tensor& output, const Conv2dProblem& problem, const Tensor& bias) {
    const Conv2dGeometry geometry = make_geometry(problem.params);
    const int64_t P = problem.output_plane();
    const int64_t K = problem.params.channels * problem.params.kernel_height * problem.params.kernel_width;
    const int64_t O = problem.out_channels;
    const int64_t Kg = problem.group_rows();
    const int64_t Og = problem.group_out_channels();
    const int64_t chunk = images_per_chunk(problem, sizeof(scalar_t));
    const bool allow_tf32 = allow_tf32_conv();

    if (bias.defined()) {
        output.copy_(bias.view({1, O, 1, 1}).expand(output.shape()));
    }
    const scalar_t beta = bias.defined() ? scalar_t(1) : scalar_t(0);
    const scalar_t* input = problem.input.data_ptr<scalar_t>();
    const scalar_t* weight = problem.weight.data_ptr<scalar_t>();
    scalar_t* out = output.mutable_data_ptr<scalar_t>();
    Tensor buffer = problem.is_pointwise ? Tensor() : empty({chunk * K * P}, problem.input.options());

    for (int64_t n0 = 0; n0 < problem.batch_size; n0 += chunk) {
        const int64_t images = std::min(chunk, problem.batch_size - n0);
        const scalar_t* image = input + n0 * problem.input_numel();
        // A pointwise image is its own [C, P] column matrix (K = C)
        const scalar_t* columns = image;
        if (!problem.is_pointwise) {
            scalar_t* buf = buffer.mutable_data_ptr<scalar_t>();
            im2col(buf, image, images, geometry, K * P, P);
            columns = buf;
        }
        for (int64_t g = 0; g < problem.groups; ++g) {
            cuda::blas::bgemm<scalar_t>(
                'n', 'n', P, Og, Kg, scalar_t(1),
                columns + g * Kg * P, P, K * P,
                weight + g * Og * Kg, Kg, 0,
                beta, out + n0 * O * P + g * Og * P, P, O * P, images, allow_tf32);
        }
    }
}

// grad_columns[n][g] = weight[g]^T @ grad_output[n][g], then col2im adds
// every column entry back into the input position it came from
template <typename scalar_t>
void conv2d_backward_input_impl(const Tensor& grad_input, const Tensor& grad_output, const Conv2dProblem& problem) {
    const Conv2dGeometry geometry = make_geometry(problem.params);
    const int64_t P = problem.output_plane();
    const int64_t K = problem.params.channels * problem.params.kernel_height * problem.params.kernel_width;
    const int64_t O = problem.out_channels;
    const int64_t Kg = problem.group_rows();
    const int64_t Og = problem.group_out_channels();
    const int64_t chunk = images_per_chunk(problem, sizeof(scalar_t));
    const bool allow_tf32 = allow_tf32_conv();

    const scalar_t* grad_out = grad_output.data_ptr<scalar_t>();
    const scalar_t* weight = problem.weight.data_ptr<scalar_t>();
    scalar_t* grad_in = grad_input.mutable_data_ptr<scalar_t>();
    Tensor buffer = problem.is_pointwise ? Tensor() : empty({chunk * K * P}, problem.input.options());

    for (int64_t n0 = 0; n0 < problem.batch_size; n0 += chunk) {
        const int64_t images = std::min(chunk, problem.batch_size - n0);
        scalar_t* grad_image = grad_in + n0 * problem.input_numel();
        scalar_t* grad_columns = problem.is_pointwise ? grad_image : buffer.mutable_data_ptr<scalar_t>();
        for (int64_t g = 0; g < problem.groups; ++g) {
            cuda::blas::bgemm<scalar_t>(
                'n', 't', P, Kg, Og, scalar_t(1),
                grad_out + n0 * O * P + g * Og * P, P, O * P,
                weight + g * Og * Kg, Kg, 0,
                scalar_t(0), grad_columns + g * Kg * P, P, K * P, images, allow_tf32);
        }
        if (!problem.is_pointwise) {
            col2im(grad_image, grad_columns, images, geometry, K * P, P);
        }
    }
}

// grad_weight[g] = sum over images and positions of grad_output[g] x
// columns[g]: with both laid out [rows, N * P] it is one gemm per chunk,
// the chunks accumulating with beta = 1. grad_output is [N, O, P] and is
// transposed to [O, N, P] by a copy; columns are unfolded straight into
// [K, N, P].
template <typename scalar_t>
Tensor conv2d_backward_weight_impl(const Tensor& grad_output, const Conv2dProblem& problem) {
    const Conv2dGeometry geometry = make_geometry(problem.params);
    const int64_t P = problem.output_plane();
    const int64_t K = problem.params.channels * problem.params.kernel_height * problem.params.kernel_width;
    const int64_t O = problem.out_channels;
    const int64_t Kg = problem.group_rows();
    const int64_t Og = problem.group_out_channels();
    const int64_t chunk = images_per_chunk(problem, sizeof(scalar_t));
    const bool allow_tf32 = allow_tf32_conv();

    Tensor grad_weight = empty({O * Kg}, problem.input.options());
    scalar_t* grad_w = grad_weight.mutable_data_ptr<scalar_t>();
    const scalar_t* input = problem.input.data_ptr<scalar_t>();
    Tensor buffer = empty({chunk * K * P}, problem.input.options());
    scalar_t* columns = buffer.mutable_data_ptr<scalar_t>();

    for (int64_t n0 = 0; n0 < problem.batch_size; n0 += chunk) {
        const int64_t images = std::min(chunk, problem.batch_size - n0);
        const int64_t NP = images * P;
        im2col(columns, input + n0 * problem.input_numel(), images, geometry, P, NP);
        const Tensor grad_out_t = grad_output.narrow(0, n0, images).view({images, O, P}).transpose(0, 1).contiguous();
        const scalar_t* gout = grad_out_t.data_ptr<scalar_t>();
        const scalar_t beta = n0 == 0 ? scalar_t(0) : scalar_t(1);
        for (int64_t g = 0; g < problem.groups; ++g) {
            cuda::blas::gemm<scalar_t>(
                't', 'n', Kg, Og, NP, scalar_t(1),
                columns + g * Kg * NP, NP,
                gout + g * Og * NP, NP,
                beta, grad_w + g * Og * Kg, Kg, allow_tf32);
        }
    }
    return grad_weight;
}

void conv2d_forward_cuda(const Tensor& output, const Conv2dProblem& problem, const Tensor& bias) {
    DeviceGuard guard(output.device());
    problem.input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        conv2d_forward_impl<scalar_t>(output, problem, bias);
    });
}

void conv2d_backward_input_cuda(const Tensor& grad_input, const Tensor& grad_output, const Conv2dProblem& problem) {
    DeviceGuard guard(grad_input.device());
    problem.input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        conv2d_backward_input_impl<scalar_t>(grad_input, grad_output, problem);
    });
}

Tensor conv2d_backward_weight_cuda(const Tensor& grad_output, const Conv2dProblem& problem) {
    DeviceGuard guard(grad_output.device());
    if (problem.batch_size == 0) {
        return zeros({problem.out_channels * problem.group_rows()}, problem.input.options());
    }
    Tensor grad_weight;
    problem.input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        grad_weight = conv2d_backward_weight_impl<scalar_t>(grad_output, problem);
    });
    return grad_weight;
}

} // anonymous namespace

REGISTER_DISPATCH(conv2d_forward_stub, &conv2d_forward_cuda);
REGISTER_DISPATCH(conv2d_backward_input_stub, &conv2d_backward_input_cuda);
REGISTER_DISPATCH(conv2d_backward_weight_stub, &conv2d_backward_weight_cuda);

} // namespace raxis
