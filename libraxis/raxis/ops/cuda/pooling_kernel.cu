#include <cstdint>
#include <limits>
#include <optional>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/integer_divider.cuh>
#include <raxis/ops/pooling.h>
#include <raxis/tensor/tensor.h>


// max_pool2d and avg_pool2d on contiguous NCHW tensors, with the CPU
// kernels' semantics (ops/cpu/max_pool_kernel.cpp, avg_pool_kernel.cpp).
// From: aten/src/ATen/native/cuda/DilatedMaxPool2d.cu (max_pool_forward_nchw,
// max_pool_backward_nchw), AveragePool2d.cu (avg_pool2d_out_cuda_frame,
// avg_pool2d_backward_out_cuda_frame)
//
// Forward: one thread per output element scans its window. Backward: one
// thread per input element gathers from the outputs whose window covers it
// (a range of rows and columns of the output plane), so nothing is
// accumulated with atomics and the result does not depend on scheduling.
// The CPU backward scatters instead: a serial loop per plane needs no
// gather.

namespace raxis {
namespace {

constexpr int kThreads = 256;

// 32-bit geometry with magic-number dividers for the flat thread index
struct PoolGeometry {
    int32_t height, width, out_height, out_width;
    int32_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w;
    cuda::IntDivider width_div, height_div, out_width_div, out_height_div;
};

PoolGeometry make_geometry(const Pool2dParams& p, int64_t numel) {
    RAXIS_CHECK(numel <= std::numeric_limits<int32_t>::max(),
        "pooling: tensors over 2^31 elements are not supported on CUDA yet");
    PoolGeometry g;
    g.height = static_cast<int32_t>(p.input_height);
    g.width = static_cast<int32_t>(p.input_width);
    g.out_height = static_cast<int32_t>(p.output_height);
    g.out_width = static_cast<int32_t>(p.output_width);
    g.kernel_h = static_cast<int32_t>(p.kernel_height);
    g.kernel_w = static_cast<int32_t>(p.kernel_width);
    g.stride_h = static_cast<int32_t>(p.stride_height);
    g.stride_w = static_cast<int32_t>(p.stride_width);
    g.pad_h = static_cast<int32_t>(p.pad_height);
    g.pad_w = static_cast<int32_t>(p.pad_width);
    g.dilation_h = static_cast<int32_t>(p.dilation_height);
    g.dilation_w = static_cast<int32_t>(p.dilation_width);
    g.width_div = cuda::IntDivider(g.width);
    g.height_div = cuda::IntDivider(g.height);
    g.out_width_div = cuda::IntDivider(g.out_width);
    g.out_height_div = cuda::IntDivider(g.out_height);
    return g;
}

unsigned blocks_for(int64_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

// The window starts at the first tap inside the image; NaN wins and the
// first maximum is kept, as on the CPU
template <typename scalar_t>
__global__ void max_pool2d_kernel(
    scalar_t* output, int64_t* indices, const scalar_t* input, uint32_t n, PoolGeometry g
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const auto ow = g.out_width_div.divmod(index);
    const auto oh = g.out_height_div.divmod(ow.div);
    const scalar_t* in = input + static_cast<int64_t>(oh.div) * g.height * g.width;

    int32_t h0 = oh.mod * g.stride_h - g.pad_h;
    int32_t w0 = ow.mod * g.stride_w - g.pad_w;
    const int32_t h1 = ::min(h0 + (g.kernel_h - 1) * g.dilation_h + 1, g.height);
    const int32_t w1 = ::min(w0 + (g.kernel_w - 1) * g.dilation_w + 1, g.width);
    while (h0 < 0) h0 += g.dilation_h;
    while (w0 < 0) w0 += g.dilation_w;

    scalar_t max_value = std::numeric_limits<scalar_t>::has_infinity
        ? -std::numeric_limits<scalar_t>::infinity()
        : std::numeric_limits<scalar_t>::lowest();
    int32_t max_index = h0 * g.width + w0;
    for (int32_t h = h0; h < h1; h += g.dilation_h) {
        for (int32_t w = w0; w < w1; w += g.dilation_w) {
            const int32_t i = h * g.width + w;
            const scalar_t value = in[i];
            if (value > max_value || value != value) {
                max_value = value;
                max_index = i;
            }
        }
    }
    output[index] = max_value;
    indices[index] = max_index;
}

// The outputs whose (dilated) window can reach position x: [start, end)
__device__ inline int32_t pool_start(int32_t x, int32_t pad, int32_t kernel, int32_t dilation, int32_t stride) {
    const int32_t extent = (kernel - 1) * dilation + 1;
    return x + pad < extent ? 0 : (x + pad - extent) / stride + 1;
}

__device__ inline int32_t pool_end(int32_t x, int32_t pad, int32_t out_size, int32_t stride) {
    return ::min((x + pad) / stride + 1, out_size);
}

template <typename scalar_t>
__global__ void max_pool2d_backward_kernel(
    scalar_t* grad_input, const scalar_t* grad_output, const int64_t* indices, uint32_t n, PoolGeometry g
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const auto w = g.width_div.divmod(index);
    const auto h = g.height_div.divmod(w.div);
    const int64_t offset = static_cast<int64_t>(h.div) * g.out_height * g.out_width;
    const int64_t position = h.mod * g.width + w.mod;
    const int32_t oh0 = pool_start(h.mod, g.pad_h, g.kernel_h, g.dilation_h, g.stride_h);
    const int32_t oh1 = pool_end(h.mod, g.pad_h, g.out_height, g.stride_h);
    const int32_t ow0 = pool_start(w.mod, g.pad_w, g.kernel_w, g.dilation_w, g.stride_w);
    const int32_t ow1 = pool_end(w.mod, g.pad_w, g.out_width, g.stride_w);
    scalar_t gradient = 0;
    for (int32_t oh = oh0; oh < oh1; ++oh) {
        for (int32_t ow = ow0; ow < ow1; ++ow) {
            const int64_t o = offset + oh * g.out_width + ow;
            if (indices[o] == position) {
                gradient += grad_output[o];
            }
        }
    }
    grad_input[index] = gradient;
}

// Window clipped to the padded image (pool_size) and to the image (summed)
struct AvgWindow {
    int32_t h0, h1, w0, w1;
    int32_t divisor;
};

__device__ inline AvgWindow avg_window(
    int32_t oh, int32_t ow, const PoolGeometry& g, bool count_include_pad, int32_t divisor_override
) {
    int32_t h0 = oh * g.stride_h - g.pad_h;
    int32_t w0 = ow * g.stride_w - g.pad_w;
    int32_t h1 = ::min(h0 + g.kernel_h, g.height + g.pad_h);
    int32_t w1 = ::min(w0 + g.kernel_w, g.width + g.pad_w);
    const int32_t pool_size = (h1 - h0) * (w1 - w0);
    h0 = ::max(h0, 0);
    w0 = ::max(w0, 0);
    h1 = ::min(h1, g.height);
    w1 = ::min(w1, g.width);
    int32_t divisor;
    if (divisor_override != 0) {
        divisor = divisor_override;
    } else if (count_include_pad) {
        divisor = pool_size;
    } else {
        divisor = (h1 - h0) * (w1 - w0);
    }
    return {h0, h1, w0, w1, divisor};
}

template <typename scalar_t>
__global__ void avg_pool2d_kernel(
    scalar_t* output, const scalar_t* input, uint32_t n, PoolGeometry g,
    bool count_include_pad, int32_t divisor_override
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const auto ow = g.out_width_div.divmod(index);
    const auto oh = g.out_height_div.divmod(ow.div);
    const scalar_t* in = input + static_cast<int64_t>(oh.div) * g.height * g.width;
    const AvgWindow win = avg_window(oh.mod, ow.mod, g, count_include_pad, divisor_override);
    // A window entirely in the padding (only with ceil_mode)
    if (win.h0 >= win.h1 || win.w0 >= win.w1) {
        output[index] = scalar_t(0);
        return;
    }
    scalar_t sum = 0;
    for (int32_t h = win.h0; h < win.h1; ++h) {
        for (int32_t w = win.w0; w < win.w1; ++w) {
            sum += in[h * g.width + w];
        }
    }
    output[index] = sum / static_cast<scalar_t>(win.divisor);
}

template <typename scalar_t>
__global__ void avg_pool2d_backward_kernel(
    scalar_t* grad_input, const scalar_t* grad_output, uint32_t n, PoolGeometry g,
    bool count_include_pad, int32_t divisor_override
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const auto w = g.width_div.divmod(index);
    const auto h = g.height_div.divmod(w.div);
    const scalar_t* grad_out = grad_output + static_cast<int64_t>(h.div) * g.out_height * g.out_width;
    const int32_t oh0 = pool_start(h.mod, g.pad_h, g.kernel_h, 1, g.stride_h);
    const int32_t oh1 = pool_end(h.mod, g.pad_h, g.out_height, g.stride_h);
    const int32_t ow0 = pool_start(w.mod, g.pad_w, g.kernel_w, 1, g.stride_w);
    const int32_t ow1 = pool_end(w.mod, g.pad_w, g.out_width, g.stride_w);
    scalar_t gradient = 0;
    for (int32_t oh = oh0; oh < oh1; ++oh) {
        for (int32_t ow = ow0; ow < ow1; ++ow) {
            const AvgWindow win = avg_window(oh, ow, g, count_include_pad, divisor_override);
            gradient += grad_out[oh * g.out_width + ow] / static_cast<scalar_t>(win.divisor);
        }
    }
    grad_input[index] = gradient;
}

int32_t divisor_or_zero(std::optional<int64_t> divisor_override) {
    return static_cast<int32_t>(divisor_override.value_or(0));
}

void max_pool2d_kernel_cuda(const Tensor& output, const Tensor& indices, const Tensor& input, const Pool2dParams& params) {
    DeviceGuard guard(input.device());
    const int64_t n = output.numel();
    const PoolGeometry g = make_geometry(params, input.numel());
    input.dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        max_pool2d_kernel<scalar_t><<<blocks_for(n), kThreads, 0, cuda::current_stream()>>>(
            output.mutable_data_ptr<scalar_t>(), indices.mutable_data_ptr<int64_t>(),
            input.data_ptr<scalar_t>(), static_cast<uint32_t>(n), g);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void max_pool2d_backward_kernel_cuda(
    const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices, const Pool2dParams& params
) {
    DeviceGuard guard(grad_input.device());
    const int64_t n = grad_input.numel();
    const PoolGeometry g = make_geometry(params, n);
    grad_input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        max_pool2d_backward_kernel<scalar_t><<<blocks_for(n), kThreads, 0, cuda::current_stream()>>>(
            grad_input.mutable_data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(), static_cast<uint32_t>(n), g);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void avg_pool2d_kernel_cuda(const Tensor& output, const Tensor& input, const Pool2dParams& params,
                            bool count_include_pad, std::optional<int64_t> divisor_override) {
    DeviceGuard guard(input.device());
    const int64_t n = output.numel();
    const PoolGeometry g = make_geometry(params, input.numel());
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        avg_pool2d_kernel<scalar_t><<<blocks_for(n), kThreads, 0, cuda::current_stream()>>>(
            output.mutable_data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), static_cast<uint32_t>(n), g,
            count_include_pad, divisor_or_zero(divisor_override));
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void avg_pool2d_backward_kernel_cuda(const Tensor& grad_input, const Tensor& grad_output, const Pool2dParams& params,
                                     bool count_include_pad, std::optional<int64_t> divisor_override) {
    DeviceGuard guard(grad_input.device());
    const int64_t n = grad_input.numel();
    const PoolGeometry g = make_geometry(params, n);
    grad_input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        avg_pool2d_backward_kernel<scalar_t><<<blocks_for(n), kThreads, 0, cuda::current_stream()>>>(
            grad_input.mutable_data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), static_cast<uint32_t>(n), g,
            count_include_pad, divisor_or_zero(divisor_override));
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // anonymous namespace

REGISTER_DISPATCH(max_pool2d_stub, &max_pool2d_kernel_cuda);
REGISTER_DISPATCH(max_pool2d_backward_stub, &max_pool2d_backward_kernel_cuda);
REGISTER_DISPATCH(avg_pool2d_stub, &avg_pool2d_kernel_cuda);
REGISTER_DISPATCH(avg_pool2d_backward_stub, &avg_pool2d_backward_kernel_cuda);

} // namespace raxis
