#include <raxis/ops/pooling.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/parallel.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

// From: aten/src/ATen/native/cpu/AvgPoolKernel.cpp (cpu_avg_pool2d,
// cpu_avg_pool2d_backward)
//
// The window of an output element, clipped to the padded image, gives
// pool_size; clipped to the image itself, the elements that are summed.
// The divisor is divisor_override if given, else pool_size when the padding
// counts (count_include_pad) or the number of summed elements when not.

struct AvgWindow {
    int64_t ih0, ih1, iw0, iw1;
    int64_t divisor;
};

inline AvgWindow avg_window(int64_t oh, int64_t ow, const Pool2dParams& p,
                            bool count_include_pad, std::optional<int64_t> divisor_override) {
    int64_t ih0 = oh * p.stride_height - p.pad_height;
    int64_t iw0 = ow * p.stride_width - p.pad_width;
    int64_t ih1 = std::min(ih0 + p.kernel_height, p.input_height + p.pad_height);
    int64_t iw1 = std::min(iw0 + p.kernel_width, p.input_width + p.pad_width);
    const int64_t pool_size = (ih1 - ih0) * (iw1 - iw0);
    ih0 = std::max<int64_t>(ih0, 0);
    iw0 = std::max<int64_t>(iw0, 0);
    ih1 = std::min(ih1, p.input_height);
    iw1 = std::min(iw1, p.input_width);

    int64_t divisor;
    if (divisor_override.has_value()) {
        divisor = divisor_override.value();
    } else if (count_include_pad) {
        divisor = pool_size;
    } else {
        divisor = (ih1 - ih0) * (iw1 - iw0);
    }
    return {ih0, ih1, iw0, iw1, divisor};
}

template <typename scalar_t>
void avg_pool2d_impl(scalar_t* output, const scalar_t* input, int64_t planes, const Pool2dParams& p,
                     bool count_include_pad, std::optional<int64_t> divisor_override) {
    const int64_t input_plane = p.input_height * p.input_width;
    const int64_t output_plane = p.output_height * p.output_width;

    parallel_for(0, planes, 0, [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            const scalar_t* in = input + c * input_plane;
            scalar_t* out = output + c * output_plane;
            for (int64_t oh = 0; oh < p.output_height; ++oh) {
                for (int64_t ow = 0; ow < p.output_width; ++ow) {
                    const AvgWindow w = avg_window(oh, ow, p, count_include_pad, divisor_override);
                    // A window entirely in the padding (only with ceil_mode)
                    if (w.ih0 >= w.ih1 || w.iw0 >= w.iw1) {
                        out[oh * p.output_width + ow] = scalar_t(0);
                        continue;
                    }
                    scalar_t sum = 0;
                    for (int64_t ih = w.ih0; ih < w.ih1; ++ih) {
                        for (int64_t iw = w.iw0; iw < w.iw1; ++iw) {
                            sum += in[ih * p.input_width + iw];
                        }
                    }
                    out[oh * p.output_width + ow] = sum / w.divisor;
                }
            }
        }
    });
}

// Each output gradient is spread evenly over the elements of its window
template <typename scalar_t>
void avg_pool2d_backward_impl(scalar_t* grad_input, const scalar_t* grad_output, int64_t planes,
                              const Pool2dParams& p, bool count_include_pad,
                              std::optional<int64_t> divisor_override) {
    const int64_t input_plane = p.input_height * p.input_width;
    const int64_t output_plane = p.output_height * p.output_width;

    parallel_for(0, planes, 0, [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            scalar_t* grad_in = grad_input + c * input_plane;
            const scalar_t* grad_out = grad_output + c * output_plane;
            std::fill_n(grad_in, input_plane, scalar_t(0));
            for (int64_t oh = 0; oh < p.output_height; ++oh) {
                for (int64_t ow = 0; ow < p.output_width; ++ow) {
                    const AvgWindow w = avg_window(oh, ow, p, count_include_pad, divisor_override);
                    const scalar_t delta = grad_out[oh * p.output_width + ow] / w.divisor;
                    for (int64_t ih = w.ih0; ih < w.ih1; ++ih) {
                        for (int64_t iw = w.iw0; iw < w.iw1; ++iw) {
                            grad_in[ih * p.input_width + iw] += delta;
                        }
                    }
                }
            }
        }
    });
}

void avg_pool2d_kernel(const Tensor& output, const Tensor& input, const Pool2dParams& params,
                       bool count_include_pad, std::optional<int64_t> divisor_override) {
    const int64_t planes = input.numel() / (params.input_height * params.input_width);
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        avg_pool2d_impl<scalar_t>(
            output.mutable_data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), planes, params,
            count_include_pad, divisor_override);
    });
}

void avg_pool2d_backward_kernel(const Tensor& grad_input, const Tensor& grad_output, const Pool2dParams& params,
                                bool count_include_pad, std::optional<int64_t> divisor_override) {
    const int64_t planes = grad_input.numel() / (params.input_height * params.input_width);
    grad_input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        avg_pool2d_backward_impl<scalar_t>(
            grad_input.mutable_data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(), planes, params,
            count_include_pad, divisor_override);
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(avg_pool2d_stub, &avg_pool2d_kernel);
REGISTER_DISPATCH(avg_pool2d_backward_stub, &avg_pool2d_backward_kernel);

} // namespace raxis
