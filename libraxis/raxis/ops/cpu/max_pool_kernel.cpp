#include <raxis/ops/pooling.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

// From: aten/src/ATen/native/cpu/MaxPoolKernel.cpp (cpu_max_pool,
// cpu_max_pool_backward)
//
// Contiguous NCHW: every (n, c) plane is independent, so the planes are the
// parallel dim. Each output element scans its window (with dilation) and
// keeps the first maximum; NaN wins over any number, so it propagates as in
// max().

template <typename scalar_t>
void max_pool2d_impl(scalar_t* output, int64_t* indices, const scalar_t* input,
                     int64_t planes, const Pool2dParams& p) {
    const int64_t input_plane = p.input_height * p.input_width;
    const int64_t output_plane = p.output_height * p.output_width;
    constexpr scalar_t lowest = std::numeric_limits<scalar_t>::has_infinity
        ? -std::numeric_limits<scalar_t>::infinity()
        : std::numeric_limits<scalar_t>::lowest();

    parallel_for(0, planes, 0, [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            const scalar_t* in = input + c * input_plane;
            scalar_t* out = output + c * output_plane;
            int64_t* ind = indices + c * output_plane;

            for (int64_t oh = 0; oh < p.output_height; ++oh) {
                // Window rows [ih0, ih1) with step dilation, starting at the
                // first one inside the image
                int64_t ih0 = oh * p.stride_height - p.pad_height;
                const int64_t ih1 = std::min(ih0 + (p.kernel_height - 1) * p.dilation_height + 1, p.input_height);
                if (ih0 < 0) {
                    ih0 += (-ih0 + p.dilation_height - 1) / p.dilation_height * p.dilation_height;
                }
                for (int64_t ow = 0; ow < p.output_width; ++ow) {
                    int64_t iw0 = ow * p.stride_width - p.pad_width;
                    const int64_t iw1 = std::min(iw0 + (p.kernel_width - 1) * p.dilation_width + 1, p.input_width);
                    if (iw0 < 0) {
                        iw0 += (-iw0 + p.dilation_width - 1) / p.dilation_width * p.dilation_width;
                    }

                    int64_t max_index = ih0 * p.input_width + iw0;
                    scalar_t max_value = lowest;
                    for (int64_t ih = ih0; ih < ih1; ih += p.dilation_height) {
                        for (int64_t iw = iw0; iw < iw1; iw += p.dilation_width) {
                            const int64_t index = ih * p.input_width + iw;
                            const scalar_t value = in[index];
                            bool is_nan = false;
                            if constexpr (std::is_floating_point_v<scalar_t>) {
                                is_nan = std::isnan(value);
                            }
                            if (value > max_value || is_nan) {
                                max_value = value;
                                max_index = index;
                            }
                        }
                    }
                    out[oh * p.output_width + ow] = max_value;
                    ind[oh * p.output_width + ow] = max_index;
                }
            }
        }
    });
}

// Scatter: each output gradient is added at its max. Windows overlap when
// stride < kernel, so a position can receive several gradients; planes do
// not share positions, so they still run in parallel.
template <typename scalar_t>
void max_pool2d_backward_impl(scalar_t* grad_input, const scalar_t* grad_output, const int64_t* indices,
                              int64_t planes, int64_t input_plane, int64_t output_plane) {
    parallel_for(0, planes, 0, [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            scalar_t* grad_in = grad_input + c * input_plane;
            const scalar_t* grad_out = grad_output + c * output_plane;
            const int64_t* ind = indices + c * output_plane;
            std::fill_n(grad_in, input_plane, scalar_t(0));
            for (int64_t i = 0; i < output_plane; ++i) {
                grad_in[ind[i]] += grad_out[i];
            }
        }
    });
}

void max_pool2d_kernel(const Tensor& output, const Tensor& indices, const Tensor& input, const Pool2dParams& params) {
    const int64_t planes = input.numel() / (params.input_height * params.input_width);
    input.dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        max_pool2d_impl<scalar_t>(
            output.mutable_data_ptr<scalar_t>(), indices.mutable_data_ptr<int64_t>(),
            input.data_ptr<scalar_t>(), planes, params);
    });
}

void max_pool2d_backward_kernel(const Tensor& grad_input, const Tensor& grad_output, const Tensor& indices) {
    const int64_t input_plane = grad_input.shape()[grad_input.ndim() - 2] * grad_input.shape()[grad_input.ndim() - 1];
    const int64_t output_plane = grad_output.shape()[grad_output.ndim() - 2] * grad_output.shape()[grad_output.ndim() - 1];
    const int64_t planes = grad_input.numel() / input_plane;
    grad_input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        max_pool2d_backward_impl<scalar_t>(
            grad_input.mutable_data_ptr<scalar_t>(), grad_output.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>(), planes, input_plane, output_plane);
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(max_pool2d_stub, &max_pool2d_kernel);
REGISTER_DISPATCH(max_pool2d_backward_stub, &max_pool2d_backward_kernel);

} // namespace raxis
