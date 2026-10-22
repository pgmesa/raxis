#include <raxis/ops/convolution.h>

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

#include <raxis/ops/cpu_blas.h>
#include <raxis/ops/reduce_ops.h>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>

#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_shape.h>


namespace raxis {

namespace {

using cpublas::TransposeType;

// From: aten/src/ATen/native/Convolution.cpp (expand_param_if_needed)
DimVector expand_param(IntArrayRef param, const char* name, int64_t spatial_dims) {
    if (param.size() == 1) {
        return DimVector(static_cast<size_t>(spatial_dims), param[0]);
    }
    RAXIS_CHECK(static_cast<int64_t>(param.size()) == spatial_dims,
        "expected ", name, " to be a single integer value or a list of ", spatial_dims,
        " values to match the convolution dimensions, but got ", name, "=", param);
    return DimVector(param.begin(), param.end());
}

std::string join_sizes(int64_t height, int64_t width, bool is_1d) {
    std::ostringstream ss;
    if (!is_1d) {
        ss << height << " x ";
    }
    ss << width;
    return ss.str();
}

// A convolution as the 2-D problem the kernels solve: 1-D inputs get a
// height of 1, input and weight are contiguous.
struct Conv2dProblem {
    Tensor input;    // [N, C, H, W]
    Tensor weight;   // [O, C / groups, kh, kw]
    Unfold2dParams params;
    int64_t groups;
    int64_t batch_size;
    int64_t out_channels;
    bool is_1d;
    // 1x1 kernel, stride 1, no padding: the image already is its columns
    bool is_pointwise;

    int64_t output_plane() const { return params.output_height * params.output_width; }
    int64_t input_numel() const { return params.channels * params.input_height * params.input_width; }
    // Rows of columns (gemm inner dim) per group
    int64_t group_rows() const {
        return params.channels / groups * params.kernel_height * params.kernel_width;
    }
    int64_t group_out_channels() const { return out_channels / groups; }
};

// From: aten/src/ATen/native/Convolution.cpp (check_shape_forward, _convolution)
Conv2dProblem make_problem(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_, int64_t groups
) {
    const int64_t dim = input.ndim();
    RAXIS_CHECK(dim == 3 || dim == 4,
        "convolution: expected a 3-D or 4-D batched input, but got input of size: ", input.shape());
    RAXIS_CHECK(weight.ndim() == dim,
        "Expected ", weight.ndim(), "-dimensional input for ", weight.ndim(),
        "-dimensional weight ", weight.shape(), ", but got ", dim,
        "-dimensional input of size ", input.shape(), " instead");
    RAXIS_CHECK(input.dtype().is_floating(),
        "convolution: expected a floating point input, but got ", input.dtype());
    RAXIS_CHECK(input.dtype() == weight.dtype(),
        "Input type (", input.dtype(), ") and weight type (", weight.dtype(), ") should be the same");

    const int64_t spatial_dims = dim - 2;
    const DimVector stride = expand_param(stride_, "stride", spatial_dims);
    const DimVector padding = expand_param(padding_, "padding", spatial_dims);
    const DimVector dilation = expand_param(dilation_, "dilation", spatial_dims);
    for (int64_t i = 0; i < spatial_dims; ++i) {
        RAXIS_CHECK(stride[i] > 0, "non-positive stride is not supported");
        RAXIS_CHECK(padding[i] >= 0, "negative padding is not supported");
        RAXIS_CHECK(dilation[i] > 0, "dilation should be greater than zero");
    }
    RAXIS_CHECK(groups > 0, "non-positive groups is not supported");

    const int64_t out_channels = weight.shape()[0];
    RAXIS_CHECK(out_channels % groups == 0,
        "Given groups=", groups, ", expected weight to be divisible by ", groups,
        " at dimension 0, but got weight of size ", weight.shape(), " instead");
    RAXIS_CHECK(input.shape()[1] == weight.shape()[1] * groups,
        "Given groups=", groups, ", weight of size ", weight.shape(),
        ", expected input", input.shape(), " to have ", weight.shape()[1] * groups,
        " channels, but got ", input.shape()[1], " channels instead");
    if (bias.defined()) {
        RAXIS_CHECK(bias.ndim() == 1 && bias.shape()[0] == out_channels,
            "Given weight of size ", weight.shape(), ", expected bias to be 1-dimensional with ",
            out_channels, " elements, but got bias of size ", bias.shape(), " instead");
        RAXIS_CHECK(bias.dtype() == input.dtype(),
            "Input type (", input.dtype(), ") and bias type (", bias.dtype(), ") should be the same");
    }

    // Conv1d is a conv2d with a height of 1, as PyTorch does (view1d_as_2d)
    const bool is_1d = dim == 3;
    Conv2dProblem problem;
    problem.input = (is_1d ? ops::unsqueeze(input, 2) : input).contiguous();
    problem.weight = (is_1d ? ops::unsqueeze(weight, 2) : weight).contiguous();
    problem.groups = groups;
    problem.batch_size = input.shape()[0];
    problem.out_channels = out_channels;
    problem.is_1d = is_1d;

    Unfold2dParams& p = problem.params;
    p.channels = problem.input.shape()[1];
    p.input_height = problem.input.shape()[2];
    p.input_width = problem.input.shape()[3];
    p.kernel_height = problem.weight.shape()[2];
    p.kernel_width = problem.weight.shape()[3];
    p.stride_height = is_1d ? 1 : stride[0];
    p.stride_width = stride[spatial_dims - 1];
    p.pad_height = is_1d ? 0 : padding[0];
    p.pad_width = padding[spatial_dims - 1];
    p.dilation_height = is_1d ? 1 : dilation[0];
    p.dilation_width = dilation[spatial_dims - 1];

    const int64_t padded_height = p.input_height + 2 * p.pad_height;
    const int64_t padded_width = p.input_width + 2 * p.pad_width;
    const int64_t dilated_kernel_height = p.dilation_height * (p.kernel_height - 1) + 1;
    const int64_t dilated_kernel_width = p.dilation_width * (p.kernel_width - 1) + 1;
    RAXIS_CHECK(padded_height >= dilated_kernel_height && padded_width >= dilated_kernel_width,
        "Calculated padded input size per channel: (",
        join_sizes(padded_height, padded_width, is_1d), "). Kernel size: (",
        join_sizes(dilated_kernel_height, dilated_kernel_width, is_1d),
        "). Kernel size can't be greater than actual input size");

    p.output_height = (padded_height - dilated_kernel_height) / p.stride_height + 1;
    p.output_width = (padded_width - dilated_kernel_width) / p.stride_width + 1;

    problem.is_pointwise =
        p.kernel_height == 1 && p.kernel_width == 1 &&
        p.stride_height == 1 && p.stride_width == 1 &&
        p.pad_height == 0 && p.pad_width == 0;
    return problem;
}

// Each image is an independent unfold + gemm. With at least one image per
// thread they run in parallel, each with a single-threaded gemm (MKL runs
// sequentially inside a parallel region), as PyTorch's slow_conv2d does.
// With fewer images that would leave threads idle, so they go one by one and
// the parallelism comes from inside: the multithreaded gemm and the unfold.
template <typename F>
void for_each_image_chunk(int64_t batch_size, const F& f) {
    if (batch_size >= get_num_threads()) {
        parallel_for(0, batch_size, 0, f);
    } else {
        f(0, batch_size);
    }
}

// From: aten/src/ATen/native/ConvolutionMM2d.cpp (slow_conv2d_forward_out_cpu,
// slow_conv2d_update_output_frame)
//
// For every image: columns = unfold(image), [C * kh * kw, OH * OW], and per
// group output[g] = weight[g] @ columns[g]. Groups are row blocks of the same
// columns and weight, so they are one gemm each on sub-matrices; PyTorch
// instead splits the tensors per group and concatenates the outputs.
//
// gemm is column-major: a row-major [r, c] matrix is a column-major [c, r]
// one, so the row-major product out = W @ cols is computed as the
// column-major out^T = cols^T @ W^T without moving any data.
//
// PyTorch unfolds the whole batch into one [N, C * kh * kw, OH * OW] buffer
// before the gemms; here each thread reuses one image-sized buffer, which
// stays in cache between the unfold and the gemm that reads it.
template <typename scalar_t>
void conv2d_forward(Tensor& output, const Conv2dProblem& problem, const Tensor& bias) {
    const Unfold2dParams& p = problem.params;
    const int64_t output_plane = problem.output_plane();
    const int64_t input_numel = problem.input_numel();
    const int64_t group_rows = problem.group_rows();
    const int64_t group_out = problem.group_out_channels();
    const int64_t out_channels = problem.out_channels;
    const int64_t groups = problem.groups;
    const bool is_pointwise = problem.is_pointwise;

    const scalar_t* input = problem.input.data_ptr<scalar_t>();
    const scalar_t* weight = problem.weight.data_ptr<scalar_t>();
    const Tensor bias_contiguous = bias.defined() ? bias.contiguous() : Tensor();
    const scalar_t* bias_data = bias.defined() ? bias_contiguous.data_ptr<scalar_t>() : nullptr;
    scalar_t* out = output.mutable_data_ptr<scalar_t>();
    const Dtype dtype = problem.input.dtype();

    for_each_image_chunk(problem.batch_size, [&](int64_t begin, int64_t end) {
        Tensor buffer = is_pointwise
            ? Tensor()
            : empty({p.channels * p.kernel_height * p.kernel_width * output_plane}, problem.input.options());
        scalar_t* columns = is_pointwise ? nullptr : buffer.mutable_data_ptr<scalar_t>();

        for (int64_t n = begin; n < end; ++n) {
            const scalar_t* image = input + n * input_numel;
            if (!is_pointwise) {
                unfolded2d_copy_stub(DeviceType::CPU, dtype, columns, image, p);
            }
            const scalar_t* cols = is_pointwise ? image : columns;
            scalar_t* out_n = out + n * out_channels * output_plane;

            // The bias is the initial value of the gemm accumulator (beta = 1)
            if (bias_data) {
                for (int64_t o = 0; o < out_channels; ++o) {
                    std::fill_n(out_n + o * output_plane, output_plane, bias_data[o]);
                }
            }
            for (int64_t g = 0; g < groups; ++g) {
                cpublas::gemm(
                    TransposeType::NoTranspose, TransposeType::NoTranspose,
                    output_plane, group_out, group_rows,
                    scalar_t(1),
                    cols + g * group_rows * output_plane, output_plane,
                    weight + g * group_out * group_rows, group_rows,
                    scalar_t(bias_data ? 1 : 0),
                    out_n + g * group_out * output_plane, output_plane);
            }
        }
    });
}

// From: aten/src/ATen/native/ConvolutionMM2d.cpp (slow_conv2d_backward_out_cpu_template,
// slow_conv2d_backward_update_grad_input_frame)
//
// grad_columns[g] = weight[g]^T @ grad_output[g], then fold adds each column
// back into the image positions it came from.
template <typename scalar_t>
void conv2d_backward_input(Tensor& grad_input, const Tensor& grad_output, const Conv2dProblem& problem) {
    const Unfold2dParams& p = problem.params;
    const int64_t output_plane = problem.output_plane();
    const int64_t input_numel = problem.input_numel();
    const int64_t group_rows = problem.group_rows();
    const int64_t group_out = problem.group_out_channels();
    const int64_t out_channels = problem.out_channels;
    const int64_t groups = problem.groups;
    const bool is_pointwise = problem.is_pointwise;

    const scalar_t* grad_out = grad_output.data_ptr<scalar_t>();
    const scalar_t* weight = problem.weight.data_ptr<scalar_t>();
    scalar_t* grad_in = grad_input.mutable_data_ptr<scalar_t>();
    const Dtype dtype = problem.input.dtype();

    for_each_image_chunk(problem.batch_size, [&](int64_t begin, int64_t end) {
        Tensor buffer = is_pointwise
            ? Tensor()
            : empty({p.channels * p.kernel_height * p.kernel_width * output_plane}, problem.input.options());
        scalar_t* grad_columns = is_pointwise ? nullptr : buffer.mutable_data_ptr<scalar_t>();

        for (int64_t n = begin; n < end; ++n) {
            const scalar_t* grad_out_n = grad_out + n * out_channels * output_plane;
            scalar_t* grad_in_n = grad_in + n * input_numel;
            // For a pointwise conv the columns are the image itself
            scalar_t* cols = is_pointwise ? grad_in_n : grad_columns;

            for (int64_t g = 0; g < groups; ++g) {
                cpublas::gemm(
                    TransposeType::NoTranspose, TransposeType::Transpose,
                    output_plane, group_rows, group_out,
                    scalar_t(1),
                    grad_out_n + g * group_out * output_plane, output_plane,
                    weight + g * group_out * group_rows, group_rows,
                    scalar_t(0),
                    cols + g * group_rows * output_plane, output_plane);
            }
            if (!is_pointwise) {
                std::fill_n(grad_in_n, input_numel, scalar_t(0));
                unfolded2d_acc_stub(DeviceType::CPU, dtype, grad_columns, grad_in_n, p);
            }
        }
    });
}

// From: aten/src/ATen/native/ConvolutionMM2d.cpp (slow_conv2d_backward_weight_out_cpu_template,
// slow_conv2d_backward_weight_frame)
//
// grad_weight[g] = sum over images of grad_output[g] @ columns[g]^T.
// PyTorch loops over the images serially and relies on the threads of each
// gemm, which are small ([O, C * kh * kw] for a single image) and scale
// poorly. Here the batch is split into one chunk per thread; each chunk
// accumulates into its own copy of grad_weight (gemm with beta = 1) and the
// copies are summed at the end. The split depends only on the thread count,
// so the result is deterministic.
template <typename scalar_t>
Tensor conv2d_backward_weight(const Tensor& grad_output, const Conv2dProblem& problem) {
    const Unfold2dParams& p = problem.params;
    const int64_t output_plane = problem.output_plane();
    const int64_t input_numel = problem.input_numel();
    const int64_t group_rows = problem.group_rows();
    const int64_t group_out = problem.group_out_channels();
    const int64_t out_channels = problem.out_channels;
    const int64_t groups = problem.groups;
    const int64_t batch_size = problem.batch_size;
    const bool is_pointwise = problem.is_pointwise;
    const int64_t weight_numel = out_channels * group_rows;

    const scalar_t* input = problem.input.data_ptr<scalar_t>();
    const scalar_t* grad_out = grad_output.data_ptr<scalar_t>();
    const Dtype dtype = problem.input.dtype();

    const int64_t num_chunks = batch_size >= get_num_threads() ? get_num_threads() : 1;
    Tensor partials = zeros({num_chunks, weight_numel}, problem.input.options());
    scalar_t* partials_data = partials.mutable_data_ptr<scalar_t>();

    auto accumulate_chunks = [&](int64_t chunk_begin, int64_t chunk_end) {
        Tensor buffer = is_pointwise
            ? Tensor()
            : empty({p.channels * p.kernel_height * p.kernel_width * output_plane}, problem.input.options());
        scalar_t* columns = is_pointwise ? nullptr : buffer.mutable_data_ptr<scalar_t>();

        for (int64_t chunk = chunk_begin; chunk < chunk_end; ++chunk) {
            scalar_t* grad_weight = partials_data + chunk * weight_numel;
            const int64_t image_begin = chunk * batch_size / num_chunks;
            const int64_t image_end = (chunk + 1) * batch_size / num_chunks;

            for (int64_t n = image_begin; n < image_end; ++n) {
                const scalar_t* image = input + n * input_numel;
                if (!is_pointwise) {
                    unfolded2d_copy_stub(DeviceType::CPU, dtype, columns, image, p);
                }
                const scalar_t* cols = is_pointwise ? image : columns;
                const scalar_t* grad_out_n = grad_out + n * out_channels * output_plane;

                for (int64_t g = 0; g < groups; ++g) {
                    cpublas::gemm(
                        TransposeType::Transpose, TransposeType::NoTranspose,
                        group_rows, group_out, output_plane,
                        scalar_t(1),
                        cols + g * group_rows * output_plane, output_plane,
                        grad_out_n + g * group_out * output_plane, output_plane,
                        scalar_t(1),
                        grad_weight + g * group_out * group_rows, group_rows);
                }
            }
        }
    };

    if (num_chunks > 1) {
        parallel_for(0, num_chunks, 1, accumulate_chunks);
        return ops::sum(partials, IntArrayRef{0});
    }
    accumulate_chunks(0, 1);
    return partials.view({weight_numel});
}

} // namespace


namespace ops {

Tensor convolution(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups
) {
    const Conv2dProblem problem = make_problem(input, weight, bias, stride, padding, dilation, groups);
    const Unfold2dParams& p = problem.params;

    Tensor output = empty(
        {problem.batch_size, problem.out_channels, p.output_height, p.output_width},
        problem.input.options());
    if (output.numel() > 0) {
        problem.input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            conv2d_forward<scalar_t>(output, problem, bias);
        });
    }
    return problem.is_1d ? ops::squeeze(output, 2) : output;
}

// From: aten/src/ATen/native/Convolution.cpp (convolution_backward, Slow2d backend)
std::tuple<Tensor, Tensor, Tensor> convolution_backward(
    const Tensor& grad_output_, const Tensor& input, const Tensor& weight,
    IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups,
    std::array<bool, 3> output_mask
) {
    const Conv2dProblem problem = make_problem(input, weight, {}, stride, padding, dilation, groups);
    const Tensor grad_output = (problem.is_1d ? ops::unsqueeze(grad_output_, 2) : grad_output_).contiguous();

    Tensor grad_input, grad_weight, grad_bias;
    if (output_mask[0]) {
        grad_input = empty(problem.input.shape(), problem.input.options());
    }
    problem.input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if (output_mask[0] && grad_input.numel() > 0) {
            conv2d_backward_input<scalar_t>(grad_input, grad_output, problem);
        }
        if (output_mask[1]) {
            grad_weight = conv2d_backward_weight<scalar_t>(grad_output, problem);
        }
    });

    if (output_mask[0]) {
        grad_input = problem.is_1d ? ops::squeeze(grad_input, 2) : grad_input;
    }
    if (output_mask[1]) {
        grad_weight = ops::view(grad_weight, weight.shape());
    }
    if (output_mask[2]) {
        grad_bias = ops::sum(grad_output, IntArrayRef{0, 2, 3});
    }
    return {grad_input, grad_weight, grad_bias};
}

} // namespace ops

DEFINE_DISPATCH(unfolded2d_copy_stub);
DEFINE_DISPATCH(unfolded2d_acc_stub);

} // namespace raxis
