#include <raxis/ops/pooling.h>

#include <cstdint>

#include <raxis/util/exception.h>

#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

namespace {

// Division rounding toward negative infinity: with a kernel larger than the
// padded input the numerator is negative and the output size must come out
// <= 0 (and be rejected), not round up to 1.
// From: aten/src/ATen/div_rtn.h
int64_t div_rtn(int64_t x, int64_t y) {
    int64_t q = x / y;
    if (x % y != 0 && ((x < 0) != (y < 0))) {
        --q;
    }
    return q;
}

// From: aten/src/ATen/native/Pool.h (pooling_output_shape)
int64_t pooling_output_shape(
    int64_t input_size, int64_t kernel_size, int64_t pad, int64_t stride, int64_t dilation, bool ceil_mode
) {
    RAXIS_CHECK(stride != 0, "stride should not be zero");
    RAXIS_CHECK(pad >= 0, "pad must be non-negative, but got pad: ", pad);
    RAXIS_CHECK(pad <= ((kernel_size - 1) * dilation + 1) / 2,
        "pad should be at most half of effective kernel size, but got pad=",
        pad, ", kernel_size=", kernel_size, " and dilation=", dilation);
    int64_t output_size = div_rtn(
        input_size + 2 * pad - dilation * (kernel_size - 1) - 1 + (ceil_mode ? stride - 1 : 0),
        stride) + 1;
    // With ceil_mode the last window must start inside the image or the
    // left padding, not in the right padding
    if (ceil_mode && (output_size - 1) * stride >= input_size + pad) {
        --output_size;
    }
    return output_size;
}

// Reads a parameter given as one int for both dims or as (height, width)
struct Pair {
    int64_t height;
    int64_t width;
};

Pair read_pair(IntArrayRef values, Pair if_empty) {
    if (values.empty()) {
        return if_empty;
    }
    return {values[0], values.size() == 1 ? values[0] : values[1]};
}

// From: aten/src/ATen/native/DilatedMaxPool2d.cpp and AveragePool2d.cpp
// (TORCH_META_FUNC), aten/src/ATen/native/Pool.h (pool2d_shape_check)
Pool2dParams make_pool2d_params(
    const char* name, IntArrayRef input_shape, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode
) {
    RAXIS_CHECK(kernel_size.size() == 1 || kernel_size.size() == 2,
        name, ": kernel_size must either be a single int, or a tuple of two ints");
    RAXIS_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 2,
        name, ": stride must either be omitted, a single int, or a tuple of two ints");
    RAXIS_CHECK(padding.size() == 1 || padding.size() == 2,
        name, ": padding must either be a single int, or a tuple of two ints");
    RAXIS_CHECK(dilation.size() == 1 || dilation.size() == 2,
        name, ": dilation must be either a single int, or a tuple of two ints");
    const int64_t ndim = static_cast<int64_t>(input_shape.size());
    RAXIS_CHECK(ndim == 3 || ndim == 4, "non-empty 3D or 4D (batch mode) tensor expected for input");

    const Pair kernel = read_pair(kernel_size, {});
    const Pair step = read_pair(stride, kernel);
    const Pair pad = read_pair(padding, {});
    const Pair dil = read_pair(dilation, {});

    Pool2dParams p;
    p.input_height = input_shape[ndim - 2];
    p.input_width = input_shape[ndim - 1];
    p.kernel_height = kernel.height;
    p.kernel_width = kernel.width;
    p.stride_height = step.height;
    p.stride_width = step.width;
    p.pad_height = pad.height;
    p.pad_width = pad.width;
    p.dilation_height = dil.height;
    p.dilation_width = dil.width;
    p.output_height = pooling_output_shape(
        p.input_height, p.kernel_height, p.pad_height, p.stride_height, p.dilation_height, ceil_mode);
    p.output_width = pooling_output_shape(
        p.input_width, p.kernel_width, p.pad_width, p.stride_width, p.dilation_width, ceil_mode);

    RAXIS_CHECK(p.kernel_width > 0 && p.kernel_height > 0,
        "kernel size should be greater than zero, but got kH: ", p.kernel_height, " kW: ", p.kernel_width);
    RAXIS_CHECK(p.stride_width > 0 && p.stride_height > 0,
        "stride should be greater than zero, but got dH: ", p.stride_height, " dW: ", p.stride_width);
    RAXIS_CHECK(p.dilation_height > 0 && p.dilation_width > 0,
        "dilation should be greater than zero, but got dilationH: ", p.dilation_height,
        " dilationW: ", p.dilation_width);
    // Only the batch dim can be 0
    const int64_t channels = input_shape[ndim - 3];
    RAXIS_CHECK(channels != 0 && p.input_height != 0 && p.input_width != 0,
        "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input, but got:",
        input_shape);
    RAXIS_CHECK(p.kernel_width / 2 >= p.pad_width && p.kernel_height / 2 >= p.pad_height,
        "pad should be smaller than or equal to half of kernel size, but got ",
        "padW = ", p.pad_width, ", padH = ", p.pad_height,
        ", kW = ", p.kernel_width, ", kH = ", p.kernel_height);
    RAXIS_CHECK(p.output_width >= 1 && p.output_height >= 1,
        "Given input size: (", channels, "x", p.input_height, "x", p.input_width, "). ",
        "Calculated output size: (", channels, "x", p.output_height, "x", p.output_width, "). ",
        "Output size is too small");
    return p;
}

DimVector pooled_shape(IntArrayRef input_shape, const Pool2dParams& p) {
    DimVector shape(input_shape.begin(), input_shape.end());
    shape[shape.size() - 2] = p.output_height;
    shape[shape.size() - 1] = p.output_width;
    return shape;
}

void check_pool_dtype(const char* name, const Tensor& input) {
    RAXIS_CHECK(input.dtype().is_floating(),
        name, ": expected a floating point input, but got ", input.dtype());
}

} // namespace


namespace ops {

std::tuple<Tensor, Tensor> max_pool2d_with_indices(
    const Tensor& input_, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode
) {
    const Pool2dParams p = make_pool2d_params(
        "max_pool2d", input_.shape(), kernel_size, stride, padding, dilation, ceil_mode);
    RAXIS_CHECK(input_.dtype() != dtypes::Bool, "max_pool2d: bool input is not supported");

    const Tensor input = input_.contiguous();
    const DimVector shape = pooled_shape(input.shape(), p);
    Tensor output = empty(shape, input.options());
    Tensor indices = empty(shape, input.options().dtype(dtypes::Int64));
    if (output.numel() > 0) {
        max_pool2d_stub(input.device().type(), output, indices, input, p);
    }
    return {output, indices};
}

Tensor max_pool2d_with_indices_backward(
    const Tensor& grad_output, IntArrayRef input_shape, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor& indices
) {
    const Pool2dParams p = make_pool2d_params(
        "max_pool2d", input_shape, kernel_size, stride, padding, dilation, ceil_mode);
    Tensor grad_input = empty(input_shape, grad_output.options());
    if (grad_input.numel() > 0) {
        max_pool2d_backward_stub(
            grad_output.device().type(), grad_input, grad_output.contiguous(), indices.contiguous(), p);
    }
    return grad_input;
}

Tensor avg_pool2d(
    const Tensor& input_, IntArrayRef kernel_size, IntArrayRef stride,
    IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override
) {
    RAXIS_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");
    const Pool2dParams p = make_pool2d_params(
        "avg_pool2d", input_.shape(), kernel_size, stride, padding, {1}, ceil_mode);
    check_pool_dtype("avg_pool2d", input_);

    const Tensor input = input_.contiguous();
    Tensor output = empty(pooled_shape(input.shape(), p), input.options());
    if (output.numel() > 0) {
        avg_pool2d_stub(input.device().type(), output, input, p, count_include_pad, divisor_override);
    }
    return output;
}

Tensor avg_pool2d_backward(
    const Tensor& grad_output, IntArrayRef input_shape, IntArrayRef kernel_size,
    IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad,
    std::optional<int64_t> divisor_override
) {
    const Pool2dParams p = make_pool2d_params(
        "avg_pool2d", input_shape, kernel_size, stride, padding, {1}, ceil_mode);
    Tensor grad_input = empty(input_shape, grad_output.options());
    if (grad_input.numel() > 0) {
        avg_pool2d_backward_stub(
            grad_output.device().type(), grad_input, grad_output.contiguous(), p, count_include_pad, divisor_override);
    }
    return grad_input;
}

} // namespace ops

DEFINE_DISPATCH(max_pool2d_stub);
DEFINE_DISPATCH(max_pool2d_backward_stub);
DEFINE_DISPATCH(avg_pool2d_stub);
DEFINE_DISPATCH(avg_pool2d_backward_stub);

} // namespace raxis
