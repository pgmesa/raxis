#include <raxis/ops/padding.h>

#include <algorithm>
#include <cstdint>

#include <raxis/ops/copy.h>
#include <raxis/ops/fill.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_shape.h>


namespace raxis {

namespace {

const char* padding_mode_name(PaddingMode mode) {
    switch (mode) {
        case PaddingMode::Constant: return "constant";
        case PaddingMode::Reflect: return "reflect";
        case PaddingMode::Replicate: return "replicate";
        case PaddingMode::Circular: return "circular";
    }
    return "unknown";
}

// From: aten/src/ATen/native/Padding.h (check_valid_input) and the meta
// functions of ReflectionPad.cpp / ReplicationPadding.cpp
DimVector padding_output_shape(const Tensor& input, IntArrayRef pad, PaddingMode mode) {
    const int64_t dim = static_cast<int64_t>(pad.size()) / 2;
    RAXIS_CHECK(pad.size() % 2 == 0 && dim >= 1 && dim <= 3,
        "padding size is expected to be 2, 4 or 6, but got: ", pad.size());
    const int64_t input_dim = input.ndim();
    const bool batch_mode = input_dim == dim + 2;
    bool valid = batch_mode || input_dim == dim + 1;
    // The batch may be empty, the other dims may not
    for (int64_t d = batch_mode ? 1 : 0; valid && d < input_dim; ++d) {
        valid = input.shape()[d] != 0;
    }
    RAXIS_CHECK(valid,
        "Expected ", dim + 1, "D or ", dim + 2,
        "D (batch mode) tensor with possibly 0 batch size and other non-zero dimensions for input, but got: ",
        input.shape());

    DimVector output_shape(input.shape().begin(), input.shape().end());
    for (int64_t i = 0; i < dim; ++i) {
        const int64_t d = input_dim - 1 - i;
        const int64_t size = input.shape()[d];
        const int64_t pad_l = pad[2 * i];
        const int64_t pad_r = pad[2 * i + 1];
        output_shape[d] = size + pad_l + pad_r;
        if (mode == PaddingMode::Reflect) {
            RAXIS_CHECK(pad_l < size && pad_r < size,
                "Padding size should be less than the corresponding input dimension, but got: padding (",
                pad_l, ", ", pad_r, ") at dimension ", d, " of input ", input.shape());
        } else if (mode == PaddingMode::Circular) {
            // From: aten/src/ATen/native/PadNd.cpp (_pad_circular_symint).
            // The pads wrap around what is left after cropping: PyTorch
            // only checks them against the whole size and then copies
            // positions outside the kept range
            const int64_t kept = size - std::max<int64_t>(0, -pad_l) - std::max<int64_t>(0, -pad_r);
            RAXIS_CHECK(pad_l <= kept && pad_r <= kept, "Padding value causes wrapping around more than once.");
            RAXIS_CHECK(output_shape[d] >= 0, "Negative padding value is resulting in an empty dimension");
            continue;
        }
        RAXIS_CHECK(output_shape[d] >= 1,
            "input (", padding_mode_name(mode), " padding) is too small: the padded size of dimension ", d,
            " would be ", output_shape[d], " (input size ", size, ", padding (", pad_l, ", ", pad_r, "))");
    }
    return output_shape;
}

// t.narrow(dim, start, length) without autograd, with narrow's check: a
// crop larger than the dim is an error, not an empty slice
Tensor narrow_view(const Tensor& t, int64_t dim, int64_t start, int64_t length) {
    RAXIS_CHECK(length >= 0, "narrow(): length must be non-negative.");
    return ops::slice(t, dim, start, start + length);
}

} // namespace


namespace ops {

// The padded output is filled with value and the input (cropped where pad
// is negative) copied into the interior view of it
Tensor constant_pad_nd(const Tensor& self, IntArrayRef pad, Scalar value) {
    RAXIS_CHECK(pad.size() % 2 == 0, "Length of pad must be even but instead it equals ", pad.size());
    const int64_t l_inp = self.ndim();
    const int64_t l_pad = static_cast<int64_t>(pad.size()) / 2;
    const int64_t l_diff = l_inp - l_pad;
    RAXIS_CHECK(l_inp >= l_pad,
        "Length of pad should be no more than twice the number of dimensions of the input. "
        "Pad length is ", pad.size(), " while the input has ", l_inp, " dimensions.");

    bool all_pads_non_positive = true;
    Tensor c_input = self;
    for (int64_t i = l_diff; i < l_inp; ++i) {
        const int64_t pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] < 0) {
            c_input = narrow_view(c_input, i, -pad[pad_idx], c_input.shape()[i] + pad[pad_idx]);
        } else if (pad[pad_idx] != 0) {
            all_pads_non_positive = false;
        }
        if (pad[pad_idx + 1] < 0) {
            c_input = narrow_view(c_input, i, 0, c_input.shape()[i] + pad[pad_idx + 1]);
        } else if (pad[pad_idx + 1] != 0) {
            all_pads_non_positive = false;
        }
    }
    // Only cropping: no new positions to fill
    if (all_pads_non_positive) {
        return ops::clone(c_input);
    }

    DimVector new_shape(self.shape().begin(), self.shape().end());
    for (int64_t i = 0; i < l_pad; ++i) {
        const int64_t pad_idx = static_cast<int64_t>(pad.size()) - (i + 1) * 2;
        const int64_t new_dim = self.shape()[l_diff + i] + pad[pad_idx] + pad[pad_idx + 1];
        RAXIS_CHECK(new_dim >= 0,
            "The input size ", self.shape()[l_diff + i], ", plus negative padding ", pad[pad_idx], " and ",
            pad[pad_idx + 1], " resulted in a negative output size, which is invalid. Check dimension ",
            l_diff + i, " of your input.");
        new_shape[l_diff + i] = new_dim;
    }

    Tensor output = empty(new_shape, self.options());
    ops::fill_(output, value);
    Tensor c_output = output;
    for (int64_t i = l_diff; i < l_inp; ++i) {
        const int64_t pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] > 0) {
            c_output = narrow_view(c_output, i, pad[pad_idx], c_output.shape()[i] - pad[pad_idx]);
        }
        if (pad[pad_idx + 1] > 0) {
            c_output = narrow_view(c_output, i, 0, c_output.shape()[i] - pad[pad_idx + 1]);
        }
    }
    copy(c_input, c_output);
    return output;
}

Tensor padding(const Tensor& self, IntArrayRef pad, PaddingMode mode) {
    RAXIS_CHECK(mode != PaddingMode::Constant, "padding: use constant_pad_nd for constant padding");
    const DimVector output_shape = padding_output_shape(self, pad, mode);
    Tensor output = empty(output_shape, self.options());
    if (output.numel() > 0) {
        padding_stub(DeviceType::CPU, output, self.contiguous(), pad, mode);
    }
    return output;
}

Tensor padding_backward(
    const Tensor& grad_output, IntArrayRef input_shape, IntArrayRef pad, PaddingMode mode
) {
    Tensor grad_input = zeros(input_shape, grad_output.options());
    if (grad_output.numel() > 0) {
        padding_backward_stub(DeviceType::CPU, grad_input, grad_output.contiguous(), pad, mode);
    }
    return grad_input;
}

} // namespace ops

DEFINE_DISPATCH(padding_stub);
DEFINE_DISPATCH(padding_backward_stub);

} // namespace raxis
