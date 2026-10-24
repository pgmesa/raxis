#include <raxis/ops/padding.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/util/parallel.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

// From: aten/src/ATen/native/cpu/PaddingKernel.cpp
//
// Reflect, replicate and circular padding only differ in which input
// position an output position reads, so one kernel serves all three through
// index(). PyTorch has separate loops for 1, 2 and 3 padded dims (and builds
// circular padding from slice copies in PadNd.cpp); here every case is
// [planes, D, H, W] with D = H = 1 (and no padding) for the missing dims,
// so there is a single loop: one parallel task per output row (plane, od,
// oh), and inside a row only the left and right borders need index(); the
// interior is a straight vectorized copy of the input row.

struct PaddingShape {
    int64_t planes;
    std::array<int64_t, 3> ishape;  // D, H, W
    std::array<int64_t, 3> oshape;
    // The size, pad and offset arguments of index() for D, H and W
    std::array<int64_t, 3> sizes;
    std::array<int64_t, 3> pads;
    std::array<int64_t, 3> offsets;
    // Both W pads >= 0: the input row lands unchanged in [pad_w, pad_w +
    // input_width) of the output row
    bool positive_width_pads;
};

PaddingShape padding_shape(const Tensor& input, const Tensor& output, IntArrayRef padding, PaddingMode mode) {
    const int64_t ndim = static_cast<int64_t>(padding.size()) / 2;
    PaddingShape p;
    p.planes = 1;
    for (int64_t d = 0; d < input.ndim() - ndim; ++d) {
        p.planes *= input.shape()[d];
    }
    for (int64_t k = 0; k < 3; ++k) {
        // k = 2 is W, padded by padding[0], padding[1]; k = 1 is H, ...
        const int64_t i = 2 - k;
        const bool padded = i < ndim;
        const int64_t d = input.ndim() - 1 - i;
        const int64_t size = padded ? input.shape()[d] : 1;
        const int64_t pad_l = padded ? padding[2 * i] : 0;
        const int64_t pad_r = padded ? padding[2 * i + 1] : 0;
        p.ishape[k] = size;
        p.oshape[k] = padded ? output.shape()[d] : 1;
        if (mode == PaddingMode::Circular) {
            // A negative pad crops first; the positive ones then wrap around
            // the kept range [crop_l, crop_l + kept) of the input
            const int64_t crop_l = std::max<int64_t>(0, -pad_l);
            p.sizes[k] = size - crop_l - std::max<int64_t>(0, -pad_r);
            p.pads[k] = std::max<int64_t>(0, pad_l);
            p.offsets[k] = crop_l;
        } else {
            // index() works in output coordinates shifted by the left pad;
            // the offset maps them back to the input (and skips the cropped
            // part when the pad is negative)
            p.sizes[k] = size;
            p.pads[k] = pad_l;
            p.offsets[k] = std::max<int64_t>(0, -pad_l) - std::max<int64_t>(0, pad_l);
        }
    }
    p.positive_width_pads = padding[0] >= 0 && padding[1] >= 0;
    return p;
}

// j: output position. Reflect and replicate get the input size and the left
// pad; circular gets the kept size, the left pad (>= 0) and the crop.
struct ReflectionPad {
    static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
        int64_t i;
        if (j < pad) {
            i = pad * 2 - j;
        } else if (j < size + pad) {
            i = j;
        } else {
            i = (size + pad - 1) * 2 - j;
        }
        return i + offset;
    }
};

struct ReplicationPad {
    static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
        int64_t i;
        if (j < pad) {
            i = pad;
        } else if (j < size + pad) {
            i = j;
        } else {
            i = size + pad - 1;
        }
        return i + offset;
    }
};

// The pads are at most the kept size, so one wrap is enough
struct CircularPad {
    static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
        int64_t i = j - pad;
        if (i < 0) {
            i += size;
        } else if (i >= size) {
            i -= size;
        }
        return i + offset;
    }
};

template <typename scalar_t>
inline void copy_row(scalar_t* out, const scalar_t* in, int64_t size) {
    using Vec = Vectorized<scalar_t>;
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec::loadu(in + d).store(out + d);
    }
    for (; d < size; d++) {
        out[d] = in[d];
    }
}

template <typename scalar_t, typename PaddingType>
void cpu_padding(const Tensor& output, const Tensor& input, const PaddingShape& p) {
    const scalar_t* input_data = input.data_ptr<scalar_t>();
    scalar_t* output_data = output.mutable_data_ptr<scalar_t>();
    const auto [input_depth, input_height, input_width] = p.ishape;
    const auto [output_depth, output_height, output_width] = p.oshape;
    const auto [size_d, size_h, size_w] = p.sizes;
    const auto [pad_d, pad_h, pad_w] = p.pads;
    const auto [offset_d, offset_h, offset_w] = p.offsets;

    const int64_t rows = p.planes * output_depth * output_height;
    const int64_t grain_size = std::max<int64_t>(1, detail::GRAIN_SIZE / output_width);
    parallel_for(0, rows, grain_size, [&](int64_t begin, int64_t end) {
        // (c, od, oh) of the first row, then stepped like an odometer
        int64_t oh = begin % output_height;
        int64_t od = (begin / output_height) % output_depth;
        int64_t c = begin / (output_height * output_depth);
        for (int64_t row = begin; row < end; ++row) {
            const int64_t id = PaddingType::index(od, size_d, pad_d, offset_d);
            const int64_t ih = PaddingType::index(oh, size_h, pad_h, offset_h);
            scalar_t* out = output_data + row * output_width;
            const scalar_t* in = input_data + ((c * input_depth + id) * input_height + ih) * input_width;
            if (p.positive_width_pads) {
                for (int64_t ow = 0; ow < pad_w; ++ow) {
                    out[ow] = in[PaddingType::index(ow, size_w, pad_w, offset_w)];
                }
                copy_row(out + pad_w, in, input_width);
                for (int64_t ow = input_width + pad_w; ow < output_width; ++ow) {
                    out[ow] = in[PaddingType::index(ow, size_w, pad_w, offset_w)];
                }
            } else {
                for (int64_t ow = 0; ow < output_width; ++ow) {
                    out[ow] = in[PaddingType::index(ow, size_w, pad_w, offset_w)];
                }
            }
            if (++oh == output_height) {
                oh = 0;
                if (++od == output_depth) {
                    od = 0;
                    ++c;
                }
            }
        }
    });
}

// Several output positions read the same input position (the edge with
// replicate, the mirrored or wrapped ones otherwise), so their gradients
// collide in grad_input. Each task owns whole planes and walks them
// serially, so no two threads add into the same element.
template <typename scalar_t, typename PaddingType>
void cpu_padding_backward(const Tensor& grad_input, const Tensor& grad_output, const PaddingShape& p) {
    const scalar_t* grad_output_data = grad_output.data_ptr<scalar_t>();
    scalar_t* grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
    const auto [input_depth, input_height, input_width] = p.ishape;
    const auto [output_depth, output_height, output_width] = p.oshape;
    const auto [size_d, size_h, size_w] = p.sizes;
    const auto [pad_d, pad_h, pad_w] = p.pads;
    const auto [offset_d, offset_h, offset_w] = p.offsets;
    const int64_t input_plane = input_depth * input_height * input_width;
    const int64_t output_plane = output_depth * output_height * output_width;

    parallel_for(0, p.planes, 1, [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            const scalar_t* grad_out = grad_output_data + c * output_plane;
            scalar_t* grad_in = grad_input_data + c * input_plane;
            for (int64_t od = 0; od < output_depth; ++od) {
                const int64_t id = PaddingType::index(od, size_d, pad_d, offset_d);
                for (int64_t oh = 0; oh < output_height; ++oh) {
                    const int64_t ih = PaddingType::index(oh, size_h, pad_h, offset_h);
                    scalar_t* grad_in_row = grad_in + (id * input_height + ih) * input_width;
                    const scalar_t* grad_out_row = grad_out + (od * output_height + oh) * output_width;
                    for (int64_t ow = 0; ow < output_width; ++ow) {
                        grad_in_row[PaddingType::index(ow, size_w, pad_w, offset_w)] += grad_out_row[ow];
                    }
                }
            }
        }
    });
}

template <typename Func>
void dispatch_padding_type(PaddingMode mode, Func&& f) {
    if (mode == PaddingMode::Reflect) {
        f(ReflectionPad{});
    } else if (mode == PaddingMode::Replicate) {
        f(ReplicationPad{});
    } else {
        f(CircularPad{});
    }
}

void padding_kernel(const Tensor& output, const Tensor& input, IntArrayRef padding, PaddingMode mode) {
    const PaddingShape p = padding_shape(input, output, padding, mode);
    input.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        dispatch_padding_type(mode, [&]<typename PaddingType>(PaddingType) {
            cpu_padding<scalar_t, PaddingType>(output, input, p);
        });
    });
}

void padding_backward_kernel(
    const Tensor& grad_input, const Tensor& grad_output, IntArrayRef padding, PaddingMode mode
) {
    const PaddingShape p = padding_shape(grad_input, grad_output, padding, mode);
    grad_output.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        dispatch_padding_type(mode, [&]<typename PaddingType>(PaddingType) {
            cpu_padding_backward<scalar_t, PaddingType>(grad_input, grad_output, p);
        });
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(padding_stub, &padding_kernel);
REGISTER_DISPATCH(padding_backward_stub, &padding_backward_kernel);

} // namespace raxis
