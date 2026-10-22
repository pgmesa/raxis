#include <raxis/ops/convolution.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/vec/functional_base.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/util/parallel.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

// From: aten/src/ATen/native/cpu/Unfold2d.cpp
//
// PyTorch's version checks the bounds per output row (and per element when
// the stride is not 1) and has no dilation, which goes through a second
// im2col (aten/src/ATen/native/im2col.h). Here the bounds are solved once
// per kernel tap: the output positions whose input falls inside the image
// form one contiguous range, so each row is zeros + one copy + zeros, and
// dilation only shifts the offset of the tap.

struct OutputRange {
    int64_t begin;
    int64_t end;
};

// Output positions o in [0, output_size) such that the input position
// o * stride + offset is inside [0, input_size)
inline OutputRange valid_output_range(
    int64_t offset, int64_t stride, int64_t input_size, int64_t output_size
) {
    // o * stride + offset >= 0          <=>  o >= ceil(-offset / stride)
    // o * stride + offset < input_size  <=>  o <  ceil((input_size - offset) / stride)
    int64_t begin = offset >= 0 ? 0 : (-offset + stride - 1) / stride;
    int64_t end = input_size - offset <= 0 ? 0 : (input_size - offset + stride - 1) / stride;
    begin = std::min(begin, output_size);
    end = std::clamp(end, begin, output_size);
    return {begin, end};
}

template <typename scalar_t>
void unfolded2d_copy(scalar_t* columns, const scalar_t* input, const Unfold2dParams& p) {
    const int64_t taps = p.kernel_height * p.kernel_width;
    const int64_t input_plane = p.input_height * p.input_width;
    const int64_t output_plane = p.output_height * p.output_width;
    const int64_t output_width = p.output_width;
    const int64_t stride_width = p.stride_width;

    // One task per row of columns: a (channel, kh, kw) tap
    parallel_for(0, p.channels * taps, 0, [&](int64_t begin, int64_t end) {
        for (int64_t k = begin; k < end; ++k) {
            const int64_t c = k / taps;
            const int64_t kh = (k % taps) / p.kernel_width;
            const int64_t kw = k % p.kernel_width;
            const int64_t h_offset = kh * p.dilation_height - p.pad_height;
            const int64_t w_offset = kw * p.dilation_width - p.pad_width;
            const auto [y0, y1] = valid_output_range(
                h_offset, p.stride_height, p.input_height, p.output_height);
            const auto [x0, x1] = valid_output_range(
                w_offset, stride_width, p.input_width, output_width);

            scalar_t* dst = columns + k * output_plane;
            const scalar_t* src = input + c * input_plane;

            std::fill(dst, dst + y0 * output_width, scalar_t(0));
            for (int64_t y = y0; y < y1; ++y) {
                scalar_t* dst_row = dst + y * output_width;
                const scalar_t* src_row = src + (y * p.stride_height + h_offset) * p.input_width;
                std::fill(dst_row, dst_row + x0, scalar_t(0));
                if (stride_width == 1) {
                    std::copy(src_row + x0 + w_offset, src_row + x1 + w_offset, dst_row + x0);
                } else {
                    for (int64_t x = x0; x < x1; ++x) {
                        dst_row[x] = src_row[x * stride_width + w_offset];
                    }
                }
                std::fill(dst_row + x1, dst_row + output_width, scalar_t(0));
            }
            std::fill(dst + y1 * output_width, dst + output_plane, scalar_t(0));
        }
    });
}

template <typename scalar_t>
void unfolded2d_acc(const scalar_t* columns, scalar_t* input, const Unfold2dParams& p) {
    using Vec = Vectorized<scalar_t>;
    const int64_t input_plane = p.input_height * p.input_width;
    const int64_t output_plane = p.output_height * p.output_width;
    const int64_t output_width = p.output_width;
    const int64_t stride_width = p.stride_width;

    // All taps of a channel add into the same image plane, so the work is
    // split by channel (unfold splits by tap)
    parallel_for(0, p.channels, 0, [&](int64_t begin, int64_t end) {
        for (int64_t c = begin; c < end; ++c) {
            scalar_t* dst = input + c * input_plane;
            for (int64_t kh = 0; kh < p.kernel_height; ++kh) {
                for (int64_t kw = 0; kw < p.kernel_width; ++kw) {
                    const int64_t h_offset = kh * p.dilation_height - p.pad_height;
                    const int64_t w_offset = kw * p.dilation_width - p.pad_width;
                    const auto [y0, y1] = valid_output_range(
                        h_offset, p.stride_height, p.input_height, p.output_height);
                    const auto [x0, x1] = valid_output_range(
                        w_offset, stride_width, p.input_width, output_width);
                    const scalar_t* src =
                        columns + ((c * p.kernel_height + kh) * p.kernel_width + kw) * output_plane;

                    for (int64_t y = y0; y < y1; ++y) {
                        scalar_t* dst_row = dst + (y * p.stride_height + h_offset) * p.input_width;
                        const scalar_t* src_row = src + y * output_width;
                        if (stride_width == 1) {
                            scalar_t* out = dst_row + x0 + w_offset;
                            map2([](Vec a, Vec b) { return a + b; }, out, out, src_row + x0, x1 - x0);
                        } else {
                            for (int64_t x = x0; x < x1; ++x) {
                                dst_row[x * stride_width + w_offset] += src_row[x];
                            }
                        }
                    }
                }
            }
        }
    });
}

void unfolded2d_copy_kernel(Dtype dtype, void* columns, const void* input, const Unfold2dParams& params) {
    dtype.dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        unfolded2d_copy(static_cast<scalar_t*>(columns), static_cast<const scalar_t*>(input), params);
    });
}

void unfolded2d_acc_kernel(Dtype dtype, const void* columns, void* input, const Unfold2dParams& params) {
    dtype.dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        unfolded2d_acc(static_cast<const scalar_t*>(columns), static_cast<scalar_t*>(input), params);
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(unfolded2d_copy_stub, &unfolded2d_copy_kernel);
REGISTER_DISPATCH(unfolded2d_acc_stub, &unfolded2d_acc_kernel);

} // namespace raxis
