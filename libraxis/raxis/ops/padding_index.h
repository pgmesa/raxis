#pragma once

#include <algorithm>
#include <cstdint>

#include <raxis/ops/padding.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/host_device.h>


// The geometry of reflect / replicate / circular padding and the input
// position each output position reads, shared by the CPU and CUDA kernels.
// From: aten/src/ATen/native/cpu/PaddingKernel.cpp
//
// Every case is seen as [planes, D, H, W] with D = H = 1 (and no padding)
// for the dims that are not padded, so the kernels have a single loop.

namespace raxis {

struct PaddingShape {
    int64_t planes;
    int64_t ishape[3];  // D, H, W
    int64_t oshape[3];
    // The size, pad and offset arguments of index() for D, H and W
    int64_t sizes[3];
    int64_t pads[3];
    int64_t offsets[3];
    // Both W pads >= 0: the input row lands unchanged in [pad_w, pad_w +
    // input_width) of the output row
    bool positive_width_pads;
};

inline PaddingShape padding_shape(const Tensor& input, const Tensor& output, IntArrayRef padding, PaddingMode mode) {
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
    RAXIS_HOST_DEVICE static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
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
    RAXIS_HOST_DEVICE static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
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
    RAXIS_HOST_DEVICE static int64_t index(int64_t j, int64_t size, int64_t pad, int64_t offset) {
        int64_t i = j - pad;
        if (i < 0) {
            i += size;
        } else if (i >= size) {
            i -= size;
        }
        return i + offset;
    }
};

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

} // namespace raxis
