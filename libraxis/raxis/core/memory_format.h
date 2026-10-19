#pragma once

#include <cstdint>


namespace raxis {

// From: c10/core/MemoryFormat.h
// Layout requested for a new tensor created from another one (clone,
// empty_like, ...):
//   - Contiguous: row-major strides, whatever the layout of the source.
//   - Preserve: same strides as the source when it is dense (e.g. a
//     transposed tensor stays transposed), else dense strides that keep its
//     dimension order. This is what PyTorch does by default.
// PyTorch also has ChannelsLast and ChannelsLast3d (NHWC layouts for
// convolutions).
enum class MemoryFormat : int8_t {
    Contiguous,
    Preserve,
};

} // namespace raxis
