#pragma once

#include <cstdint>
#include <type_traits>

#include <raxis/util/host_device.h>


namespace raxis {

// From: c10/util/TypeCast.h (static_cast_with_inter_type)
// A static_cast, except from a floating point type to uint8, which goes
// through int64. A float to unsigned conversion out of range is undefined
// behaviour and the hardware disagrees: x86 wraps -2.0f to 254, ARM and
// CUDA saturate it to 0. Through int64 every device gives 254, as torch.
template <typename dest_t, typename src_t>
RAXIS_HOST_DEVICE inline dest_t cast_to(src_t value) {
    if constexpr (std::is_same_v<dest_t, uint8_t> && std::is_floating_point_v<src_t>) {
        return static_cast<uint8_t>(static_cast<int64_t>(value));
    } else {
        return static_cast<dest_t>(value);
    }
}

} // namespace raxis
