#pragma once

#include <raxis/core/device.h>
#include <raxis/core/cuda_hooks.h>


namespace raxis::cuda {

// The CUDA allocator is registered with REGISTER_ALLOCATOR(DeviceType::CUDA)
// and reached through get_allocator(); these are its statistics and cache
// control, behind raxis.cuda.memory_allocated(), empty_cache(), ...

CUDAMemoryStats memory_stats(DeviceIndex device);
void reset_peak_memory_stats(DeviceIndex device);
/// Frees the cached blocks that hold no tensor, on every device
void empty_cache();

} // namespace raxis::cuda
