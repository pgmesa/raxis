#pragma once

#include <cuda_runtime.h>

#include <raxis/core/device.h>
#include <raxis/core/cuda_hooks.h>
#include <raxis/util/data_ptr.h>


namespace raxis::cuda {

// The CUDA allocator is registered with REGISTER_ALLOCATOR(DeviceType::CUDA)
// and reached through get_allocator(); these are its statistics and cache
// control, behind raxis.cuda.memory_allocated(), empty_cache(), ...

CUDAMemoryStats memory_stats(DeviceIndex device);
void reset_peak_memory_stats(DeviceIndex device);
/// Records that work queued on `stream` uses the memory of `data_ptr`
/// (Tensor.record_stream): when it is freed, its block waits for that work
/// before it is reused. Does nothing for memory of another allocator.
void record_stream(const DataPtr& data_ptr, cudaStream_t stream);
/// Frees the cached blocks that hold no tensor, on every device
void empty_cache();

} // namespace raxis::cuda
