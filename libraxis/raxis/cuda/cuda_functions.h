#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <raxis/core/device.h>
// current_stream(), which every launch reads, lives there
#include <raxis/cuda/cuda_stream.h>


namespace raxis::cuda {

/// Number of visible GPUs, 0 without a driver or a GPU (never throws, so
/// is_available() can ask it on any machine)
int device_count() noexcept;

DeviceIndex current_device();
void set_device(DeviceIndex index);

/// Waits for all the work queued on the current device
void device_synchronize();

/// Properties of the current device (SM count, threads per SM, ...), queried
/// once per device. From: aten/src/ATen/cuda/CUDAContext.h
/// (getCurrentDeviceProperties)
const cudaDeviceProp& current_device_properties();

/// cudaMemcpyAsync on `stream`, then waits for it: a device to host copy
/// must be complete before the CPU reads the data
/// From: c10/cuda/CUDAFunctions.h
void memcpy_and_sync(void* dst, const void* src, size_t nbytes, cudaMemcpyKind kind, cudaStream_t stream);

} // namespace raxis::cuda
