#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include <raxis/core/device.h>


namespace raxis::cuda {

/// Number of visible GPUs, 0 without a driver or a GPU (never throws, so
/// is_available() can ask it on any machine)
int device_count() noexcept;

DeviceIndex current_device();
void set_device(DeviceIndex index);

/// Waits for all the work queued on the current device
void device_synchronize();

/// The stream every op enqueues its work on.
///
/// raxis has a single stream per device for now: the legacy default stream,
/// which runs everything on the device in launch order. Two consequences: the
/// host never waits for a kernel unless it reads the result (copy to the CPU,
/// item(), synchronize()), and the caching allocator can hand a freed block
/// to the next op at once, because that op runs after every op that used the
/// block. PyTorch also starts on the default stream; user streams
/// (torch.cuda.Stream) are what make record_stream necessary.
/// From: c10/cuda/CUDAStream.h (getCurrentCUDAStream)
inline cudaStream_t current_stream() {
    return cudaStreamLegacy;
}

/// cudaMemcpyAsync on `stream`, then waits for it: a device to host copy
/// must be complete before the CPU reads the data
/// From: c10/cuda/CUDAFunctions.h
void memcpy_and_sync(void* dst, const void* src, size_t nbytes, cudaMemcpyKind kind, cudaStream_t stream);

} // namespace raxis::cuda
