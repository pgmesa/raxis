#pragma once

#include <cuda_runtime.h>

#include <raxis/core/device.h>


namespace raxis::cuda {

// Streams, as PyTorch's CUDAStream.
// From: c10/cuda/CUDAStream.h
//
// Every op queues its work on the current stream of its device. A thread
// starts on the default stream (the legacy one, which runs everything on the
// device in launch order) and can switch to a stream of its own, so that,
// for example, a copy runs while the default stream computes. The current
// stream is per thread and per device, as in PyTorch.
//
// The streams a user gets come from a pool: 32 per device and priority,
// created on first use, handed out round robin and never destroyed. So a
// stream handle stays valid for the life of the process, and blocks the
// caching allocator keeps for a stream can never be reused by an unrelated
// stream that happens to get the same handle.

/// The default stream of every device
inline cudaStream_t default_stream() {
    return cudaStreamLegacy;
}

/// The stream the ops of this thread queue their work on, on the current
/// device
cudaStream_t current_stream();
cudaStream_t current_stream(DeviceIndex device);

/// Makes `stream` (of `device`) the current stream of this thread on `device`
void set_current_stream(DeviceIndex device, cudaStream_t stream);

/// A stream of the pool of `device`. Created with cudaStreamNonBlocking, as
/// PyTorch's: it does not wait for the default stream, nor the default
/// stream for it, so the two can overlap. `priority` is 0 (normal) or
/// negative (higher), clamped to the levels the device has, at most 4
/// From: c10/cuda/CUDAStream.cpp (getStreamFromPool)
cudaStream_t stream_from_pool(DeviceIndex device, int priority);

} // namespace raxis::cuda
