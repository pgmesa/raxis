#pragma once

#include <cuda_runtime.h>

#include <raxis/core/allocator.h>
#include <raxis/util/data_ptr.h>


namespace raxis::cuda {

// Pinned (page-locked) host memory for tensor.pin_memory(). The GPU can read
// and write it with DMA while the CPU goes on, so copies from and to it can
// be asynchronous (non_blocking=True).

/// The allocator behind pin_memory(): CPU memory from cudaHostAlloc, cached
Allocator* host_allocator();

/// Records that work queued on `stream` uses the pinned block of `data_ptr`
/// (an asynchronous copy): when the tensor is freed, the block is not reused
/// until that work finished. Does nothing for memory of another allocator.
void host_allocator_record_stream(const DataPtr& data_ptr, cudaStream_t stream);

/// Returns the cached pinned blocks that hold no tensor to the driver
void host_empty_cache();

} // namespace raxis::cuda
