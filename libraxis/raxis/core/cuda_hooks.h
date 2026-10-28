#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include <raxis/export.h>
#include <raxis/core/allocator.h>
#include <raxis/core/device.h>


namespace raxis {

// From: torch.cuda.get_device_properties (a subset)
struct CUDADeviceProperties {
    std::string name;
    int major = 0;
    int minor = 0;
    int64_t total_memory = 0;
    int multi_processor_count = 0;
    // Largest dynamic shared memory a block can opt into (164 KB on an
    // A100, 99 KB on RTX 30xx / 40xx, 64 KB on a T4)
    int64_t shared_memory_per_block_optin = 0;
};

struct CUDAMemoryStats {
    int64_t allocated_bytes = 0;
    int64_t max_allocated_bytes = 0;
    int64_t reserved_bytes = 0;
    int64_t max_reserved_bytes = 0;
};

/// What the CPU library needs from CUDA, implemented by raxis_cuda.
///
/// libraxis does not link CUDA: the CUDA kernels, the allocator and the
/// runtime calls live in a separate library that registers them when it is
/// loaded (kernels into the DispatchStubs, the allocator with
/// REGISTER_ALLOCATOR, the rest here). Without it this default
/// implementation answers "no CUDA", so the CPU wheel stays as it is.
///
/// PyTorch splits this in two: CUDAHooksInterface in ATen (is_available,
/// device_count, ...) and c10's DeviceGuardImplInterface (get/set the current
/// device). raxis has one interface for both.
/// From: aten/src/ATen/detail/CUDAHooksInterface.h
struct RAXIS_API CUDAHooksInterface {
    virtual ~CUDAHooksInterface() = default;

    virtual bool is_available() const;
    virtual int device_count() const;
    virtual DeviceIndex current_device() const;
    virtual void set_device(DeviceIndex index) const;
    /// (major, minor) compute capability, 8.0 for an A100
    virtual std::pair<int, int> device_capability(DeviceIndex index) const;
    virtual CUDADeviceProperties device_properties(DeviceIndex index) const;
    /// Waits for all the work queued on the device
    virtual void synchronize(DeviceIndex index) const;

    virtual CUDAMemoryStats memory_stats(DeviceIndex index) const;
    virtual void reset_peak_memory_stats(DeviceIndex index) const;
    /// Returns the cached, unused blocks of every device to the driver
    virtual void empty_cache() const;

    /// The allocator of pinned (page-locked) CPU memory, for pin_memory()
    virtual Allocator* pinned_memory_allocator() const;
    /// Whether a CPU pointer is in page-locked memory (false without CUDA)
    virtual bool is_pinned(const void* data) const;
    /// Returns the cached, unused pinned blocks to the driver
    virtual void host_empty_cache() const;
};

RAXIS_API const CUDAHooksInterface& get_cuda_hooks();
RAXIS_API void set_cuda_hooks(const CUDAHooksInterface* hooks);

} // namespace raxis
