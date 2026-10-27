#pragma once

#include <raxis/core/device.h>
#include <raxis/core/cuda_hooks.h>


namespace raxis {

/// Makes `device` the current CUDA device for the scope (cudaSetDevice) and
/// restores the previous one on exit. CUDA allocates memory and launches
/// kernels on the current device of the calling thread, so every op that
/// allocates or launches for a tensor on cuda:1 must switch first.
/// A no-op for CPU devices and for a CUDA device without an index.
/// From: c10/core/DeviceGuard.h (OptionalDeviceGuard)
class DeviceGuard {
public:
    explicit DeviceGuard(Device device) {
        if (device.is_cuda() && device.has_index()) {
            const auto& hooks = get_cuda_hooks();
            DeviceIndex current = hooks.current_device();
            if (current != device.index()) {
                hooks.set_device(device.index());
                original_index_ = current;
            }
        }
    }

    ~DeviceGuard() {
        if (original_index_ != -1) {
            get_cuda_hooks().set_device(original_index_);
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    DeviceIndex original_index_ = -1;
};

} // namespace raxis
