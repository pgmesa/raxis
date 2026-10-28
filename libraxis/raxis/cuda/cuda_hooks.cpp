#include <raxis/core/cuda_hooks.h>

#include <raxis/cuda/cuda_caching_allocator.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/cuda/cuda_host_allocator.h>
#include <raxis/core/device_guard.h>


namespace raxis::cuda {
namespace {

// From: aten/src/ATen/cuda/detail/CUDAHooks.cpp
struct CUDAHooks final : public CUDAHooksInterface {
    bool is_available() const override {
        return device_count() > 0;
    }

    int device_count() const override {
        return cuda::device_count();
    }

    DeviceIndex current_device() const override {
        return cuda::current_device();
    }

    void set_device(DeviceIndex index) const override {
        cuda::set_device(index);
    }

    void synchronize(DeviceIndex index) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, index));
        device_synchronize();
    }

    std::pair<int, int> device_capability(DeviceIndex index) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, index));
        const cudaDeviceProp& props = cuda::current_device_properties();
        return {props.major, props.minor};
    }

    CUDADeviceProperties device_properties(DeviceIndex index) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, index));
        const cudaDeviceProp& props = cuda::current_device_properties();
        return {props.name, props.major, props.minor, static_cast<int64_t>(props.totalGlobalMem),
                props.multiProcessorCount, static_cast<int64_t>(props.sharedMemPerBlockOptin)};
    }

    CUDAMemoryStats memory_stats(DeviceIndex index) const override {
        return cuda::memory_stats(index);
    }

    void reset_peak_memory_stats(DeviceIndex index) const override {
        cuda::reset_peak_memory_stats(index);
    }

    void empty_cache() const override {
        cuda::empty_cache();
    }

    Allocator* pinned_memory_allocator() const override {
        return host_allocator();
    }

    // From: aten/src/ATen/cuda/detail/CUDAHooks.cpp (isPinnedPtr): asks the
    // driver, so memory pinned by someone else counts too
    bool is_pinned(const void* data) const override {
        if (!is_available()) {
            return false;
        }
        cudaPointerAttributes attributes;
        const cudaError_t status = cudaPointerGetAttributes(&attributes, data);
        if (status == cudaErrorInvalidValue) {
            (void)cudaGetLastError();   // a pointer the driver never saw
            return false;
        }
        RAXIS_CUDA_CHECK(status);
        return attributes.type == cudaMemoryTypeHost;
    }

    void host_empty_cache() const override {
        cuda::host_empty_cache();
    }
};

const CUDAHooks g_cuda_hooks;

// Runs when libraxis_cuda is loaded; from then on libraxis reaches CUDA
// through these hooks
struct RegisterCUDAHooks {
    RegisterCUDAHooks() {
        set_cuda_hooks(&g_cuda_hooks);
    }
};

const RegisterCUDAHooks g_register_cuda_hooks;

} // anonymous namespace
} // namespace raxis::cuda
