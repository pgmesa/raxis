#include <raxis/core/cuda_hooks.h>

#include <raxis/cuda/cuda_caching_allocator.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
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

    CUDAMemoryStats memory_stats(DeviceIndex index) const override {
        return cuda::memory_stats(index);
    }

    void reset_peak_memory_stats(DeviceIndex index) const override {
        cuda::reset_peak_memory_stats(index);
    }

    void empty_cache() const override {
        cuda::empty_cache();
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
