#include <raxis/core/cuda_hooks.h>

#include <raxis/cuda/cuda_caching_allocator.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/cuda/cuda_host_allocator.h>
#include <raxis/cuda/cuda_sleep.h>
#include <raxis/cuda/cuda_stream.h>
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

    // From: torch/csrc/cuda/Stream.cpp and Event.cpp, through
    // c10/cuda/impl/CUDAGuardImpl.h

    void* current_stream(DeviceIndex device) const override {
        return cuda::current_stream(device);
    }

    void set_current_stream(DeviceIndex device, void* stream) const override {
        cuda::set_current_stream(device, static_cast<cudaStream_t>(stream));
    }

    void* default_stream(DeviceIndex) const override {
        return cuda::default_stream();
    }

    void* stream_from_pool(DeviceIndex device, int priority) const override {
        return cuda::stream_from_pool(device, priority);
    }

    bool stream_query(DeviceIndex device, void* stream) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        const cudaError_t status = cudaStreamQuery(static_cast<cudaStream_t>(stream));
        if (status == cudaErrorNotReady) {
            (void)cudaGetLastError();
            return false;
        }
        RAXIS_CUDA_CHECK(status);
        return true;
    }

    void stream_synchronize(DeviceIndex device, void* stream) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        RAXIS_CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
    }

    void stream_wait_event(DeviceIndex device, void* stream, void* event) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        RAXIS_CUDA_CHECK(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(event), 0));
    }

    // From: aten/src/ATen/cuda/CUDAEvent.h (createEvent): events without
    // timing are cheaper to record, so timing is opt-in; a blocking event
    // makes synchronize() sleep instead of spin
    void* event_create(DeviceIndex device, bool enable_timing, bool blocking) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        unsigned int flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
        if (blocking) {
            flags |= cudaEventBlockingSync;
        }
        cudaEvent_t event;
        RAXIS_CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
        return event;
    }

    void event_destroy(DeviceIndex device, void* event) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        RAXIS_CUDA_CHECK(cudaEventDestroy(static_cast<cudaEvent_t>(event)));
    }

    void event_record(DeviceIndex device, void* event, void* stream) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        RAXIS_CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)));
    }

    bool event_query(DeviceIndex device, void* event) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        const cudaError_t status = cudaEventQuery(static_cast<cudaEvent_t>(event));
        if (status == cudaErrorNotReady) {
            (void)cudaGetLastError();
            return false;
        }
        RAXIS_CUDA_CHECK(status);
        return true;
    }

    void event_synchronize(DeviceIndex device, void* event) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        RAXIS_CUDA_CHECK(cudaEventSynchronize(static_cast<cudaEvent_t>(event)));
    }

    float event_elapsed_time(DeviceIndex device, void* start, void* end) const override {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        float ms = 0;
        RAXIS_CUDA_CHECK(cudaEventElapsedTime(&ms, static_cast<cudaEvent_t>(start), static_cast<cudaEvent_t>(end)));
        return ms;
    }

    // A CUDA tensor's block, or a pinned CPU tensor's (the host allocator
    // needs the same note for asynchronous copies from it)
    void record_stream(const DataPtr& data_ptr, void* stream) const override {
        auto s = static_cast<cudaStream_t>(stream);
        if (data_ptr.device().is_cuda()) {
            cuda::record_stream(data_ptr, s);
        } else {
            cuda::host_allocator_record_stream(data_ptr, s);
        }
    }

    void sleep(int64_t cycles) const override {
        cuda::sleep(cycles);
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
