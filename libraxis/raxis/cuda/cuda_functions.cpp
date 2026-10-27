#include <raxis/cuda/cuda_functions.h>

#include <mutex>
#include <vector>

#include <raxis/cuda/cuda_exception.h>


namespace raxis::cuda {

// From: c10/cuda/CUDAFunctions.cpp (device_count_impl)
int device_count() noexcept {
    // Asked once: the count does not change while the process runs
    static const int count = [] {
        int n = 0;
        // No driver (cudaErrorInsufficientDriver), no GPU
        // (cudaErrorNoDevice): CUDA is just not available
        if (cudaGetDeviceCount(&n) != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        return n;
    }();
    return count;
}

DeviceIndex current_device() {
    int index = 0;
    RAXIS_CUDA_CHECK(cudaGetDevice(&index));
    return static_cast<DeviceIndex>(index);
}

void set_device(DeviceIndex index) {
    RAXIS_CHECK(index >= 0 && index < device_count(),
        "Invalid CUDA device index ", static_cast<int>(index), ": there are ",
        device_count(), " CUDA devices");
    RAXIS_CUDA_CHECK(cudaSetDevice(index));
}

void device_synchronize() {
    RAXIS_CUDA_CHECK(cudaDeviceSynchronize());
}

const cudaDeviceProp& current_device_properties() {
    static std::mutex mutex;
    // Leaked, as the allocator: references handed out stay valid at exit
    static auto* props = new std::vector<cudaDeviceProp>(device_count());
    static auto* queried = new std::vector<bool>(device_count(), false);
    const DeviceIndex device = current_device();
    std::lock_guard<std::mutex> lock(mutex);
    if (!(*queried)[device]) {
        RAXIS_CUDA_CHECK(cudaGetDeviceProperties(&(*props)[device], device));
        (*queried)[device] = true;
    }
    return (*props)[device];
}

void memcpy_and_sync(void* dst, const void* src, size_t nbytes, cudaMemcpyKind kind, cudaStream_t stream) {
    RAXIS_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, kind, stream));
    RAXIS_CUDA_CHECK(cudaStreamSynchronize(stream));
}

} // namespace raxis::cuda
