#include <raxis/cuda/cuda_stream.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>


namespace raxis::cuda {
namespace {

// From: c10/cuda/CUDAStream.cpp
constexpr int kStreamsPerPool = 32;
constexpr int kMaxPriorities = 4;   // max_compile_time_stream_priorities

/// The stream pools of one device, one per priority level
struct DevicePools {
    std::once_flag init;
    std::array<std::array<cudaStream_t, kStreamsPerPool>, kMaxPriorities> streams{};
    std::array<std::atomic<uint32_t>, kMaxPriorities> next{};
};

/// The priority levels the pools use: as many as the device has, at most 4
int priority_levels() {
    static const int levels = [] {
        int least = 0;
        int greatest = 0;
        RAXIS_CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        return std::min(least - greatest + 1, kMaxPriorities);
    }();
    return levels;
}

DevicePools& device_pools(DeviceIndex device) {
    // Leaked, as the allocators: streams handed out stay valid at exit
    static auto* pools = [] {
        auto* v = new std::vector<std::unique_ptr<DevicePools>>();
        for (int i = 0; i < device_count(); i++) {
            v->push_back(std::make_unique<DevicePools>());
        }
        return v;
    }();
    RAXIS_CHECK(device >= 0 && device < static_cast<int>(pools->size()),
        "Invalid CUDA device index ", static_cast<int>(device));
    return *(*pools)[device];
}

// The current stream of each device for this thread; nullptr means the
// default stream (a thread starts there)
thread_local std::vector<cudaStream_t> t_current_streams;

} // anonymous namespace

cudaStream_t current_stream(DeviceIndex device) {
    if (device < static_cast<DeviceIndex>(t_current_streams.size())) {
        cudaStream_t stream = t_current_streams[device];
        if (stream != nullptr) {
            return stream;
        }
    }
    return default_stream();
}

cudaStream_t current_stream() {
    return current_stream(current_device());
}

void set_current_stream(DeviceIndex device, cudaStream_t stream) {
    RAXIS_CHECK(device >= 0 && device < device_count(),
        "Invalid CUDA device index ", static_cast<int>(device));
    if (t_current_streams.size() < static_cast<size_t>(device_count())) {
        t_current_streams.resize(device_count(), nullptr);
    }
    t_current_streams[device] = stream == default_stream() ? nullptr : stream;
}

cudaStream_t stream_from_pool(DeviceIndex device, int priority) {
    DevicePools& pools = device_pools(device);
    const int levels = priority_levels();
    std::call_once(pools.init, [&] {
        DeviceGuard guard(Device(DeviceType::CUDA, device));
        for (int level = 0; level < levels; level++) {
            for (cudaStream_t& stream : pools.streams[level]) {
                // Lower numbers are higher priorities
                RAXIS_CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -level));
            }
        }
    });
    const int level = std::clamp(-priority, 0, levels - 1);
    const uint32_t index = pools.next[level]++ % kStreamsPerPool;
    return pools.streams[level][index];
}

} // namespace raxis::cuda
