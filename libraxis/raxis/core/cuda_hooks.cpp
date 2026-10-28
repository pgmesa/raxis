#include <raxis/core/cuda_hooks.h>

#include <atomic>
#include <cstdlib>

#include <raxis/util/exception.h>


namespace raxis {
namespace {

[[noreturn]] void fail_no_cuda() {
    RAXIS_CHECK(false,
        "raxis was not loaded with CUDA support: install the CUDA package "
        "(pip install raxis[cuda12] or raxis[cuda13]) or build with RAXIS_USE_CUDA=ON");
    std::abort();
}

const CUDAHooksInterface g_default_hooks;
std::atomic<const CUDAHooksInterface*> g_hooks{&g_default_hooks};

} // anonymous namespace

bool CUDAHooksInterface::is_available() const {
    return false;
}

int CUDAHooksInterface::device_count() const {
    return 0;
}

DeviceIndex CUDAHooksInterface::current_device() const {
    fail_no_cuda();
}

void CUDAHooksInterface::set_device(DeviceIndex) const {
    fail_no_cuda();
}

void CUDAHooksInterface::synchronize(DeviceIndex) const {
    fail_no_cuda();
}

std::pair<int, int> CUDAHooksInterface::device_capability(DeviceIndex) const {
    fail_no_cuda();
}

CUDAMemoryStats CUDAHooksInterface::memory_stats(DeviceIndex) const {
    fail_no_cuda();
}

void CUDAHooksInterface::reset_peak_memory_stats(DeviceIndex) const {
    fail_no_cuda();
}

void CUDAHooksInterface::empty_cache() const {
    fail_no_cuda();
}

Allocator* CUDAHooksInterface::pinned_memory_allocator() const {
    fail_no_cuda();
}

bool CUDAHooksInterface::is_pinned(const void*) const {
    return false;
}

void CUDAHooksInterface::host_empty_cache() const {
    fail_no_cuda();
}

const CUDAHooksInterface& get_cuda_hooks() {
    return *g_hooks.load(std::memory_order_acquire);
}

void set_cuda_hooks(const CUDAHooksInterface* hooks) {
    g_hooks.store(hooks ? hooks : &g_default_hooks, std::memory_order_release);
}

} // namespace raxis
