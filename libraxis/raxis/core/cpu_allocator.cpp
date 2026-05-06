
#include <cstdlib>
#include <stdexcept>

#ifdef _MSC_VER
#include <malloc.h>
#endif

#if RAXIS_USE_MIMALLOC
#include <mimalloc.h>
#endif

#include <raxis/core/allocator.h>
#include <raxis/core/device.h>


namespace raxis {
namespace {

constexpr size_t kAlignment = 64; // AVX-512 friendly

/// Round `n` up to the nearest multiple of `alignment`.
/// Requires: alignment is a power of two.
inline size_t aligned_size(size_t n, size_t alignment) {
    return (n + alignment - 1) & ~(alignment - 1);
}

inline void* aligned_malloc(size_t size, size_t alignment) {
#if RAXIS_USE_MIMALLOC
    return mi_malloc_aligned(size, alignment);
#elif defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void aligned_free(void* ptr) {
#if RAXIS_USE_MIMALLOC
    mi_free(ptr);
#elif defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}



class CPUAllocator final : public Allocator {
public:
    DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return DataPtr(nullptr, Device(DeviceType::CPU));
        }

        size_t alloc_size = aligned_size(nbytes, kAlignment);
        void* data = aligned_malloc(alloc_size, kAlignment);
        if (data == nullptr) {
            throw std::bad_alloc();
        }
        return DataPtr(data, data, &aligned_free, Device(DeviceType::CPU));
    }
};

// Static instance: lives for the duration of the program.
static CPUAllocator g_cpu_allocator;
REGISTER_ALLOCATOR(DeviceType::CPU, &g_cpu_allocator)

} // anonymous namespace
} // namespace raxis