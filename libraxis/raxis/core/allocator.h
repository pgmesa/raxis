#pragma once

#include <raxis/core/device.h>
#include <raxis/util/data_ptr.h>
#include <raxis/export.h>


namespace raxis {

struct RAXIS_API Allocator {
    virtual ~Allocator() = default;
    virtual DataPtr allocate(size_t nbytes) = 0;
};

RAXIS_API Allocator* get_allocator(DeviceType device_type);
RAXIS_API void set_allocator(DeviceType device_type, Allocator* allocator);

template <DeviceType t>
struct AllocatorRegisterer {
    explicit AllocatorRegisterer(Allocator* alloc) {
        set_allocator(t, alloc);
    }
};

#define REGISTER_ALLOCATOR(device_type, allocator_ptr)    \
    namespace {                                                  \
    static ::raxis::AllocatorRegisterer<device_type>             \
        g_allocator_registerer(allocator_ptr);                   \
    }

}