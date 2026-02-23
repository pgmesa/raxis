
#include <raxis/core/allocator.h>

#include <array>
#include <stdexcept>
#include <string>

#include <raxis/core/device.h>
#include <raxis/util/exception.h>


namespace raxis {
namespace {

constexpr size_t kNumDeviceTypes = static_cast<size_t>(DeviceType::NumTypes);
std::array<Allocator*, kNumDeviceTypes> g_allocators = {};

} // anonymous namespace

Allocator* get_allocator(DeviceType type) {
    auto index = static_cast<size_t>(type);
    if (index >= kNumDeviceTypes) {
        throw std::out_of_range(
            "get_allocator: invalid DeviceType " + std::to_string(index));
    }
    Allocator* alloc = g_allocators[index];
    RAXIS_CHECK(alloc != nullptr, 
        "No allocator registered for DeviceType ", device_type_name(type, false));

    return alloc;
}

void set_allocator(DeviceType type, Allocator* allocator) {
    auto index = static_cast<size_t>(type);
    if (index >= kNumDeviceTypes) {
        throw std::out_of_range(
            "set_allocator: invalid DeviceType " + std::to_string(index));
    }
    g_allocators[index] = allocator;
}

} // namespace raxis