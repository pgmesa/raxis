#pragma once

#include <cstdint>
#include <string>
#include <ostream>

#include <raxis/export.h>


namespace raxis {

using DeviceIndex = int8_t;

enum class DeviceType : int8_t {
    CPU = 0,
    CUDA = 1,
    NumTypes
};

constexpr const char* device_type_name(DeviceType type, bool lower_case) {
    switch (type) {
        case DeviceType::CPU:    
            return lower_case? "cpu" : "CPU";
        case DeviceType::CUDA:   
            return lower_case? "cuda" : "CUDA"; 
        default:
            return "undefined";
    }
}

struct RAXIS_API Device {

    /* implicit */ Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {}


    /// Constructs a `Device` from a string description, for convenience.
    /// The string supplied must follow the following schema:
    /// `(cpu|cuda)[:<device-index>]`
    /// where `cpu` or `cuda` specifies the device type, and
    /// `:<device-index>` optionally specifies a device index.
    /* implicit */ Device(const std::string& str);

    bool operator==(const Device& other) const noexcept {
        return this->type_ == other.type_ && this->index_ == other.index_;
    }

    bool operator!=(const Device& other) const noexcept {
        return !(*this == other);
    }

    void set_index(DeviceIndex index) {
        index_ = index;
    }

    DeviceType type() const noexcept {
        return type_;
    }

    DeviceIndex index() const noexcept {
        return index_;
    }

    bool has_index() const noexcept {
        return index_ != -1;
    }

    bool is_cpu() const noexcept {
        return type_ == DeviceType::CPU;
    }
    
    bool is_cuda() const noexcept {
        return type_ == DeviceType::CUDA;
    }

    std::string str() const;

private:
    DeviceType type_;
    DeviceIndex index_ = -1;
};

RAXIS_API std::ostream& operator<<(std::ostream& stream, const Device& device);

}