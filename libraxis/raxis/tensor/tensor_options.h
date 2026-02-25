#pragma once

#include <raxis/core/device.h>
#include <raxis/core/dtype.h>
#include <raxis/export.h>

#include <optional>
#include <ostream>
#include <type_traits>


namespace raxis {


inline Dtype dtype_or_default(std::optional<Dtype> dtype) {
    return dtype.value_or(default_dtype());
}

inline Device device_or_default(std::optional<Device> device) {
    return device.value_or(Device(DeviceType::CPU));
}

// ============================================================================
// TensorOptions
// ============================================================================

/// A lightweight "bag of optional properties" for tensor construction.
///
/// Modeled after PyTorch's c10::TensorOptions, but without Layout,
/// MemoryFormat, or pinned_memory — those can be added later without
/// breaking the API.
///
/// The class supports three usage styles:
///
///   1. Implicit conversion from Dtype, Device, or DeviceType:
///        raxis::zeros({3, 3}, dtypes::Float64);
///        raxis::zeros({3, 3}, DeviceType::CUDA);
///
///   2. Builder / chaining:
///        raxis::zeros({3, 3}, raxis::dtype(dtypes::Float64).device(DeviceType::CUDA));
///
///   3. Default (all inherited or global defaults):
///        raxis::zeros({3, 3});
///
struct RAXIS_API TensorOptions {
    TensorOptions() = default;

    /* implicit */ TensorOptions(Dtype dtype) : dtype_(dtype) {}

    /// Construct from a Device.
    ///   raxis::zeros({3, 3}, Device(DeviceType::CUDA, 0));
    ///
    /// NOTE: This is templatized to avoid ambiguity with brace-initialization.
    /// Without the template, `TensorOptions({DeviceType::CUDA, 0})` would be
    /// ambiguous between the copy constructor and this converting constructor,
    /// because `{DeviceType::CUDA, 0}` can construct both a TensorOptions and 
    /// a Device. Template resolution happens after overload resolution, which
    /// breaks the tie. See PyTorch's "NOTE [ TensorOptions Constructors ]".
    template <typename T,
              typename = std::enable_if_t<std::is_same_v<std::decay_t<T>, Device>>>
    /* implicit */ TensorOptions(T&& device) : device_(std::forward<T>(device)) {}

    /// Construct from anything that can construct a Device (e.g. DeviceType).
    /// This enables a single implicit conversion step:
    ///   raxis::zeros({3, 3}, DeviceType::CUDA);
    ///
    /// Without this, DeviceType → Device → TensorOptions would require two
    /// user-defined implicit conversions, which C++ does not allow.
    template <typename... Args,
              typename = std::enable_if_t<std::is_constructible_v<Device, Args&&...>>>
    /* implicit */ TensorOptions(Args&&... args)
        : TensorOptions(Device(std::forward<Args>(args)...)) {}

    [[nodiscard]] TensorOptions dtype(std::optional<Dtype> dtype) const noexcept {
        TensorOptions r = *this;
        r.dtype_ = dtype;
        return r;
    }

    [[nodiscard]] TensorOptions device(std::optional<Device> device) const noexcept {
        TensorOptions r = *this;
        r.device_ = device;
        return r;
    }

    /// Variadic overload so that .device(DeviceType::CUDA, 0) works.
    template <typename... Args,
              typename = std::enable_if_t<std::is_constructible_v<Device, Args&&...>>>
    [[nodiscard]] TensorOptions device(Args&&... args) const noexcept {
        return device(std::make_optional<Device>(std::forward<Args>(args)...));
    }

    [[nodiscard]] TensorOptions requires_grad(std::optional<bool> rg) const noexcept {
        TensorOptions r = *this;
        r.requires_grad_ = rg;
        return r;
    }

    Dtype dtype() const noexcept { return dtype_or_default(dtype_); }
    Device device() const noexcept { return device_or_default(device_); }
    bool requires_grad() const noexcept { return requires_grad_.value_or(false); }

    bool has_dtype() const noexcept { return dtype_.has_value(); }
    bool has_device() const noexcept { return device_.has_value(); }
    bool has_requires_grad() const noexcept { return requires_grad_.has_value(); }

    std::optional<Dtype> dtype_opt() const noexcept { return dtype_; }
    std::optional<Device> device_opt() const noexcept { return device_; }
    std::optional<bool> requires_grad_opt() const noexcept { return requires_grad_; }

    /// Right-biased merge: fields set in `other` override fields in `*this`.
    /// Unset fields in `other` are left unchanged.
    TensorOptions merge_in(const TensorOptions& other) const noexcept {
        TensorOptions merged = *this;
        if (other.has_dtype()) merged.dtype_ = other.dtype_;
        if (other.has_device()) merged.device_ = other.device_;
        if (other.has_requires_grad()) merged.requires_grad_ = other.requires_grad_;
        return merged;
    }

    bool operator==(const TensorOptions& other) const noexcept {
        return dtype() == other.dtype() 
            && device() == other.device() 
            && requires_grad() == other.requires_grad();
    }

    bool operator!=(const TensorOptions& other) const noexcept {
        return !(*this == other);
    }

private:
    std::optional<Dtype> dtype_;
    std::optional<Device> device_;
    std::optional<bool> requires_grad_;
};

// ============================================================================
// Free convenience functions (enable chaining from scratch)
// ============================================================================
//   raxis::zeros({3, 3}, raxis::dtype(dtypes::Float64).device(DeviceType::CUDA));

inline TensorOptions dtype(Dtype dtype) {
    return TensorOptions(dtype);
}

inline TensorOptions device(Device device) {
    return TensorOptions(device);
}

inline TensorOptions requires_grad(bool rg = true) {
    return TensorOptions().requires_grad(rg);
}

RAXIS_API std::ostream& operator<<(std::ostream& stream, const TensorOptions& options);

} // namespace raxis