#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <ostream>

#include <raxis/core/dtype.h>
#include <raxis/util/exception.h>


namespace raxis {

// ============================================================================
// Scalar — type-preserving wrapper for a single numeric value
// ============================================================================
//
// Scalar holds a single numeric value alongside a tag that records whether the
// original value was boolean, integral, or floating-point.  This tag is used
// during type promotion so that, for example:
//
//     int_tensor + Scalar(3)   → stays int   (3 is integral)
//     int_tensor + Scalar(3.0) → promotes to float

class Scalar {
public:
    enum class Tag : uint8_t {
        Bool,
        Integral,
        Floating,
    };

    template <typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0>
    Scalar(T value) : tag_(Tag::Bool), ival_(value ? 1 : 0) {}

    // Integers (any non-bool integral type)
    template <
        typename T,
        std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
    Scalar(T value) : tag_(Tag::Integral), ival_(static_cast<int64_t>(value)) {}

    // Floating point
    template <
        typename T,
        std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
    Scalar(T value) : tag_(Tag::Floating), fval_(static_cast<double>(value)) {}

    // Default: zero (integral)
    Scalar() : tag_(Tag::Integral), ival_(0) {}

    Tag tag() const { return tag_; }

    bool is_bool() const { return tag_ == Tag::Bool; }
    bool is_integral() const { return tag_ == Tag::Integral; }
    bool is_floating() const { return tag_ == Tag::Floating; }

    Dtype dtype() const {
        switch (tag_) {
            case Tag::Bool:     return dtypes::Bool;
            case Tag::Integral: return dtypes::Int64;
            case Tag::Floating: return dtypes::Float64;
        }
        return dtypes::Undefined;
    }

    template <typename T>
    T to() const {
        if constexpr (std::is_same_v<T, bool>) {
            // Any non-zero value is truthy
            return is_floating() ? (fval_ != 0.0) : (ival_ != 0);
        } else if constexpr (std::is_floating_point_v<T>) {
            return is_floating()
                ? static_cast<T>(fval_)
                : static_cast<T>(ival_);
        } else if constexpr (std::is_integral_v<T>) {
            return is_floating()
                ? static_cast<T>(fval_)
                : static_cast<T>(ival_);
        } else {
            static_assert(sizeof(T) == 0, "Unsupported type for Scalar::to<T>()");
        }
    }

    Scalar operator-() const {
        RAXIS_INTERNAL_ASSERT(!is_bool(), "Cannot negate a boolean scalar");
        return is_floating() ? Scalar(-fval_) : Scalar(-ival_);
    }

    bool operator==(const Scalar& other) const {
        return to<double>() == other.to<double>();
    }
    bool operator!=(const Scalar& other) const { return !(*this == other); }

    std::string to_string() const {
        switch (tag_) {
            case Tag::Bool:     return ival_ ? "true" : "false";
            case Tag::Integral: return std::to_string(ival_);
            case Tag::Floating: return std::to_string(fval_);
        }
        return "undefined";
    }

private:
    Tag tag_;

    // We use separate members instead of a union so that the class is trivially
    // copyable without needing a user-defined copy constructor (unions with
    // non-trivial members need one). Only one is "active" depending on tag_, but
    // reading the other is harmless for our trivial types and the optimizer will
    // see through it.  Total size: 1 (tag) + 7 (padding) + 8 (largest member)
    // = 16 bytes if both members are present. Using a union is also fine and
    // saves 8 bytes

    union {
        int64_t ival_;
        double  fval_;
    };
};

inline std::ostream& operator<<(std::ostream& os, const Scalar& s) {
    return os << s.to_string();
}

// Sanity checks
static_assert(sizeof(Scalar) == 16, "Scalar should be 16 bytes");
static_assert(std::is_trivially_copyable_v<Scalar>, "Scalar must be trivially copyable");

} // namespace raxis