#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <algorithm>
#include <array>
#include <ostream>

#include <raxis/export.h>
#include <raxis/util/exception.h>


namespace raxis {

using float32 = float;
using float64 = double;

enum class DtypeKind : uint8_t {
    Undefined = 0,
    Bool,
    Uint8,
    Int32,
    Int64,
    Float32,
    Float64,
    NumTypes
};

// ============================================================================
// Metadata functions
// ============================================================================

constexpr size_t dtype_itemsize(DtypeKind kind) {
    switch (kind) {
        case DtypeKind::Bool:    return sizeof(bool);
        case DtypeKind::Uint8:   return sizeof(uint8_t);
        case DtypeKind::Int32:   return sizeof(int32_t);
        case DtypeKind::Int64:   return sizeof(int64_t);
        case DtypeKind::Float32: return sizeof(float32);
        case DtypeKind::Float64: return sizeof(float64);
        default: return 0;
    }
}

constexpr const char* dtype_name(DtypeKind kind) {
    switch (kind) {
        case DtypeKind::Bool:    return "bool";
        case DtypeKind::Uint8:   return "uint8";
        case DtypeKind::Int32:   return "int32";
        case DtypeKind::Int64:   return "int64";
        case DtypeKind::Float32: return "float32";
        case DtypeKind::Float64: return "float64";
        default: return "undefined";
    }
}

constexpr bool dtype_is_floating(DtypeKind kind) {
    return kind == DtypeKind::Float32 || kind == DtypeKind::Float64;
}

constexpr bool dtype_is_integral(DtypeKind kind, bool include_bool = false) {
    if (include_bool && kind == DtypeKind::Bool) return true;
    return kind == DtypeKind::Uint8 || 
           kind == DtypeKind::Int32 || 
           kind == DtypeKind::Int64;
}

constexpr bool dtype_is_bool(DtypeKind kind) {
    return kind == DtypeKind::Bool;
}

constexpr bool dtype_is_signed(DtypeKind kind) {
    return kind == DtypeKind::Int32 || 
           kind == DtypeKind::Int64 ||
           kind == DtypeKind::Float32 ||
           kind == DtypeKind::Float64;
}

// ============================================================================
// Dtype wrapper class
// ============================================================================

class RAXIS_API Dtype {
    DtypeKind kind_;
    
public: 
    constexpr Dtype()
        : kind_(DtypeKind::Undefined) {}

    constexpr explicit Dtype(DtypeKind kind)
        : kind_(kind) {}
    
    constexpr DtypeKind kind() const { 
        return kind_; 
    }
    
    constexpr size_t itemsize() const { 
        return dtype_itemsize(kind_);
    }

    constexpr const char* name() const {
        return dtype_name(kind_);
    }

    constexpr bool defined() const {
        return kind_ != DtypeKind::Undefined;
    }
    
    constexpr bool is_floating() const {
        return dtype_is_floating(kind_);
    }

    constexpr bool is_integral(bool include_bool = false) const {
        return dtype_is_integral(kind_, include_bool);
    }
    
    constexpr bool is_bool() const {
        return dtype_is_bool(kind_);
    }

    constexpr bool is_signed() const {
        return dtype_is_signed(kind_);
    }
    
    constexpr bool operator==(Dtype other) const {
        return kind_ == other.kind_;
    }

    constexpr bool operator!=(Dtype other) const {
        return kind_ != other.kind_;
    }
    
    template<typename T>
    static constexpr Dtype from_type() {
        if constexpr (std::is_same_v<T, bool>)          return Dtype(DtypeKind::Bool);
        else if constexpr (std::is_same_v<T, uint8_t>)  return Dtype(DtypeKind::Uint8);
        else if constexpr (std::is_same_v<T, int32_t>)  return Dtype(DtypeKind::Int32);
        else if constexpr (std::is_same_v<T, int64_t>)  return Dtype(DtypeKind::Int64);
        else if constexpr (std::is_same_v<T, float32>)  return Dtype(DtypeKind::Float32);
        else if constexpr (std::is_same_v<T, float64>)  return Dtype(DtypeKind::Float64);
        else static_assert(sizeof(T) == 0, "Unsupported type for Dtype::from_type");
    }
    
    // Runtime type dispatch
    template<typename Func>
    auto dispatch_all_types(Func&& func) const {
        switch (kind_) {
            case DtypeKind::Bool:
                return func(std::type_identity<bool>{});
            case DtypeKind::Uint8:
                return func(std::type_identity<uint8_t>{});
            case DtypeKind::Int32:
                return func(std::type_identity<int32_t>{});
            case DtypeKind::Int64:
                return func(std::type_identity<int64_t>{});
            case DtypeKind::Float32:
                return func(std::type_identity<float32>{});
            case DtypeKind::Float64:
                return func(std::type_identity<float64>{});
            default:
                RAXIS_INTERNAL_ASSERT(false, "Cannot dispatch undefined dtype");
        }
    }

    template<typename Func>
    auto dispatch_all_types_except_bool(Func&& func) const {
        switch (kind_) {
            case DtypeKind::Uint8:
                return func(std::type_identity<uint8_t>{});
            case DtypeKind::Int32:
                return func(std::type_identity<int32_t>{});
            case DtypeKind::Int64:
                return func(std::type_identity<int64_t>{});
            case DtypeKind::Float32:
                return func(std::type_identity<float32>{});
            case DtypeKind::Float64:
                return func(std::type_identity<float64>{});
            default:
                RAXIS_INTERNAL_ASSERT(false, "Cannot dispatch undefined dtype");
        }
    }

    template<typename Func>
    auto dispatch_signed_types(Func&& func) const {
        switch (kind_) {
            case DtypeKind::Int32:
                return func(std::type_identity<int32_t>{});
            case DtypeKind::Int64:
                return func(std::type_identity<int64_t>{});
            case DtypeKind::Float32:
                return func(std::type_identity<float32>{});
            case DtypeKind::Float64:
                return func(std::type_identity<float64>{});
            default:
                RAXIS_INTERNAL_ASSERT(false, "Cannot dispatch undefined dtype");
        }
    }

    template<typename Func>
    auto dispatch_floating_types(Func&& func) const {
        switch (kind_) {
            case DtypeKind::Float32:
                return func(std::type_identity<float32>{});
            case DtypeKind::Float64:
                return func(std::type_identity<float64>{});
            default:
                RAXIS_INTERNAL_ASSERT(false, 
                    "Cannot float dispatch '", dtype_name(kind_), "' dtype");
        }
    }

    template<typename Func>
    auto dispatch_integral_types(Func&& func) const {
        // default: no bool
        switch (kind_) {
            case DtypeKind::Uint8:  return func(std::type_identity<uint8_t>{});
            case DtypeKind::Int32:  return func(std::type_identity<int32_t>{});
            case DtypeKind::Int64:  return func(std::type_identity<int64_t>{});
            default: break;
        }
        RAXIS_INTERNAL_ASSERT(false, 
            "Cannot integer dispatch '", dtype_name(kind_), "' dtype");
    }

    template<typename Func>
    auto dispatch_integral_and_bool_types(Func&& func) const {
        switch (kind_) {
            case DtypeKind::Bool:   return func(std::type_identity<bool>{});
            case DtypeKind::Uint8:  return func(std::type_identity<uint8_t>{});
            case DtypeKind::Int32:  return func(std::type_identity<int32_t>{});
            case DtypeKind::Int64:  return func(std::type_identity<int64_t>{});
            default: break;
        }
        RAXIS_INTERNAL_ASSERT(false, 
            "Cannot integer dispatch '", dtype_name(kind_), "' dtype");
    }

    // Utility: convert single element to string
    std::string to_string(const void* data) const;
};

inline std::ostream& operator<<(std::ostream& os, Dtype dtype) {
    return os << dtype.name();
}

inline std::ostream& operator<<(std::ostream& os, DtypeKind kind) {
    return os << dtype_name(kind);
}

namespace dtypes {
inline static constexpr Dtype Undefined = Dtype(DtypeKind::Undefined);
inline static constexpr Dtype Bool      = Dtype(DtypeKind::Bool);
inline static constexpr Dtype Uint8     = Dtype(DtypeKind::Uint8);
inline static constexpr Dtype Int32     = Dtype(DtypeKind::Int32);
inline static constexpr Dtype Int64     = Dtype(DtypeKind::Int64);
inline static constexpr Dtype Float32   = Dtype(DtypeKind::Float32);
inline static constexpr Dtype Float64   = Dtype(DtypeKind::Float64);
}

// Ensure it's tiny and efficient
static_assert(sizeof(Dtype) == 1, "Dtype must be 1 byte");
static_assert(std::is_trivially_copyable_v<Dtype>, "Dtype must be trivially copyable");

inline Dtype parse_dtype(const std::string& name) {
    // Normalize: trim + lowercase
    std::string normalized = name;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    // Simple lookup
    if (normalized == "bool")                          return dtypes::Bool;
    if (normalized == "uint8" || normalized == "u1")   return dtypes::Uint8;
    if (normalized == "int32" || normalized == "i4")   return dtypes::Int32;
    if (normalized == "int64" || normalized == "i8")   return dtypes::Int64;
    if (normalized == "float32" || normalized == "f4") return dtypes::Float32;
    if (normalized == "float64" || normalized == "f8") return dtypes::Float64;
    
    RAXIS_CHECK(false, "Unknown dtype: '", name, "'");
}

// ============================================================================
// Type promotion
// ============================================================================

namespace detail {
    // Short aliases for the promotion table
    constexpr auto b1 = DtypeKind::Bool;
    constexpr auto u1 = DtypeKind::Uint8;
    constexpr auto i4 = DtypeKind::Int32;
    constexpr auto i8 = DtypeKind::Int64;
    constexpr auto f4 = DtypeKind::Float32;
    constexpr auto f8 = DtypeKind::Float64;
    constexpr auto ud = DtypeKind::Undefined;
    
    // Map dtype to index in promotion table
    constexpr size_t dtype_to_index(DtypeKind kind) {
        return static_cast<size_t>(kind) - 1;  // Skip Undefined
    }
    
    constexpr size_t NUM_DTYPES = static_cast<size_t>(DtypeKind::NumTypes) - 1;
    
    // Type promotion lookup table
    // clang-format off
    constexpr std::array<std::array<DtypeKind, NUM_DTYPES>, NUM_DTYPES>
        promotion_table = {{
        /*        b1  u1  i4  i8  f4  f8*/
        /* b1 */ {b1, u1, i4, i8, f4, f8},
        /* u1 */ {u1, u1, i4, i8, f4, f8},
        /* i4 */ {i4, i4, i4, i8, f4, f8},
        /* i8 */ {i8, i8, i8, i8, f4, f8},
        /* f4 */ {f4, f4, f4, f4, f4, f8},
        /* f8 */ {f8, f8, f8, f8, f8, f8},
    }};
    // clang-format on
    
} // namespace detail

constexpr Dtype promote_types(Dtype a, Dtype b) {
    if (!a.defined() || !b.defined()) {
        return dtypes::Undefined;
    }
    
    if (a == b) return a;
    
    // Lookup in table
    size_t idx_a = detail::dtype_to_index(a.kind());
    size_t idx_b = detail::dtype_to_index(b.kind());
    
    DtypeKind result = detail::promotion_table[idx_a][idx_b];
    return Dtype(result);
}

constexpr bool can_cast(Dtype from, Dtype to) {
    // Disallow float -> integral, e.g., int_tensor *= float is disallowed.
    if (from.is_floating() && to.is_integral(false)) {
        return false;
    }

    // Disallow `bool_tensor += 5`
    if (!from.is_bool() && to.is_bool()) {
        return false;
    }

    return true;
}

namespace raxis::test {
    // Verify promotion rules at compile-time
    static_assert(promote_types(dtypes::Bool, dtypes::Bool) == dtypes::Bool);
    static_assert(promote_types(dtypes::Bool, dtypes::Uint8) == dtypes::Uint8);
    static_assert(promote_types(dtypes::Uint8, dtypes::Int32) == dtypes::Int32);
    static_assert(promote_types(dtypes::Int32, dtypes::Int64) == dtypes::Int64);
    static_assert(promote_types(dtypes::Int64, dtypes::Float32) == dtypes::Float32);
    static_assert(promote_types(dtypes::Float32, dtypes::Float64) == dtypes::Float64);
    
    // Verify symmetry
    static_assert(promote_types(dtypes::Uint8, dtypes::Int32) == 
                  promote_types(dtypes::Int32, dtypes::Uint8));
}

// ============================================================================
// Default dtype management
// ============================================================================

constexpr Dtype default_float_dtype() {
    return dtypes::Float32;
}

constexpr Dtype default_int_dtype() {
    return dtypes::Int64;
}

RAXIS_API Dtype default_dtype();
RAXIS_API void set_default_dtype(Dtype dtype);

} // namespace raxis