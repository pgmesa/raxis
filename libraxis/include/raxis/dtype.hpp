#pragma once

#include <string>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include <raxis/array_types.hpp>


namespace raxis {
    
    // Dtype descriptor - runtime type information
    enum class DtypeKind {
        Float32,
        Float64,
        Int32,
        Int64,
        Uint8,
        Bool
    };

    struct DtypeDescriptor {
        DtypeKind kind;
        size_t itemsize;
        std::string name;
        std::type_index type_index;

        DtypeDescriptor(DtypeKind k, size_t size, const std::string& n, 
                        std::type_index ti)
            : kind(k), itemsize(size), name(n), type_index(ti) {}
    };

    // Static dtype registry
    class Dtype {
    public:
        // Predefined dtypes
        static const Dtype Float32;
        static const Dtype Float64;
        static const Dtype Int32;
        static const Dtype Int64;
        static const Dtype Uint8;
        static const Dtype Bool;

        Dtype();

        DtypeKind kind() const { return desc_.kind; }
        size_t itemsize() const { return desc_.itemsize; }
        const std::string& name() const { return desc_.name; }
        std::type_index type_index() const { return desc_.type_index; }

        bool operator==(const Dtype& other) const {
            return desc_.kind == other.desc_.kind;
        }

        bool operator!=(const Dtype& other) const {
            return !(*this == other);
        }

        // Template helper to get dtype from C++ type
        template<typename T>
        static Dtype from_type() {
            if constexpr (std::is_same_v<T, f32>) return Float32;
            else if constexpr (std::is_same_v<T, f64>) return Float64;
            else if constexpr (std::is_same_v<T, i32>) return Int32;
            else if constexpr (std::is_same_v<T, i64>) return Int64;
            else if constexpr (std::is_same_v<T, u8>) return Uint8;
            else if constexpr (std::is_same_v<T, bool>) return Bool;
            else throw std::runtime_error("Unsupported type");
        }
        
        std::string to_string(const void* data) const;
        void fill(void* data, size_t n, double value) const;

    private:
        explicit Dtype(const DtypeDescriptor& desc) : desc_(desc) {}
        DtypeDescriptor desc_;
    };

    // Define static dtypes
    inline const Dtype Dtype::Float32 {DtypeDescriptor(
        DtypeKind::Float32, sizeof(f32), "float32", typeid(f32)
    )};
    inline const Dtype Dtype::Float64{DtypeDescriptor(
        DtypeKind::Float64, sizeof(f64), "float64", typeid(f64)
    )};
    inline const Dtype Dtype::Int32{DtypeDescriptor(
        DtypeKind::Int32, sizeof(i32), "int32", typeid(i32)
    )};
    inline const Dtype Dtype::Int64{DtypeDescriptor(
        DtypeKind::Int64, sizeof(i64), "int64", typeid(i64)
    )};
    inline const Dtype Dtype::Uint8{DtypeDescriptor(
        DtypeKind::Uint8, sizeof(u8), "uint8", typeid(u8)
    )};
    inline const Dtype Dtype::Bool{DtypeDescriptor(
        DtypeKind::Bool, sizeof(bool), "bool", typeid(bool)
    )};    

    class DtypeDispatcher {
    public:
        template<typename Func>
        static auto dispatch(const Dtype& dtype, Func&& func) {
            switch (dtype.kind()) {
                case DtypeKind::Float32:
                    return func(std::type_identity<f32>{});
                case DtypeKind::Float64:
                    return func(std::type_identity<f64>{});
                case DtypeKind::Int32:
                    return func(std::type_identity<i32>{});
                case DtypeKind::Int64:
                    return func(std::type_identity<i64>{});
                default:
                    throw std::runtime_error("Unsupported");
            }
        }
    };

    // Implement Dtype functions
    inline Dtype::Dtype() : desc_(Float32.desc_) {};

    inline void Dtype::fill(void* data, size_t n, double value) const {
        DtypeDispatcher::dispatch(*this, [&](auto type_tag) {
            using T = typename decltype(type_tag)::type;
            auto* ptr = static_cast<T*>(data);
            std::fill(ptr, ptr + n, static_cast<T>(value));
        });
    }

    inline std::string Dtype::to_string(const void* data) const {
        return DtypeDispatcher::dispatch(*this, [&](auto type_tag) -> std::string {
            using T = typename decltype(type_tag)::type;
            const T* ptr = static_cast<const T*>(data);

            if constexpr (std::is_same_v<T, bool>) {
                return (*ptr) ? "true" : "false";
            } else if constexpr (std::is_floating_point_v<T>) {
                // nicer float formatting than std::to_string
                std::ostringstream oss;
                oss << *ptr;
                return oss.str();
            } else {
                return std::to_string(*ptr);
            }
        });
    }
}