#pragma once

#include <string>
#include <sstream>
#include <typeindex>
#include <typeinfo>

#include <raxis/api.hpp>
#include <raxis/array_types.hpp>


namespace raxis {

    class Dtype;
    RAXIS_API const Dtype& default_dtype();
    RAXIS_API void set_default_dtype(const Dtype& dtype);
    
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
    class RAXIS_API Dtype {
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
            if constexpr (std::is_same_v<T, float32>) return Float32;
            else if constexpr (std::is_same_v<T, float64>) return Float64;
            else if constexpr (std::is_same_v<T, int32_t>) return Int32;
            else if constexpr (std::is_same_v<T, int64_t>) return Int64;
            else if constexpr (std::is_same_v<T, uint8_t>) return Uint8;
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
        DtypeKind::Float32, sizeof(float32), "float32", typeid(float32)
    )};
    inline const Dtype Dtype::Float64{DtypeDescriptor(
        DtypeKind::Float64, sizeof(float64), "float64", typeid(float64)
    )};
    inline const Dtype Dtype::Int32{DtypeDescriptor(
        DtypeKind::Int32, sizeof(int32_t), "int32", typeid(int32_t)
    )};
    inline const Dtype Dtype::Int64{DtypeDescriptor(
        DtypeKind::Int64, sizeof(int64_t), "int64", typeid(int64_t)
    )};
    inline const Dtype Dtype::Uint8{DtypeDescriptor(
        DtypeKind::Uint8, sizeof(uint8_t), "uint8", typeid(uint8_t)
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
                    return func(std::type_identity<float32>{});
                case DtypeKind::Float64:
                    return func(std::type_identity<float64>{});
                case DtypeKind::Int32:
                    return func(std::type_identity<int32_t>{});
                case DtypeKind::Int64:
                    return func(std::type_identity<int64_t>{});
                case DtypeKind::Uint8:
                    return func(std::type_identity<uint8_t>{});
                default:
                    throw std::runtime_error("Unsupported");
            }
        }
    };

    // Implement Dtype functions
    inline Dtype::Dtype() : desc_(default_dtype().desc_) {};

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
                std::ostringstream oss;
                oss << *ptr;
                return oss.str();
            } else {
                return std::to_string(*ptr);
            }
        });
    }
}