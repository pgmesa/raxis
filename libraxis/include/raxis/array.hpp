#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <format>

#include <raxis/api.hpp>
#include <raxis/dtype.hpp>
#include <raxis/array_types.hpp>


namespace raxis {
    
    class RAXIS_API NDArray {
    public:
        NDArray(void* data, const IntArray& shape, const Dtype& dtype);
        
        template<typename T>
        NDArray(const std::vector<T>& data, const IntArray& shape) {
            init_from_buffer(data.data(), data.size(), shape, Dtype::from_type<T>());
        }

        ~NDArray();

        size_t numel() const;
        size_t len() const;
        IntArray shape() const;
        IntArray strides() const;
        const Dtype& dtype() const;
        void* data() const;

        // Type-safe accessors
        template<typename T>
        T* data_ptr() {
            if (Dtype::from_type<T>() != dtype()) {
                throw std::runtime_error("Type mismatch");
            }
            return static_cast<T*>(data());
        }

        template<typename T>
        const T* data_ptr() const {
            if (Dtype::from_type<T>() != dtype()) {
                throw std::runtime_error("Type mismatch");
            }
            return static_cast<const T*>(data());
        }

        std::string to_string() const;

        // Operators
        NDArray operator+(const NDArray& other) const;

    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;

        void init_from_buffer(const void* data, size_t count,
                          const IntArray& shape, const Dtype& dtype);
    };

    // Factory functions with explicit dtype
    RAXIS_API NDArray ones(const IntArray& shape, const Dtype& dtype = Dtype::Float32);
    RAXIS_API NDArray zeros(const IntArray& shape, const Dtype& dtype = Dtype::Float32);
    RAXIS_API NDArray full(const IntArray& shape, double value, 
                const Dtype& dtype = Dtype::Float32);

} // namespace


template <>
struct std::formatter<raxis::NDArray> : std::formatter<std::string_view>
{
    auto format(const raxis::NDArray& a, format_context& ctx) const
    {
        auto s = a.to_string();
        return std::formatter<std::string_view>::format(s, ctx);
    }
};