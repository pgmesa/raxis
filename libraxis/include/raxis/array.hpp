#ifndef ARRAY_HPP
#define ARRAY_HPP

#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>

#include <raxis/dtype.hpp>


namespace raxis {

    using IntArray = std::vector<int64_t>;

    class NDArray {
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

        // Operators
        NDArray operator+(const NDArray& other) const;

    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;

        void init_from_buffer(const void* data, size_t count,
                          const IntArray& shape, const Dtype& dtype);
    };

    // Factory functions with explicit dtype
    NDArray ones(const IntArray& shape, const Dtype& dtype = Dtype::Float32);
    NDArray zeros(const IntArray& shape, const Dtype& dtype = Dtype::Float32);
    NDArray full(const IntArray& shape, double value, 
                const Dtype& dtype = Dtype::Float32);

} // namespace

#endif