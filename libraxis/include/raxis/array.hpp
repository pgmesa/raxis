#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>

#include <raxis/api.hpp>
#include <raxis/dtype.hpp>
#include <raxis/array_types.hpp>
#include <raxis/core/array_impl.hpp>


namespace raxis {
    
    class RAXIS_API NDArray {
    public:
        NDArray() = default;
        explicit NDArray(std::shared_ptr<ArrayImpl> impl):  impl_(std::move(impl)) {};
        
        // TODO
        // NDArray& operator=(double v) && {
        //     return fill_(v);
        // }

        // NDArray& operator=(const NDArray& rhs) && {
        //     return copy_(rhs);
        // }

        // NDArray& operator=(NDArray&& rhs) && {
        //     return copy_(rhs);
        // }

        const void* data_ptr() const;
        void* mutable_data_ptr() const;

        template<typename T>
        const T* data_ptr() const {
            if (Dtype::from_type<T>() != dtype()) {
                throw std::runtime_error("Type mismatch");
            }
            return static_cast<const T*>(data_ptr());
        }

        template<typename T>
        T* mutable_data_ptr() const {
            if (Dtype::from_type<T>() != dtype()) {
                throw std::runtime_error("Type mismatch");
            }
            return static_cast<T*>(mutable_data_ptr());
        }

        int64_t numel() const;
        int ndim() const;
        const IntArray& shape() const;
        const IntArray& strides() const;
        Dtype dtype() const;

        std::string to_string() const;

        // Operators
        NDArray operator+(const NDArray& other) const;

    private:
        std::shared_ptr<ArrayImpl> impl_;
    };

} // namespace