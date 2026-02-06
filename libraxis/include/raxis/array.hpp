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
    NDArray(const NDArray &array) = default;
    NDArray(NDArray &&array) = default;

    NDArray& operator=(const NDArray& x) & noexcept {
        impl_ = x.impl_;
        return *this;
    }
    NDArray& operator=(NDArray&& x) & noexcept {
        impl_ = std::move(x.impl_);
        return *this;
    }

    // t1[:25] = t2
    NDArray& operator=(const NDArray& rhs) && {
        return copy_(rhs);
    }

    // t1[3:5] = t3[2:4]
    NDArray& operator=(NDArray&& rhs) && {
        return copy_(rhs);
    }

    // t1[3:5] = 3
    // NDArray& operator=(Scalar v) && {
    //     return fill_(v);
    // }

    bool defined() const noexcept;
    bool is_same(const NDArray& other) const noexcept;

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
    IntArray byte_strides() const;
    Dtype dtype() const;

    std::string to_string() const;

    // Operations
    NDArray clone() const;
    NDArray& copy_(const NDArray& other) const;
    NDArray to(Dtype dtype) const;

    NDArray add(const NDArray& other) const;
    NDArray sub(const NDArray& other) const;
    NDArray mul(const NDArray& other) const;
    NDArray div(const NDArray& other) const;
    
    NDArray& add_(const NDArray& other) const;
    NDArray& sub_(const NDArray& other) const;
    NDArray& mul_(const NDArray& other) const;
    NDArray& div_(const NDArray& other) const;

    NDArray rsub(const NDArray& other) const;
    NDArray rdiv(const NDArray& other) const;

    NDArray& rsub_(const NDArray& other) const;
    NDArray& rdiv_(const NDArray& other) const;

    // Operators
    NDArray operator+(const NDArray& other) const;
    NDArray operator-(const NDArray& other) const;
    NDArray operator*(const NDArray& other) const;
    NDArray operator/(const NDArray& other) const;

    NDArray& operator+=(const NDArray& other) const;
    NDArray& operator-=(const NDArray& other) const;
    NDArray& operator*=(const NDArray& other) const;
    NDArray& operator/=(const NDArray& other) const;

private:
    std::shared_ptr<ArrayImpl> impl_;

    void ensure_defined(const char* context = nullptr) const;
};

} // namespace raxis