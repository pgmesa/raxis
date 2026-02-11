#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>

#include <raxis/core/export.h>
#include <raxis/core/dtype.h>
#include <raxis/core/types.h>
#include <raxis/tensor/detail/tensor_impl.h>


namespace raxis {
    
class RAXIS_API Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<TensorImpl> impl):  impl_(std::move(impl)) {};
    Tensor(const Tensor &tensor) = default;
    Tensor(Tensor &&tensor) = default;

    Tensor& operator=(const Tensor& x) & noexcept {
        impl_ = x.impl_;
        return *this;
    }
    Tensor& operator=(Tensor&& x) & noexcept {
        impl_ = std::move(x.impl_);
        return *this;
    }

    // t1[:25] = t2
    Tensor& operator=(const Tensor& rhs) && {
        return copy_(rhs);
    }

    // t1[3:5] = t3[2:4]
    Tensor& operator=(Tensor&& rhs) && {
        return copy_(rhs);
    }

    // t1[3:5] = 3
    // Tensor& operator=(Scalar v) && {
    //     return fill_(v);
    // }

    bool defined() const noexcept;
    bool is_same(const Tensor& other) const noexcept;

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
    Tensor clone() const;
    Tensor& copy_(const Tensor& other) const;
    Tensor to(Dtype dtype) const;

    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    
    Tensor& add_(const Tensor& other) const;
    Tensor& sub_(const Tensor& other) const;
    Tensor& mul_(const Tensor& other) const;
    Tensor& div_(const Tensor& other) const;

    Tensor rsub(const Tensor& other) const;
    Tensor rdiv(const Tensor& other) const;

    Tensor& rsub_(const Tensor& other) const;
    Tensor& rdiv_(const Tensor& other) const;

    // Operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor& operator+=(const Tensor& other) const;
    Tensor& operator-=(const Tensor& other) const;
    Tensor& operator*=(const Tensor& other) const;
    Tensor& operator/=(const Tensor& other) const;

private:
    std::shared_ptr<TensorImpl> impl_;

    void ensure_defined(const char* context = nullptr) const;
};

} // namespace raxis