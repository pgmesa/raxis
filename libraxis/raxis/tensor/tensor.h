#pragma once

#include <memory>
#include <vector>
#include <stdexcept>
#include <cstdint>

#include <raxis/export.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>
#include <raxis/core/types.h>
#include <raxis/core/scalar.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/tensor_options.h>


namespace raxis {
    
class RAXIS_API Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::shared_ptr<detail::TensorImpl> impl):  impl_(std::move(impl)) {};
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
    Tensor& operator=(Scalar v) && {
        return fill_(v);
    }

    bool defined() const noexcept;
    bool is_same(const Tensor& other) const noexcept;

    const void* data_ptr() const;
    void* mutable_data_ptr() const;

    template<typename T>
    const T* data_ptr() const {
        RAXIS_CHECK(Dtype::from_type<T>() == dtype(), "Type mismatch");
        return static_cast<const T*>(data_ptr());
    }

    template<typename T>
    T* mutable_data_ptr() const {
        RAXIS_CHECK(Dtype::from_type<T>() == dtype(), "Type mismatch");
        return static_cast<T*>(mutable_data_ptr());
    }

    int64_t numel() const;
    int ndim() const;
    const IntArray& shape() const;
    const IntArray& strides() const;
    IntArray byte_strides() const;
    Dtype dtype() const;
    Device device() const;
    bool requires_grad() const;
    TensorOptions options() const;
    
    bool is_cpu() const;
    bool is_cuda() const;

    detail::TensorImpl* unsafe_get_tensor_impl() const;

    std::string to_string() const;

    // Operations
    Tensor clone() const;
    Tensor& copy_(const Tensor& other) const;
    Tensor to(Dtype dtype) const;

    Tensor& fill_(Scalar value) const;
    Tensor& zero_() const;

    Tensor add(const Tensor& other) const;
    Tensor add(Scalar s) const;
    Tensor sub(const Tensor& other) const;
    Tensor sub(Scalar s) const;
    Tensor mul(const Tensor& other) const;
    Tensor mul(Scalar s) const;
    Tensor div(const Tensor& other) const;
    Tensor div(Scalar s) const;

    Tensor& add_(const Tensor& other) const;
    Tensor& add_(Scalar s) const;
    Tensor& sub_(const Tensor& other) const;
    Tensor& sub_(Scalar s) const;
    Tensor& mul_(const Tensor& other) const;
    Tensor& mul_(Scalar s) const;
    Tensor& div_(const Tensor& other) const;
    Tensor& div_(Scalar s) const;

    Tensor rsub(const Tensor& other) const;
    Tensor rsub(Scalar s) const;
    Tensor rdiv(const Tensor& other) const;
    Tensor rdiv(Scalar s) const;

    Tensor& rsub_(const Tensor& other) const;
    Tensor& rsub_(Scalar s) const;
    Tensor& rdiv_(const Tensor& other) const;
    Tensor& rdiv_(Scalar s) const;

    // Operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator+(Scalar s) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator-(Scalar s) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(Scalar s) const;
    Tensor operator/(const Tensor& other) const;
    Tensor operator/(Scalar s) const;

    Tensor& operator+=(const Tensor& other) const;
    Tensor& operator+=(Scalar s) const;
    Tensor& operator-=(const Tensor& other) const;
    Tensor& operator-=(Scalar s) const;
    Tensor& operator*=(const Tensor& other) const;
    Tensor& operator*=(Scalar s) const;
    Tensor& operator/=(const Tensor& other) const;
    Tensor& operator/=(Scalar s) const;

private:
    std::shared_ptr<detail::TensorImpl> impl_;

    void ensure_defined(const char* context = nullptr) const;
};

} // namespace raxis