
#include <raxis/array.hpp>

#include <vector>
#include <memory>
#include <stdexcept>

#include <fmt/format.h>
#include <raxis/array_ops.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/core/array_impl.hpp>
#include <raxis/format/array_fmt.hpp>
#include <raxis/factories.hpp>


namespace raxis {

bool NDArray::defined() const noexcept {
    return impl_ != nullptr;
}

bool NDArray::is_same(const NDArray& other) const noexcept {
    return impl_ == other.impl_;
}

void NDArray::ensure_defined(const char* context) const {
    if (!defined()) {
        std::string msg = "Array is not defined";
        if (context) {
            msg += " (accessing ";
            msg += context;
            msg += ")";
        }
        throw std::runtime_error(msg);
    }
}

const void* NDArray::data_ptr() const {
    ensure_defined("data_ptr");
    return impl_->data();
}

void* NDArray::mutable_data_ptr() const {
    ensure_defined("mutable_data_ptr");
    return impl_->mutable_data();
}
    
const IntArray& NDArray::shape() const { 
    ensure_defined("shape");
    return impl_->shape(); 
}

const IntArray& NDArray::strides() const { 
    ensure_defined("strides");
    return impl_->strides(); 
}

Dtype NDArray::dtype() const {
    ensure_defined("dtype");
    return impl_->dtype(); 
}

int64_t NDArray::numel() const {
    ensure_defined("numel");
    return impl_->numel();
}

int NDArray::ndim() const {
    ensure_defined("ndim");
    return impl_->ndim();
}

IntArray NDArray::byte_strides() const {
    ensure_defined("byte_strides");
    IntArray byte_strides;
    byte_strides.reserve(ndim());

    const int64_t itemsize = static_cast<int64_t>(dtype().itemsize());
    const IntArray& step_strides = strides();
    for (const int64_t s : step_strides)
        byte_strides.push_back(s * itemsize);

    return byte_strides;
}

std::string NDArray::to_string() const {
    if (!defined()) 
        return "raxis(undefined array!)";

    return fmt::format(
        "raxis(shape={}, strides={}, dtype={})", 
        impl_->shape(), impl_->strides(), impl_->dtype().name()
    );
}

// ========================= Operations =========================
NDArray NDArray::clone() const {
    return ::raxis::clone(*this);
}

NDArray& NDArray::copy_(const NDArray& other) const {
    return ::raxis::copy(other, const_cast<NDArray&>(*this));
}

NDArray NDArray::to(Dtype new_dtype) const {
    return ::raxis::to(*this, new_dtype);
}


NDArray NDArray::add(const NDArray& other) const {
    return ::raxis::add(*this, other);
}

NDArray NDArray::sub(const NDArray& other) const {
    return ::raxis::sub(*this, other);
}

NDArray NDArray::mul(const NDArray& other) const {
    return ::raxis::mul(*this, other);
}

NDArray NDArray::div(const NDArray& other) const {
    return ::raxis::div(*this, other);
}


NDArray& NDArray::add_(const NDArray& other) const {
    return ::raxis::add_(const_cast<NDArray&>(*this), other);
}

NDArray& NDArray::sub_(const NDArray& other) const {
    return ::raxis::sub_(const_cast<NDArray&>(*this), other);
}

NDArray& NDArray::mul_(const NDArray& other) const {
    return ::raxis::mul_(const_cast<NDArray&>(*this), other);
}

NDArray& NDArray::div_(const NDArray& other) const {
    return ::raxis::div_(const_cast<NDArray&>(*this), other);
}


NDArray NDArray::rsub(const NDArray& other) const {
    return ::raxis::rsub(*this, other);
}

NDArray NDArray::rdiv(const NDArray& other) const {
    return ::raxis::rdiv(*this, other);
}


NDArray& NDArray::rsub_(const NDArray& other) const {
    return ::raxis::rsub_(const_cast<NDArray&>(*this), other);
}

NDArray& NDArray::rdiv_(const NDArray& other) const {
    return ::raxis::rdiv_(const_cast<NDArray&>(*this), other);
}

// ========================= Operators =========================
NDArray NDArray::operator+(const NDArray& other) const {
    return ::raxis::add(*this, other);
}

NDArray NDArray::operator-(const NDArray& other) const {
    return ::raxis::sub(*this, other);
}

NDArray NDArray::operator*(const NDArray& other) const {
    return ::raxis::mul(*this, other);
}

NDArray NDArray::operator/(const NDArray& other) const {
    return ::raxis::div(*this, other);
}

// In-place
NDArray& NDArray::operator+=(const NDArray& other) const {
    return ::raxis::add_(const_cast<NDArray&>(*this), other);
}

NDArray& NDArray::operator-=(const NDArray& other) const {
    return ::raxis::sub_(const_cast<NDArray&>(*this), other);
}

NDArray& NDArray::operator*=(const NDArray& other) const {
    return ::raxis::mul_(const_cast<NDArray&>(*this), other);
}

NDArray& NDArray::operator/=(const NDArray& other) const {
    return ::raxis::div_(const_cast<NDArray&>(*this), other);
}

}