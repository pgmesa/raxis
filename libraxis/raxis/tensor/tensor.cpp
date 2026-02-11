
#include <raxis/tensor/tensor.h>

#include <vector>
#include <memory>
#include <stdexcept>

#include <fmt/format.h>

#include <raxis/format.h>
#include <raxis/ops/binary_ops.h>

#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/factories.h>


namespace raxis {

bool Tensor::defined() const noexcept {
    return impl_ != nullptr;
}

bool Tensor::is_same(const Tensor& other) const noexcept {
    return impl_ == other.impl_;
}

void Tensor::ensure_defined(const char* context) const {
    if (!defined()) {
        std::string msg = "Tensor is not defined";
        if (context) {
            msg += " (accessing ";
            msg += context;
            msg += ")";
        }
        throw std::runtime_error(msg);
    }
}

const void* Tensor::data_ptr() const {
    ensure_defined("data_ptr");
    return impl_->data();
}

void* Tensor::mutable_data_ptr() const {
    ensure_defined("mutable_data_ptr");
    return impl_->mutable_data();
}
    
const IntArray& Tensor::shape() const { 
    ensure_defined("shape");
    return impl_->shape(); 
}

const IntArray& Tensor::strides() const { 
    ensure_defined("strides");
    return impl_->strides(); 
}

Dtype Tensor::dtype() const {
    ensure_defined("dtype");
    return impl_->dtype(); 
}

int64_t Tensor::numel() const {
    ensure_defined("numel");
    return impl_->numel();
}

int Tensor::ndim() const {
    ensure_defined("ndim");
    return impl_->ndim();
}

IntArray Tensor::byte_strides() const {
    ensure_defined("byte_strides");
    IntArray byte_strides;
    byte_strides.reserve(ndim());

    const int64_t itemsize = static_cast<int64_t>(dtype().itemsize());
    const IntArray& step_strides = strides();
    for (const int64_t s : step_strides)
        byte_strides.push_back(s * itemsize);

    return byte_strides;
}

std::string Tensor::to_string() const {
    if (!defined()) 
        return "raxis(undefined tensor!)";

    return fmt::format(
        "raxis(shape={}, strides={}, dtype={})", 
        impl_->shape(), impl_->strides(), impl_->dtype().name()
    );
}

// ========================= Operations =========================
Tensor Tensor::clone() const {
    return ::raxis::clone(*this);
}

Tensor& Tensor::copy_(const Tensor& other) const {
    return ::raxis::copy(other, const_cast<Tensor&>(*this));
}

Tensor Tensor::to(Dtype new_dtype) const {
    return ::raxis::to(*this, new_dtype);
}


Tensor Tensor::add(const Tensor& other) const {
    return ::raxis::add(*this, other);
}

Tensor Tensor::sub(const Tensor& other) const {
    return ::raxis::sub(*this, other);
}

Tensor Tensor::mul(const Tensor& other) const {
    return ::raxis::mul(*this, other);
}

Tensor Tensor::div(const Tensor& other) const {
    return ::raxis::div(*this, other);
}


Tensor& Tensor::add_(const Tensor& other) const {
    return ::raxis::add_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::sub_(const Tensor& other) const {
    return ::raxis::sub_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::mul_(const Tensor& other) const {
    return ::raxis::mul_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::div_(const Tensor& other) const {
    return ::raxis::div_(const_cast<Tensor&>(*this), other);
}


Tensor Tensor::rsub(const Tensor& other) const {
    return ::raxis::rsub(*this, other);
}

Tensor Tensor::rdiv(const Tensor& other) const {
    return ::raxis::rdiv(*this, other);
}


Tensor& Tensor::rsub_(const Tensor& other) const {
    return ::raxis::rsub_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::rdiv_(const Tensor& other) const {
    return ::raxis::rdiv_(const_cast<Tensor&>(*this), other);
}

// ========================= Operators =========================
Tensor Tensor::operator+(const Tensor& other) const {
    return ::raxis::add(*this, other);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return ::raxis::sub(*this, other);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return ::raxis::mul(*this, other);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return ::raxis::div(*this, other);
}

// In-place
Tensor& Tensor::operator+=(const Tensor& other) const {
    return ::raxis::add_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::operator-=(const Tensor& other) const {
    return ::raxis::sub_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::operator*=(const Tensor& other) const {
    return ::raxis::mul_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::operator/=(const Tensor& other) const {
    return ::raxis::div_(const_cast<Tensor&>(*this), other);
}

}