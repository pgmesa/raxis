
#include <raxis/tensor/tensor.h>

#include <vector>
#include <memory>
#include <stdexcept>

#include <fmt/format.h>

#include <raxis/format.h>
#include <raxis/ops/binary_ops.h>
#include <raxis/ops/copy.h>
#include <raxis/ops/fill.h>

#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/tensor_options.h>
#include <raxis/tensor/factories.h>


namespace raxis {

bool Tensor::defined() const noexcept {
    return impl_ != nullptr;
}

bool Tensor::is_same(const Tensor& other) const noexcept {
    return impl_ == other.impl_;
}

void Tensor::ensure_defined(const char* context) const {
    RAXIS_CHECK(defined(), "Tensor is not defined (accessing '", context , "')");
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

Dtype Tensor::dtype() const {
    ensure_defined("dtype");
    return impl_->dtype(); 
}

Device Tensor::device() const {
    ensure_defined("device");
    return impl_->device();
}

bool Tensor::requires_grad() const {
    return false; // TODO
}

TensorOptions Tensor::options() const {
    return TensorOptions()
        .dtype(dtype())
        .device(device())
        .requires_grad(requires_grad());
}

bool Tensor::is_cpu() const {
    return device().is_cpu();
}

bool Tensor::is_cuda() const {
    return device().is_cuda();
}

detail::TensorImpl* Tensor::unsafe_get_tensor_impl() const {
    return impl_.get();
}

std::string Tensor::to_string() const {
    if (!defined()) 
        return "raxis(undefined tensor!)";

    return fmt::format(
        "raxis(shape={}, strides={}, dtype={}, device='{}')", 
        impl_->shape(), impl_->strides(), impl_->dtype().name(), impl_->device().str()
    );
}

// ========================= Operations =========================
Tensor& Tensor::copy_(const Tensor& other) const {
    return ::raxis::copy(other, const_cast<Tensor&>(*this));
}

Tensor Tensor::clone() const {
    Tensor out = empty_like(*this);
    out.copy_(*this);
    return out;
}

Tensor Tensor::to(Dtype new_dtype) const {
    if (dtype() == new_dtype) return *this;

    Tensor out = empty_like(*this, new_dtype);
    out.copy_(*this);
    return out;
}

Tensor Tensor::to(Device new_device) const {
    if (device() == new_device) return *this;

    RAXIS_CHECK(false, 
        "Cross-device transfer not yet implemented: ",
        device(), " → ", new_device);
    return *this;
}


Tensor& Tensor::fill_(Scalar value) const {
    return ::raxis::fill_(const_cast<Tensor&>(*this), value);
}

Tensor& Tensor::zero_() const {
    return ::raxis::zero_(const_cast<Tensor&>(*this));
}


Tensor Tensor::add(const Tensor& other) const {
    return ::raxis::add(*this, other);
}

Tensor Tensor::add(Scalar s) const {
    return ::raxis::add(*this, s);
}

Tensor Tensor::sub(const Tensor& other) const {
    return ::raxis::sub(*this, other);
}

Tensor Tensor::sub(Scalar s) const {
    return ::raxis::sub(*this, s);
}

Tensor Tensor::mul(const Tensor& other) const {
    return ::raxis::mul(*this, other);
}

Tensor Tensor::mul(Scalar s) const {
    return ::raxis::mul(*this, s);
}

Tensor Tensor::div(const Tensor& other) const {
    return ::raxis::div(*this, other);
}

Tensor Tensor::div(Scalar s) const {
    return ::raxis::div(*this, s);
}


Tensor& Tensor::add_(const Tensor& other) const {
    return ::raxis::add_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::add_(Scalar s) const {
    return ::raxis::add_(const_cast<Tensor&>(*this), s);
}

Tensor& Tensor::sub_(const Tensor& other) const {
    return ::raxis::sub_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::sub_(Scalar s) const {
    return ::raxis::sub_(const_cast<Tensor&>(*this), s);
}

Tensor& Tensor::mul_(const Tensor& other) const {
    return ::raxis::mul_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::mul_(Scalar s) const {
    return ::raxis::mul_(const_cast<Tensor&>(*this), s);
}

Tensor& Tensor::div_(const Tensor& other) const {
    return ::raxis::div_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::div_(Scalar s) const {
    return ::raxis::div_(const_cast<Tensor&>(*this), s);
}


Tensor Tensor::rsub(const Tensor& other) const {
    return ::raxis::rsub(*this, other);
}

Tensor Tensor::rsub(Scalar s) const {
    return ::raxis::rsub(*this, s);
}

Tensor Tensor::rdiv(const Tensor& other) const {
    return ::raxis::rdiv(*this, other);
}

Tensor Tensor::rdiv(Scalar s) const {
    return ::raxis::rdiv(*this, s);
}


Tensor& Tensor::rsub_(const Tensor& other) const {
    return ::raxis::rsub_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::rsub_(Scalar s) const {
    return ::raxis::rsub_(const_cast<Tensor&>(*this), s);
}

Tensor& Tensor::rdiv_(const Tensor& other) const {
    return ::raxis::rdiv_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::rdiv_(Scalar s) const {
    return ::raxis::rdiv_(const_cast<Tensor&>(*this), s);
}

// ========================= Operators =========================
Tensor Tensor::operator+(const Tensor& other) const {
    return ::raxis::add(*this, other);
}

Tensor Tensor::operator+(Scalar s) const {
    return ::raxis::add(*this, s);
}

Tensor Tensor::operator-(const Tensor& other) const {
    return ::raxis::sub(*this, other);
}

Tensor Tensor::operator-(Scalar s) const {
    return ::raxis::sub(*this, s);
}

Tensor Tensor::operator*(const Tensor& other) const {
    return ::raxis::mul(*this, other);
}

Tensor Tensor::operator*(Scalar s) const {
    return ::raxis::mul(*this, s);
}

Tensor Tensor::operator/(const Tensor& other) const {
    return ::raxis::div(*this, other);
}

Tensor Tensor::operator/(Scalar s) const {
    return ::raxis::div(*this, s);
}

// In-place
Tensor& Tensor::operator+=(const Tensor& other) const {
    return ::raxis::add_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::operator+=(Scalar s) const {
    return ::raxis::add_(const_cast<Tensor&>(*this), s);
}

Tensor& Tensor::operator-=(const Tensor& other) const {
    return ::raxis::sub_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::operator-=(Scalar s) const {
    return ::raxis::sub_(const_cast<Tensor&>(*this), s);
}

Tensor& Tensor::operator*=(const Tensor& other) const {
    return ::raxis::mul_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::operator*=(Scalar s) const {
    return ::raxis::mul_(const_cast<Tensor&>(*this), s);
}

Tensor& Tensor::operator/=(const Tensor& other) const {
    return ::raxis::div_(const_cast<Tensor&>(*this), other);
}

Tensor& Tensor::operator/=(Scalar s) const {
    return ::raxis::div_(const_cast<Tensor&>(*this), s);
}

} // namespace raxis