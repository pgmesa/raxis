
#include <raxis/tensor/tensor.h>

#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

#include <fmt/format.h>

#include <raxis/format.h>
#include <raxis/ops.h>

#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/detail/storage_impl.h>
#include <raxis/tensor/tensor_options.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_indexing.h>
#include <raxis/tensor/tensor_shape.h>


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
    
IntArrayRef Tensor::shape() const { 
    ensure_defined("shape");
    return impl_->shape(); 
}

IntArrayRef Tensor::strides() const { 
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

DimVector Tensor::byte_strides() const {
    ensure_defined("byte_strides");
    DimVector byte_strides;
    byte_strides.reserve(ndim());

    const int64_t itemsize = static_cast<int64_t>(dtype().itemsize());
    IntArrayRef step_strides = strides();
    for (const int64_t s : step_strides)
        byte_strides.push_back(s * itemsize);

    return byte_strides;
}

int64_t Tensor::storage_offset() const {
    ensure_defined("storage_offset");
    return impl_->storage_offset();
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

Scalar Tensor::item() const {
    RAXIS_CHECK(numel() == 1, "a Tensor with ", numel(), " elements cannot be converted to Scalar");
    return dtype().dispatch_all_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) -> Scalar {
            return Scalar(data_ptr<scalar_t>()[0]);
        }
    );
}

bool Tensor::is_contiguous() const {
    ensure_defined("is_contiguous");
    IntArrayRef sh = shape();
    IntArrayRef st = strides();
    if (sh.empty()) return true;
    
    int64_t expected_stride = 1;
    for (int i = static_cast<int>(sh.size()) - 1; i >= 0; --i) {
        if (sh[i] == 1) continue;
        if (st[i] != expected_stride) return false;
        expected_stride *= sh[i];
    }
    return true;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    return clone();
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

bool Tensor::is_wrapped_number() const {
    return impl_->is_wrapped_number();
}

std::string Tensor::to_string() const {
    if (!defined()) 
        return "raxis(undefined tensor!)";

    return fmt::format(
        "raxis(shape={}, strides={}, dtype={}, device='{}')", 
        impl_->shape(), impl_->strides(), impl_->dtype().name(), impl_->device().str()
    );
}

// ===================== View operations ======================
Tensor Tensor::alias() const {
    return ::raxis::alias(*this);
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    return ::raxis::unsqueeze(*this, dim);
}

Tensor Tensor::squeeze(std::optional<int64_t> dim_opt) const {
    return ::raxis::squeeze(*this, dim_opt);
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
    return ::raxis::select(*this, dim, index);
}

Tensor Tensor::slice(int64_t dim, const indexing::Slice& slice) const {
    return ::raxis::slice(*this, dim, slice);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t stop, int64_t step) const {
    return ::raxis::slice(*this, dim, start, stop, step);
}


Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    return ::raxis::transpose(*this, dim0, dim1);
}

Tensor Tensor::permute(IntArrayRef dims) const {
    return ::raxis::permute(*this, dims);
}

Tensor Tensor::swapdims(int64_t dim0, int64_t dim1) const {
    return ::raxis::swapdims(*this, dim0, dim1);
}

Tensor Tensor::movedim(IntArrayRef src, IntArrayRef dst) const {
    return ::raxis::movedim(*this, src, dst);
}

Tensor Tensor::movedim(int64_t src, int64_t dst) const {
    return ::raxis::movedim(*this, src, dst);
}

Tensor Tensor::view(IntArrayRef shape) const {
    return ::raxis::view(*this, shape);
}

Tensor Tensor::reshape(IntArrayRef shape) const {
    return ::raxis::reshape(*this, shape);
}

Tensor Tensor::broadcast_to(IntArrayRef target_shape) const {
    return ::raxis::broadcast_to(*this, target_shape);
}

// ========================= Indexing =========================
Tensor Tensor::index(ArrayRef<indexing::TensorIndex> indices) const {
    return indexing::get_item(*this, indices);
}

void Tensor::index_put_(ArrayRef<indexing::TensorIndex> indices, const Tensor& value) {
    indexing::set_item(*this, indices, value);
}

void Tensor::index_put_(ArrayRef<indexing::TensorIndex> indices, Scalar value) {
    indexing::set_item(*this, indices, value);
}

// ======================== Copy & Fill =======================
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
        device(), " -> ", new_device);
    return *this;
}

Tensor Tensor::cpu() const {
    return *this;
}

Tensor& Tensor::fill_(Scalar value) const {
    return ::raxis::fill_(const_cast<Tensor&>(*this), value);
}

Tensor& Tensor::zero_() const {
    return ::raxis::zero_(const_cast<Tensor&>(*this));
}

Tensor& Tensor::uniform_(double low, double high) const {
    return ::raxis::uniform_(const_cast<Tensor&>(*this), low, high);
}

Tensor& Tensor::normal_(double mean, double std) const {
    return ::raxis::normal_(const_cast<Tensor&>(*this), mean, std);
}

// =================== Reduction Operations ===================
Tensor Tensor::sum(std::optional<Dtype> dtype) const {
    return ::raxis::sum(*this, dtype);
}

Tensor Tensor::sum(
    IntArrayRef dim, 
    bool keepdim, 
    std::optional<Dtype> dtype
) const {
    return ::raxis::sum(*this, dim, keepdim, dtype);
}

Tensor Tensor::nansum(std::optional<Dtype> dtype) const {
    return ::raxis::nansum(*this, dtype);
}

Tensor Tensor::nansum(
    IntArrayRef dim, 
    bool keepdim, 
    std::optional<Dtype> dtype
) const {
    return ::raxis::nansum(*this, dim, keepdim, dtype);
}

Tensor Tensor::mean(std::optional<Dtype> dtype) const {
    return ::raxis::mean(*this, dtype);
}

Tensor Tensor::mean(
    IntArrayRef dim, 
    bool keepdim, 
    std::optional<Dtype> dtype
) const {
    return ::raxis::mean(*this, dim, keepdim, dtype);
}

Tensor Tensor::nanmean(std::optional<Dtype> dtype) const {
    return ::raxis::nanmean(*this, dtype);
}

Tensor Tensor::nanmean(
    IntArrayRef dim, 
    bool keepdim, 
    std::optional<Dtype> dtype
) const {
    return ::raxis::nanmean(*this, dim, keepdim, dtype);
}

Tensor Tensor::prod(std::optional<Dtype> dtype) const {
    return ::raxis::prod(*this, dtype);
}

Tensor Tensor::prod(
    IntArrayRef dim, 
    bool keepdim, 
    std::optional<Dtype> dtype
) const {
    return ::raxis::prod(*this, dim, keepdim, dtype);
}


Tensor Tensor::max() const {
    return ::raxis::max(*this);
}

Tensor Tensor::min() const {
    return ::raxis::min(*this);
}

Tensor Tensor::amax(IntArrayRef dim, bool keepdim) const {
    return ::raxis::amax(*this, dim, keepdim);
}

Tensor Tensor::amin(IntArrayRef dim, bool keepdim) const {
    return ::raxis::amin(*this, dim, keepdim);
}

// ===================== Binary Operations ====================
Tensor Tensor::add(const Tensor& other) const { return ::raxis::add(*this, other); }
Tensor Tensor::sub(const Tensor& other) const { return ::raxis::sub(*this, other); }
Tensor Tensor::mul(const Tensor& other) const { return ::raxis::mul(*this, other); }
Tensor Tensor::div(const Tensor& other) const { return ::raxis::div(*this, other); }

Tensor Tensor::add(Scalar s) const { return ::raxis::add(*this, s); }
Tensor Tensor::sub(Scalar s) const { return ::raxis::sub(*this, s); }
Tensor Tensor::mul(Scalar s) const { return ::raxis::mul(*this, s); }
Tensor Tensor::div(Scalar s) const { return ::raxis::div(*this, s); }


Tensor& Tensor::add_(const Tensor& other) const { return ::raxis::add_(const_cast<Tensor&>(*this), other); }
Tensor& Tensor::sub_(const Tensor& other) const { return ::raxis::sub_(const_cast<Tensor&>(*this), other); }
Tensor& Tensor::mul_(const Tensor& other) const { return ::raxis::mul_(const_cast<Tensor&>(*this), other); }
Tensor& Tensor::div_(const Tensor& other) const { return ::raxis::div_(const_cast<Tensor&>(*this), other); }

Tensor& Tensor::add_(Scalar s) const { return ::raxis::add_(const_cast<Tensor&>(*this), s); }
Tensor& Tensor::sub_(Scalar s) const { return ::raxis::sub_(const_cast<Tensor&>(*this), s); }
Tensor& Tensor::mul_(Scalar s) const { return ::raxis::mul_(const_cast<Tensor&>(*this), s); }
Tensor& Tensor::div_(Scalar s) const { return ::raxis::div_(const_cast<Tensor&>(*this), s); }


Tensor Tensor::rsub(const Tensor& other) const { return ::raxis::rsub(*this, other); }
Tensor Tensor::rdiv(const Tensor& other) const { return ::raxis::rdiv(*this, other); }

Tensor Tensor::rsub(Scalar s) const { return ::raxis::rsub(*this, s); }
Tensor Tensor::rdiv(Scalar s) const { return ::raxis::rdiv(*this, s); }


Tensor& Tensor::rsub_(const Tensor& other) const { return ::raxis::rsub_(const_cast<Tensor&>(*this), other); }
Tensor& Tensor::rdiv_(const Tensor& other) const { return ::raxis::rdiv_(const_cast<Tensor&>(*this), other); }

Tensor& Tensor::rsub_(Scalar s) const { return ::raxis::rsub_(const_cast<Tensor&>(*this), s); }
Tensor& Tensor::rdiv_(Scalar s) const { return ::raxis::rdiv_(const_cast<Tensor&>(*this), s); }


Tensor Tensor::maximum(const Tensor& other) const { return ::raxis::maximum(*this, other); }
Tensor Tensor::minimum(const Tensor& other) const { return ::raxis::minimum(*this, other); }
Tensor Tensor::pow(const Tensor& exponent) const  { return ::raxis::pow(*this, exponent); }
Tensor Tensor::pow(Scalar exponent) const         { return ::raxis::pow(*this, exponent); }

// =================== Comparison Operations ===================
Tensor Tensor::gt(const Tensor& other) const { return ::raxis::gt(*this, other); }
Tensor Tensor::ge(const Tensor& other) const { return ::raxis::ge(*this, other); }
Tensor Tensor::lt(const Tensor& other) const { return ::raxis::lt(*this, other); }
Tensor Tensor::le(const Tensor& other) const { return ::raxis::le(*this, other); }
Tensor Tensor::eq(const Tensor& other) const { return ::raxis::eq(*this, other); }
Tensor Tensor::ne(const Tensor& other) const { return ::raxis::ne(*this, other); }

Tensor Tensor::gt(Scalar s) const { return ::raxis::gt(*this, s); }
Tensor Tensor::ge(Scalar s) const { return ::raxis::ge(*this, s); }
Tensor Tensor::lt(Scalar s) const { return ::raxis::lt(*this, s); }
Tensor Tensor::le(Scalar s) const { return ::raxis::le(*this, s); }
Tensor Tensor::eq(Scalar s) const { return ::raxis::eq(*this, s); }
Tensor Tensor::ne(Scalar s) const { return ::raxis::ne(*this, s); }

Tensor Tensor::isnan() const { return ::raxis::isnan(*this); }
 
// ==================== Logical Operations =====================
Tensor Tensor::logical_and(const Tensor& other) const { return ::raxis::logical_and(*this, other); }
Tensor Tensor::logical_or (const Tensor& other) const { return ::raxis::logical_or(*this, other); }

// ====================== Unary Operations =====================
Tensor Tensor::neg() const { return ::raxis::neg(*this); }
Tensor Tensor::abs() const { return ::raxis::abs(*this); }
Tensor Tensor::ceil() const { return ::raxis::ceil(*this); }
Tensor Tensor::floor() const { return ::raxis::floor(*this); }
Tensor Tensor::round() const { return ::raxis::round(*this); }
Tensor Tensor::exp() const { return ::raxis::exp(*this); }
Tensor Tensor::log() const { return ::raxis::log(*this); }
Tensor Tensor::sqrt() const { return ::raxis::sqrt(*this); }
Tensor Tensor::sigmoid() const { return ::raxis::sigmoid(*this); }
Tensor Tensor::logical_not() const { return ::raxis::logical_not(*this); }

Tensor& Tensor::neg_() const { return ::raxis::neg_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::abs_() const { return ::raxis::abs_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::ceil_() const { return ::raxis::ceil_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::floor_() const { return ::raxis::floor_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::round_() const { return ::raxis::round_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::exp_() const { return ::raxis::exp_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::log_() const { return ::raxis::log_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::sqrt_() const { return ::raxis::sqrt_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::sigmoid_() const { return ::raxis::sigmoid_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::logical_not_() const { return ::raxis::logical_not_(const_cast<Tensor&>(*this)); }

// ========================= Operators =========================
Tensor Tensor::operator[](int64_t index) const {
    return select(0, index);
}

Tensor Tensor::operator[](indexing::TensorIndex idx) const {
    return index({idx});
}

Tensor Tensor::operator+(const Tensor& other) const { return ::raxis::add(*this, other); }
Tensor Tensor::operator-(const Tensor& other) const { return ::raxis::sub(*this, other); }
Tensor Tensor::operator*(const Tensor& other) const { return ::raxis::mul(*this, other); }
Tensor Tensor::operator/(const Tensor& other) const { return ::raxis::div(*this, other); }

Tensor Tensor::operator+(Scalar s) const { return ::raxis::add(*this, s); }
Tensor Tensor::operator-(Scalar s) const { return ::raxis::sub(*this, s); }
Tensor Tensor::operator*(Scalar s) const { return ::raxis::mul(*this, s); }
Tensor Tensor::operator/(Scalar s) const { return ::raxis::div(*this, s); }

// In-place
Tensor& Tensor::operator+=(const Tensor& other) const { return ::raxis::add_(const_cast<Tensor&>(*this), other); }
Tensor& Tensor::operator-=(const Tensor& other) const { return ::raxis::sub_(const_cast<Tensor&>(*this), other); }
Tensor& Tensor::operator*=(const Tensor& other) const { return ::raxis::mul_(const_cast<Tensor&>(*this), other); }
Tensor& Tensor::operator/=(const Tensor& other) const { return ::raxis::div_(const_cast<Tensor&>(*this), other); }

Tensor& Tensor::operator+=(Scalar s) const { return ::raxis::add_(const_cast<Tensor&>(*this), s); }
Tensor& Tensor::operator-=(Scalar s) const { return ::raxis::sub_(const_cast<Tensor&>(*this), s); }
Tensor& Tensor::operator*=(Scalar s) const { return ::raxis::mul_(const_cast<Tensor&>(*this), s); }
Tensor& Tensor::operator/=(Scalar s) const { return ::raxis::div_(const_cast<Tensor&>(*this), s); }

// Unary
Tensor Tensor::operator-() const { return ::raxis::neg(*this); }
Tensor Tensor::operator~() const { return ::raxis::logical_not(*this); }

// Comparison
Tensor Tensor::operator> (const Tensor& other) const { return ::raxis::gt(*this, other); }
Tensor Tensor::operator>=(const Tensor& other) const { return ::raxis::ge(*this, other); }
Tensor Tensor::operator< (const Tensor& other) const { return ::raxis::lt(*this, other); }
Tensor Tensor::operator<=(const Tensor& other) const { return ::raxis::le(*this, other); }
Tensor Tensor::operator==(const Tensor& other) const { return ::raxis::eq(*this, other); }
Tensor Tensor::operator!=(const Tensor& other) const { return ::raxis::ne(*this, other); }

Tensor Tensor::operator> (Scalar s) const { return ::raxis::gt(*this, s); }
Tensor Tensor::operator>=(Scalar s) const { return ::raxis::ge(*this, s); }
Tensor Tensor::operator< (Scalar s) const { return ::raxis::lt(*this, s); }
Tensor Tensor::operator<=(Scalar s) const { return ::raxis::le(*this, s); }
Tensor Tensor::operator==(Scalar s) const { return ::raxis::eq(*this, s); }
Tensor Tensor::operator!=(Scalar s) const { return ::raxis::ne(*this, s); }

} // namespace raxis