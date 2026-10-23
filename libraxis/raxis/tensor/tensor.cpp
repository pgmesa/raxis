
#include <raxis/tensor/tensor.h>

#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <iosfwd>
#include <sstream>

#include <raxis/ops.h>

#include <raxis/autograd/engine.h>
#include <raxis/autograd/autograd_meta.h>
#include <raxis/autograd/variable.h>

#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/detail/storage_impl.h>
#include <raxis/tensor/tensor_options.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/formatting.h>
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

int64_t Tensor::stride(int64_t dim) const { 
    ensure_defined("strides");
    const auto strides = this->strides();
    const auto ndim = static_cast<int64_t>(strides.size());
    return strides[wrap_dim(dim, ndim)];
}

int64_t Tensor::numel() const {
    ensure_defined("numel");
    return impl_->numel();
}

size_t Tensor::nbytes() const {
    return numel() * dtype().itemsize();
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

// requires_grad is deliberately left out (as in PyTorch): every *_like factory
// and op output allocated with options() would otherwise become a leaf that
// requires grad, silently cutting the graph.
TensorOptions Tensor::options() const {
    return TensorOptions()
        .dtype(dtype())
        .device(device());
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
    return impl_->is_contiguous();
}

bool Tensor::is_non_overlapping_and_dense() const {
    ensure_defined("is_non_overlapping_and_dense");
    return impl_->is_non_overlapping_and_dense();
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    return clone(MemoryFormat::Contiguous);
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
    std::ostringstream os;
    print(os, *this);
    return os.str();
}

// ======================== Autograd =========================
// Helper to lazily create the metadata
static autograd::AutogradMeta& ensure_autograd_meta(detail::TensorImpl* impl) {
    if (!impl->autograd_meta()) {
        impl->set_autograd_meta(std::make_unique<autograd::AutogradMeta>());
    }
    return *impl->autograd_meta();
}

// From: torch/csrc/autograd/variable.cpp (VariableHooks::grad_fn)
// The memory of a view was written in place (through its base or another
// view) since its grad_fn was set: rebuild it before reading the autograd
// state. A null check for any tensor that is not a view.
static void refresh_if_stale_view(const Tensor& t) {
    detail::TensorImpl* impl = t.unsafe_get_tensor_impl();
    if (impl->is_view() && impl->view_version() != impl->storage()->version()) {
        autograd::refresh_view_grad_fn(t);
    }
}

bool Tensor::requires_grad() const {
    if (!defined()) return false;
    refresh_if_stale_view(*this);
    auto* meta = impl_->autograd_meta();
    if (meta && meta->requires_grad) return true;
    // From: torch/csrc/autograd/variable.h (DifferentiableViewMeta::requires_grad)
    // A view of a tensor that requires grad does too, even one made under
    // no_grad (which has no grad_fn: gradients through it go nowhere, but an
    // in-place op on it is caught by check_inplace)
    if (impl_->is_view()) {
        auto* base_meta = impl_->view_base()->autograd_meta();
        return base_meta && base_meta->requires_grad;
    }
    return false;
}

Tensor& Tensor::requires_grad_(bool requires_grad) {
    ensure_defined("requires_grad_");
    RAXIS_CHECK(dtype().is_floating(),
        "Only floating point tensors can require gradients");
    RAXIS_CHECK(is_leaf(),
        "You can only change requires_grad flags of leaf variables");
    ensure_autograd_meta(impl_.get()).requires_grad = requires_grad;
    return *this;
}

Tensor Tensor::grad() const {
    if (!defined()) return Tensor{};
    auto* meta = impl_->autograd_meta();
    if (!meta) return Tensor{};
    return meta->grad;
}

void Tensor::set_grad(const Tensor& grad) {
    ensure_defined("set_grad");
    ensure_autograd_meta(impl_.get()).grad = grad;
}

std::shared_ptr<autograd::Node> Tensor::grad_fn() const {
    if (!defined()) return nullptr;
    refresh_if_stale_view(*this);
    auto* meta = impl_->autograd_meta();
    return meta ? meta->grad_fn : nullptr;
}

// A view keeps a grad_fn until its memory is written again
void Tensor::set_grad_fn(std::shared_ptr<autograd::Node> fn) {
    ensure_defined("set_grad_fn");
    ensure_autograd_meta(impl_.get()).grad_fn = std::move(fn);
    if (impl_->is_view()) {
        impl_->set_view_version(impl_->storage()->version());
    }
}

uint32_t Tensor::output_nr() const {
    if (!defined()) return 0;
    refresh_if_stale_view(*this);
    auto* meta = impl_->autograd_meta();
    return meta ? meta->output_nr : 0;
}

void Tensor::set_output_nr(uint32_t nr) {
    ensure_defined("set_output_nr");
    ensure_autograd_meta(impl_.get()).output_nr = nr;
}

void Tensor::set_requires_grad(bool requires_grad) {
    ensure_defined("set_requires_grad");
    ensure_autograd_meta(impl_.get()).requires_grad = requires_grad;
}

bool Tensor::is_leaf() const {
    if (!defined()) return true;
    refresh_if_stale_view(*this);
    auto* meta = impl_->autograd_meta();
    return !meta || meta->is_leaf();
}

bool Tensor::retains_grad() const {
    if (!defined()) return false;
    auto* meta = impl_->autograd_meta();
    if (!meta) return false;
    if (meta->is_leaf() && meta->requires_grad) return true;
    return meta->retains_grad;
}

void Tensor::retain_grad() const {
    RAXIS_CHECK(requires_grad(),
        "can't retain_grad on a tensor that does not require grad");
    if (is_leaf()) return;
    auto& meta = ensure_autograd_meta(impl_.get());
    meta.retains_grad = true;
    if (meta.grad_fn) {
        meta.grad_fn->set_retained_tensor(*this);
    }
}

uint32_t Tensor::version() const {
    ensure_defined("version");
    return impl_->storage()->version();
}

bool Tensor::is_view() const {
    return defined() && impl_->is_view();
}

Tensor Tensor::base() const {
    return is_view() ? Tensor(impl_->view_base()) : Tensor();
}

// From: aten/src/ATen/native/TensorShape.cpp (detach)
// Shares the memory (and so the version counter), but it is not a
// differentiable view and starts with no autograd meta
Tensor Tensor::detach() const {
    ensure_defined("detach");
    return make_view(*this, storage_offset(), shape(), strides(), /*is_differentiable_view=*/false);
}

void Tensor::backward(const Tensor& grad, bool retain_graph) const {
    autograd::run_backward(*this, grad, retain_graph);
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

Tensor Tensor::narrow(int64_t dim, int64_t start, int64_t length) const {
    return ::raxis::narrow(*this, dim, start, length);
}

Tensor Tensor::mT() const { 
    return transpose(-2, -1);
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

Tensor Tensor::flatten(int64_t start_dim, int64_t end_dim) const {
    return ::raxis::flatten(*this, start_dim, end_dim);
}

Tensor Tensor::expand(IntArrayRef shape) const {
    return ::raxis::expand(*this, shape);
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
    return ::raxis::copy_(const_cast<Tensor&>(*this), other);
}

Tensor Tensor::clone(MemoryFormat memory_format) const {
    return ::raxis::clone(*this, memory_format);
}

Tensor Tensor::to(Dtype new_dtype) const {
    if (dtype() == new_dtype) return *this;
    return ::raxis::to_copy(*this, new_dtype);
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

Tensor& Tensor::randint_(int64_t low, int64_t high) const {
    return ::raxis::randint_(const_cast<Tensor&>(*this), low, high);
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

Tensor Tensor::cumsum(int64_t dim, std::optional<Dtype> dtype) const {
    return ::raxis::cumsum(*this, dim, dtype);
}

Tensor Tensor::all(IntArrayRef dim, bool keepdim) const {
    return ::raxis::all(*this, dim, keepdim);
}

Tensor Tensor::any(IntArrayRef dim, bool keepdim) const {
    return ::raxis::any(*this, dim, keepdim);
}

// ====================== Linear Algebra =====================
Tensor Tensor::dot(const Tensor& other) const { 
    return ::raxis::dot(*this, other); 
}

Tensor Tensor::mv(const Tensor& vec) const { 
    return ::raxis::mv(*this, vec); 
}

Tensor Tensor::mm(const Tensor& mat2) const { 
    return ::raxis::mm(*this, mat2); 
}

Tensor Tensor::bmm(const Tensor& mat2) const { 
    return ::raxis::bmm(*this, mat2); 
}

Tensor Tensor::matmul(const Tensor& other) const { 
    return ::raxis::matmul(*this, other); 
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
Tensor Tensor::sign() const { return ::raxis::sign(*this); }
Tensor Tensor::ceil() const { return ::raxis::ceil(*this); }
Tensor Tensor::floor() const { return ::raxis::floor(*this); }
Tensor Tensor::round() const { return ::raxis::round(*this); }
Tensor Tensor::exp() const { return ::raxis::exp(*this); }
Tensor Tensor::log() const { return ::raxis::log(*this); }
Tensor Tensor::sqrt() const { return ::raxis::sqrt(*this); }
Tensor Tensor::rsqrt() const { return ::raxis::rsqrt(*this); }
Tensor Tensor::reciprocal() const { return ::raxis::reciprocal(*this); }
Tensor Tensor::sigmoid() const { return ::raxis::sigmoid(*this); }
Tensor Tensor::relu() const { return ::raxis::relu(*this); }
Tensor Tensor::softmax(int64_t dim) const { return ::raxis::softmax(*this, dim); }
Tensor Tensor::log_softmax(int64_t dim) const { return ::raxis::log_softmax(*this, dim); }
Tensor Tensor::logical_not() const { return ::raxis::logical_not(*this); }
Tensor Tensor::expm1() const { return ::raxis::expm1(*this); }
Tensor Tensor::log2() const { return ::raxis::log2(*this); }
Tensor Tensor::log10() const { return ::raxis::log10(*this); }
Tensor Tensor::log1p() const { return ::raxis::log1p(*this); }
Tensor Tensor::sin() const { return ::raxis::sin(*this); }
Tensor Tensor::cos() const { return ::raxis::cos(*this); }
Tensor Tensor::tan() const { return ::raxis::tan(*this); }
Tensor Tensor::asin() const { return ::raxis::asin(*this); }
Tensor Tensor::acos() const { return ::raxis::acos(*this); }
Tensor Tensor::atan() const { return ::raxis::atan(*this); }
Tensor Tensor::sinh() const { return ::raxis::sinh(*this); }
Tensor Tensor::cosh() const { return ::raxis::cosh(*this); }
Tensor Tensor::tanh() const { return ::raxis::tanh(*this); }
Tensor Tensor::erf() const { return ::raxis::erf(*this); }
Tensor Tensor::erfc() const { return ::raxis::erfc(*this); }

Tensor& Tensor::neg_() const { return ::raxis::neg_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::abs_() const { return ::raxis::abs_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::sign_() const { return ::raxis::sign_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::ceil_() const { return ::raxis::ceil_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::floor_() const { return ::raxis::floor_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::round_() const { return ::raxis::round_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::exp_() const { return ::raxis::exp_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::log_() const { return ::raxis::log_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::sqrt_() const { return ::raxis::sqrt_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::rsqrt_() const { return ::raxis::rsqrt_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::reciprocal_() const { return ::raxis::reciprocal_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::sigmoid_() const { return ::raxis::sigmoid_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::relu_() const { return ::raxis::relu_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::logical_not_() const { return ::raxis::logical_not_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::expm1_() const { return ::raxis::expm1_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::log2_() const { return ::raxis::log2_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::log10_() const { return ::raxis::log10_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::log1p_() const { return ::raxis::log1p_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::sin_() const { return ::raxis::sin_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::cos_() const { return ::raxis::cos_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::tan_() const { return ::raxis::tan_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::asin_() const { return ::raxis::asin_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::acos_() const { return ::raxis::acos_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::atan_() const { return ::raxis::atan_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::sinh_() const { return ::raxis::sinh_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::cosh_() const { return ::raxis::cosh_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::tanh_() const { return ::raxis::tanh_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::erf_() const { return ::raxis::erf_(const_cast<Tensor&>(*this)); }
Tensor& Tensor::erfc_() const { return ::raxis::erfc_(const_cast<Tensor&>(*this)); }

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