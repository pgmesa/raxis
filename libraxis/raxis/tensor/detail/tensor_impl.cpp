
#include <raxis/tensor/detail/tensor_impl.h>

#include <optional>

#include <raxis/core/dtype.h>
#include <raxis/core/device.h>

#include <raxis/util/exception.h>
#include <raxis/util/data_ptr.h>

#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/detail/storage_impl.h>

#include <raxis/autograd/autograd_meta.h>


namespace raxis::detail {

// This must be here (not in header) because AutogradMeta is incomplete there
TensorImpl::~TensorImpl() = default;

TensorImpl::TensorImpl(
    DataPtr data,
    size_t nbytes,
    IntArrayRef shape,
    std::optional<IntArrayRef> strides,
    Dtype dtype,
    Allocator* allocator
) : storage_(std::make_shared<StorageImpl>(std::move(data), nbytes, allocator)),
    storage_offset_(0),
    shape_(shape),
    strides_(strides.value_or(compute_strides(shape))),
    dtype_(dtype),
    numel_(compute_numel(shape)) {
    compute_layout();
}

TensorImpl::TensorImpl(
    std::shared_ptr<StorageImpl> storage,
    int64_t storage_offset,
    IntArrayRef shape,
    IntArrayRef strides,
    Dtype dtype
) : storage_(std::move(storage)),
    storage_offset_(storage_offset),
    shape_(shape),
    strides_(strides),
    dtype_(dtype),
    numel_(compute_numel(shape)) {
    compute_layout();
}

const void* TensorImpl::data() const {
    const auto* base = static_cast<const char*>(storage_->data());
    return base + storage_offset_ * static_cast<int64_t>(dtype_.itemsize());
}

void* TensorImpl::mutable_data() {
    auto* base = static_cast<char*>(storage_->mutable_data());
    return base + storage_offset_ * static_cast<int64_t>(dtype_.itemsize());
}

void TensorImpl::set_wrapped_number(bool is_wrapped) {
    RAXIS_INTERNAL_ASSERT(ndim() == 0); 
    is_wrapped_number_ = is_wrapped;
}

bool TensorImpl::is_wrapped_number() const {
    return is_wrapped_number_;
}

IntArrayRef TensorImpl::shape() const {
    return shape_;
}

IntArrayRef TensorImpl::strides() const {
    return strides_;
}

Dtype TensorImpl::dtype() const {
    return dtype_;
}

Device TensorImpl::device() const {
    return storage_->device();
}

int64_t TensorImpl::numel() const {
    return numel_;
}

int TensorImpl::ndim() const {
    return static_cast<int>(shape_.size());
}

int64_t TensorImpl::storage_offset() const {
    return storage_offset_;
}

bool TensorImpl::is_contiguous() const {
    return is_contiguous_;
}

bool TensorImpl::is_non_overlapping_and_dense() const {
    return is_non_overlapping_and_dense_;
}

void TensorImpl::shallow_copy_from(const TensorImpl& src) {
    storage_ = src.storage_;
    storage_offset_ = src.storage_offset_;
    shape_ = src.shape_;
    strides_ = src.strides_;
    dtype_ = src.dtype_;
    numel_ = src.numel_;
    is_wrapped_number_ = src.is_wrapped_number_;
    compute_layout();
}

void TensorImpl::set_shape_and_strides(IntArrayRef shape, IntArrayRef strides) {
    shape_ = DimVector(shape);
    strides_ = DimVector(strides);
    numel_ = compute_numel(shape);
    compute_layout();
}

void TensorImpl::compute_layout() {
    is_contiguous_ = compute_contiguous(shape_, strides_);
    is_non_overlapping_and_dense_ =
        is_contiguous_ || compute_non_overlapping_and_dense(shape_, strides_);
}

const std::shared_ptr<StorageImpl>& TensorImpl::storage() const {
    return storage_;
}

autograd::AutogradMeta* TensorImpl::autograd_meta() const { 
    return autograd_meta_.get(); 
}

void TensorImpl::set_autograd_meta(std::unique_ptr<autograd::AutogradMeta> meta) {
    autograd_meta_ = std::move(meta);
}

void TensorImpl::set_view_base(std::shared_ptr<TensorImpl> base, autograd::CreationMeta creation_meta) {
    RAXIS_INTERNAL_ASSERT(base && !base->is_view(), "the base of a view cannot be a view");
    view_base_ = std::move(base);
    creation_meta_ = creation_meta;
    view_version_ = storage_->version();
}

} // namespace raxis