
#include <raxis/tensor/detail/tensor_impl.h>

#include <optional>

#include <raxis/core/dtype.h>
#include <raxis/core/device.h>

#include <raxis/util/exception.h>
#include <raxis/util/data_ptr.h>

#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/detail/storage_impl.h>


namespace raxis::detail {

TensorImpl::TensorImpl(
    DataPtr data,
    size_t nbytes,
    const IntArray& shape,
    std::optional<IntArray> strides,
    Dtype dtype
) : storage_(std::make_shared<StorageImpl>(std::move(data), nbytes)),
    storage_offset_(0),
    shape_(shape),
    strides_(strides.value_or(compute_strides(shape))),
    dtype_(dtype),
    numel_(compute_numel(shape)) {}

TensorImpl::TensorImpl(
    std::shared_ptr<StorageImpl> storage,
    int64_t storage_offset,
    const IntArray& shape,
    const IntArray& strides,
    Dtype dtype
) : storage_(std::move(storage)),
    storage_offset_(storage_offset),
    shape_(shape),
    strides_(strides),
    dtype_(dtype),
    numel_(compute_numel(shape)) {}

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

const IntArray& TensorImpl::shape() const {
    return shape_;
}

const IntArray& TensorImpl::strides() const {
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

const std::shared_ptr<StorageImpl>& TensorImpl::storage() const {
    return storage_;
}

} // namespace raxis