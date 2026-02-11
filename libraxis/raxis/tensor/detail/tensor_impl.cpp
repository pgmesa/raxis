
#include <raxis/tensor/detail/tensor_impl.h>

#include <raxis/tensor/shape_utils.h>


namespace raxis {

TensorImpl::TensorImpl(
    detail::UniqueVoidPtr data,
    const IntArray& shape,
    Dtype dtype,
    const IntArray& strides
): data_(std::move(data)), shape_(shape), dtype_(dtype), strides_(strides), 
    numel_(compute_numel(shape)) {}

TensorImpl::TensorImpl(
    detail::UniqueVoidPtr data,
    const IntArray& shape,
    Dtype dtype
): data_(std::move(data)), shape_(shape), dtype_(dtype), strides_(compute_strides(shape)),
    numel_(compute_numel(shape)) {}


const void* TensorImpl::data() const noexcept {
    return data_.get();
}

void* TensorImpl::mutable_data() noexcept {
    return data_.get();
}

const IntArray& TensorImpl::shape() const noexcept {
    return shape_;
}

const IntArray& TensorImpl::strides() const noexcept {
    return strides_;
}

Dtype TensorImpl::dtype() const noexcept {
    return dtype_;
}

int64_t TensorImpl::numel() const noexcept {
    return numel_;
}

int TensorImpl::ndim() const noexcept {
    return static_cast<int>(shape_.size());
}

} // namespace raxis