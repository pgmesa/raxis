
#include <raxis/tensor/detail/tensor_impl.h>

#include <raxis/core/dtype.h>
#include <raxis/core/device.h>

#include <raxis/util/exception.h>
#include <raxis/util/data_ptr.h>

#include <raxis/tensor/shape_utils.h>


namespace raxis::detail {

TensorImpl::TensorImpl(
    DataPtr data,
    const IntArray& shape,
    Dtype dtype,
    const IntArray& strides
): data_ptr_(std::move(data)), shape_(shape), dtype_(dtype), strides_(strides), 
    numel_(compute_numel(shape)) {}

TensorImpl::TensorImpl(
    DataPtr data,
    const IntArray& shape,
    Dtype dtype
): data_ptr_(std::move(data)), shape_(shape), dtype_(dtype), strides_(compute_strides(shape)),
    numel_(compute_numel(shape)) {}


const void* TensorImpl::data() const {
    return data_ptr_.get();
}

void* TensorImpl::mutable_data() {
    return data_ptr_.mutable_get();
}

void TensorImpl::set_wrapped_number(bool is_wrapped) {
    RAXIS_INTERNAL_ASSERT(ndim() == 0); 
    is_wrapped_number_ = is_wrapped;
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
    return data_ptr_.device();
}

int64_t TensorImpl::numel() const {
    return numel_;
}

int TensorImpl::ndim() const {
    return static_cast<int>(shape_.size());
}

} // namespace raxis