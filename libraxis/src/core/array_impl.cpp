
#include <raxis/core/array_impl.hpp>

#include <raxis/core/shape.hpp>


namespace raxis {

ArrayImpl::ArrayImpl(
    detail::UniqueVoidPtr data,
    const IntArray& shape,
    Dtype dtype,
    const IntArray& strides
): data_(std::move(data)), shape_(shape), dtype_(dtype), strides_(strides), 
    numel_(compute_numel(shape)) {}

ArrayImpl::ArrayImpl(
    detail::UniqueVoidPtr data,
    const IntArray& shape,
    Dtype dtype
): data_(std::move(data)), shape_(shape), dtype_(dtype), strides_(compute_strides(shape)),
    numel_(compute_numel(shape)) {}


const void* ArrayImpl::data() const noexcept {
    return data_.get();
}

void* ArrayImpl::mutable_data() noexcept {
    return data_.get();
}

const IntArray& ArrayImpl::shape() const noexcept {
    return shape_;
}

const IntArray& ArrayImpl::strides() const noexcept {
    return strides_;
}

Dtype ArrayImpl::dtype() const noexcept {
    return dtype_;
}

int64_t ArrayImpl::numel() const noexcept {
    return numel_;
}

int ArrayImpl::ndim() const noexcept {
    return static_cast<int>(shape_.size());
}

} // namespace raxis