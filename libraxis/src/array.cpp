
#include <raxis/array.hpp>

#include <vector>
#include <memory>

#include <fmt/format.h>
#include <raxis/array_ops.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/core/array_impl.hpp>
#include <raxis/format/array_fmt.hpp>


namespace raxis {

const void* NDArray::data_ptr() const {
    return impl_->data();
}

void* NDArray::mutable_data_ptr() const {
    return impl_->mutable_data();
}
    
const IntArray& NDArray::shape() const { return impl_->shape(); }
const IntArray& NDArray::strides() const { return impl_->strides(); }
Dtype NDArray::dtype() const { return impl_->dtype(); }
int64_t NDArray::numel() const { return impl_->numel(); }
int NDArray::ndim() const { return impl_->ndim(); }

std::string NDArray::to_string() const {
    return fmt::format(
        "ndarray(shape={}, strides={}, dtype={})", 
        impl_->shape(), impl_->strides(), impl_->dtype().name()
    );
}

// Operators
NDArray NDArray::operator+(const NDArray& other) const {
    return add(*this, other);
}

}