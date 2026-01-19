
#include <raxis/array.hpp>

#include <vector>
#include <memory>
#include <format>

#include <raxis/array_ops.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/core/array_impl.hpp>


namespace raxis {

    const void* NDArray::data_ptr() const {
        return impl_->data();
    }

    void* NDArray::mutable_data_ptr() {
        return impl_->mutable_data();
    }
        
    const IntArray& NDArray::shape() const { return impl_->shape(); }
    const IntArray& NDArray::strides() const { return impl_->strides(); }
    Dtype NDArray::dtype() const { return impl_->dtype(); }
    size_t NDArray::numel() const { return impl_->numel(); }
    size_t NDArray::ndim() const { return impl_->ndim(); }

    std::string NDArray::to_string() const {
        return std::format(
            "ndarray(shape={}, strides={}, dtype={})", 
            impl_->shape(), impl_->strides(), impl_->dtype().name()
        );
    }

    // Operators
    NDArray NDArray::operator+(const NDArray& other) const {
        return add(*this, other);
    }

}