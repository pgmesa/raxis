
#include <raxis/array.hpp>

#include <vector>
#include <memory>

#include <fmt/format.h>
#include <raxis/array_ops.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/core/array_impl.hpp>
#include <raxis/format/array_fmt.hpp>
#include <raxis/factories.hpp>


namespace raxis {

bool NDArray::defined() const noexcept {
    return impl_ != nullptr;
}

bool NDArray::is_same(const NDArray& other) const noexcept {
    return impl_ == other.impl_;
}

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

NDArray NDArray::to(Dtype new_dtype) const {
    if (!defined()) {
        throw std::runtime_error("Cannot convert undefined array");
    }

    Dtype curr_dtype = impl_->dtype(); 
    
    if (curr_dtype == new_dtype) {
        return *this;
    }
    
    // Allocate new array with target dtype
    NDArray result = empty(impl_->shape(), new_dtype);
    
    // Dispatch on source and dest types to do conversion
    curr_dtype.dispatch_all_types([&]<typename src_T>(std::type_identity<src_T>) {
        const src_T* src_ptr = data_ptr<src_T>();
        
        new_dtype.dispatch_all_types([&]<typename dest_T>(std::type_identity<dest_T>) {
            dest_T* dst_ptr = result.mutable_data_ptr<dest_T>();
            
            // Simple element-wise conversion
            for (size_t i = 0; i < numel(); ++i) {
                dst_ptr[i] = static_cast<dest_T>(src_ptr[i]);
            }
        });
    });
    
    return result;
}

// Operators
NDArray NDArray::operator+(const NDArray& other) const {
    return add(*this, other);
}

NDArray NDArray::operator-(const NDArray& other) const {
    return sub(*this, other);
}

NDArray NDArray::operator*(const NDArray& other) const {
    return mul(*this, other);
}

NDArray NDArray::operator/(const NDArray& other) const {
    return div(*this, other);
}

}