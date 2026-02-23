
#include <raxis/tensor/factories.h>

#include <memory>

#include <raxis/core/types.h>
#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>
#include <raxis/core/device.h>
#include <raxis/core/allocator.h>

#include <raxis/util/data_ptr.h>
#include <raxis/util/unique_void_ptr.h>

#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_options.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/detail/tensor_impl.h>


namespace raxis {

Tensor from_blob(void* data, const IntArray& shape, Dtype dtype, Device device) {
    // Non-owning: Caller guarantees lifetime.
    DataPtr ptr (data, device);
    auto impl = std::make_shared<detail::TensorImpl>(std::move(ptr), shape, dtype);
    return Tensor(std::move(impl));
}

Tensor from_blob(
    void* data,
    DeleterFnPtr deleter,
    const IntArray& shape,
    Dtype dtype,
    Device device
) {
    DataPtr ptr(data, data, deleter, device);
    auto impl = std::make_shared<detail::TensorImpl>(std::move(ptr), shape, dtype);
    return Tensor(std::move(impl));
}

Tensor empty(const IntArray& shape, TensorOptions options) {
    Dtype dtype = options.dtype();
    size_t n = static_cast<size_t>(compute_numel(shape));
    auto allocator = get_allocator(options.device().type());
    DataPtr data_ptr = allocator->allocate(n * dtype.itemsize());
    auto impl = std::make_shared<detail::TensorImpl>(std::move(data_ptr), shape, dtype);
    return Tensor(std::move(impl));
}

Tensor empty_like(const Tensor& other, TensorOptions options) {
    return empty(other.shape(), other.options().merge_in(options));
}

Tensor empty_strided(const IntArray& shape, const IntArray& strides, TensorOptions options) {
    // Validate shapes and strides match
    if (shape.size() != strides.size()) {
        throw std::invalid_argument(
            "Shape and strides must have the same number of dimensions"
        );
    }
    
    // Compute required memory based on shape and strides
    Dtype dtype = options.dtype();
    size_t n = compute_numel_from_strides(shape, strides);
    auto allocator = get_allocator(options.device().type());
    DataPtr data_ptr = allocator->allocate(n * dtype.itemsize());
    auto impl = std::make_shared<detail::TensorImpl>(std::move(data_ptr), shape, dtype, strides);
    return Tensor(std::move(impl));
}

Tensor full(const IntArray& shape, Scalar value, TensorOptions options) {  
    Tensor out = empty(shape, options);
    if (out.numel() > 0) {
        out.fill_(value);
    }
    return out;
}

Tensor full_like(const Tensor& other, Scalar value, TensorOptions options) {  
    return full(other.shape(), value, other.options().merge_in(options));
}

Tensor zeros(const IntArray& shape, TensorOptions options) {
    Tensor out = empty(shape, options);
    if (out.numel() > 0) {
        out.zero_();
    }
    return out;
}

Tensor zeros_like(const Tensor& other, TensorOptions options) {
    return zeros(other.shape(), other.options().merge_in(options));
}

Tensor ones(const IntArray& shape, TensorOptions options) {
    return full(shape, 1, options);
}

Tensor ones_like(const Tensor& other, TensorOptions options) {
    return ones(other.shape(), other.options().merge_in(options));
}

Tensor scalar_tensor(Scalar s, TensorOptions options) {
    return empty({}, options).fill_(s);
}

}