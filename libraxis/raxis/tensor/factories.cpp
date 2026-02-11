
#include <raxis/tensor/factories.h>

#include <memory>

#include <raxis/core/types.h>

#include <raxis/util/unique_void_ptr.h>

#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/detail/tensor_impl.h>


namespace raxis {

Tensor from_blob(void* data, const IntArray& shape, Dtype dtype) {
    // Non-owning: Caller guarantees lifetime.
    auto ptr = detail::UniqueVoidPtr(data);
    auto impl = std::make_shared<TensorImpl>(std::move(ptr), shape, dtype);
    return Tensor(std::move(impl));
}

Tensor from_blob(
    void* data,
    const IntArray& shape,          
    Dtype dtype,
    detail::DeleterFnPtr deleter
) {
    detail::UniqueVoidPtr ptr(data, data, deleter);
    auto impl = std::make_shared<TensorImpl>(std::move(ptr), shape, dtype);
    return Tensor(std::move(impl));
}

Tensor empty(const IntArray& shape, Dtype dtype) {
    size_t n = static_cast<size_t>(compute_numel(shape)); //returns int64_t
    
    detail::UniqueVoidPtr ptr;

    if (n > 0) {
        void* data = std::malloc(n * dtype.itemsize());
        if (!data) {
            throw std::bad_alloc();
        }
        ptr = detail::UniqueVoidPtr(data, data, &std::free);
    }
    
    auto impl = std::make_shared<TensorImpl>(std::move(ptr), shape, dtype);
    return Tensor(std::move(impl));
}

Tensor empty_strided(const IntArray& shape, const IntArray& strides, Dtype dtype) {
    // Validate shapes and strides match
    if (shape.size() != strides.size()) {
        throw std::invalid_argument(
            "Shape and strides must have the same number of dimensions"
        );
    }
    
    // Compute required memory based on shape and strides
    size_t n = compute_numel_from_strides(shape, strides);
    
    detail::UniqueVoidPtr ptr;
    if (n > 0) {
        void* data = std::malloc(n * dtype.itemsize());
        if (!data) {
            throw std::bad_alloc();
        }
        ptr = detail::UniqueVoidPtr(data, data, &std::free);
    }
    
    auto impl = std::make_shared<TensorImpl>(std::move(ptr), shape, dtype, strides);
    return Tensor(std::move(impl));
}

Tensor full(const IntArray& shape, double value, Dtype dtype) {  
    Tensor out = empty(shape, dtype);
    size_t n = static_cast<size_t>(out.numel());
    if (n > 0) dtype.fill(out.mutable_data_ptr(), n, value);
    return out;
}

Tensor zeros(const IntArray& shape, Dtype dtype) {
    return full(shape, 0, dtype);
}

Tensor ones(const IntArray& shape, Dtype dtype) {
    return full(shape, 1, dtype);
}


// Tensor scalar_tensor(Scalar s, Dtype dtype) {
//     return empty({}, dtype);
// }

}