
#include <raxis/tensor/factories.h>

#include <memory>

#include <raxis/core/types.h>
#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>
#include <raxis/core/device.h>
#include <raxis/core/allocator.h>

#include <raxis/util/data_ptr.h>
#include <raxis/util/exception.h>
#include <raxis/util/unique_void_ptr.h>

#include <raxis/tensor/detail/storage_impl.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_options.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/detail/tensor_data_container.h>


namespace raxis {

Tensor from_blob(void* data, const IntArray& shape, Dtype dtype, Device device) {
    // Non-owning: Caller guarantees lifetime.
    DataPtr ptr (data, device);
    auto impl = std::make_shared<detail::TensorImpl>(std::move(ptr), compute_numel(shape), shape, dtype);
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
    auto impl = std::make_shared<detail::TensorImpl>(std::move(ptr), compute_numel(shape), shape, dtype);
    return Tensor(std::move(impl));
}

Tensor from_blob(
    void* data,
    void* ctx,
    DeleterFnPtr deleter,
    const IntArray& shape,
    Dtype dtype,
    Device device
) {
    DataPtr ptr(data, ctx, deleter, device);
    auto impl = std::make_shared<detail::TensorImpl>(std::move(ptr), compute_numel(shape), shape, dtype);
    return Tensor(std::move(impl));
}

Tensor empty(const IntArray& shape, TensorOptions options) {
    Dtype dtype = options.dtype();
    size_t n = static_cast<size_t>(compute_numel(shape));
    auto allocator = get_allocator(options.device().type());
    size_t nbytes = n * dtype.itemsize();
    DataPtr data_ptr = allocator->allocate(nbytes);
    auto impl = std::make_shared<detail::TensorImpl>(std::move(data_ptr), nbytes, shape, dtype);
    return Tensor(std::move(impl));
}

Tensor empty_like(const Tensor& other, TensorOptions options) {
    return empty(other.shape(), other.options().merge_in(options));
}

Tensor empty_strided(const IntArray& shape, const IntArray& strides, TensorOptions options) {
    // Validate shapes and strides match
    RAXIS_CHECK(shape.size() == strides.size(), 
        "Shape and strides must have the same number of dimensions");
    
    // Compute required memory based on shape and strides
    Dtype dtype = options.dtype();
    size_t n = compute_numel_from_strides(shape, strides);
    auto allocator = get_allocator(options.device().type());
    size_t nbytes = n * dtype.itemsize();
    DataPtr data_ptr = allocator->allocate(nbytes);
    auto impl = std::make_shared<detail::TensorImpl>(std::move(data_ptr), nbytes, shape, dtype, strides);
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

Tensor tensor_from_scalars(
    const std::vector<Scalar>& data,
    const IntArray& shape,
    TensorOptions options
) {
    // Allocate on CPU, bulk-write, then move to target device
    TensorOptions cpu_opts = options.device(DeviceType::CPU);
    Tensor out = empty(shape, cpu_opts);

    if (!data.empty()) {
        out.dtype().dispatch_all_types([&]<typename dest_T>(std::type_identity<dest_T>) {
            dest_T* out_ptr = out.mutable_data_ptr<dest_T>();
            for (size_t i = 0; i < data.size(); ++i) {
                out_ptr[i] = data[i].to<dest_T>();
            }
        });
    }

    if (options.device().type() != DeviceType::CPU) {
        out = out.to(options.device());
    }
    return out;
}

Tensor tensor(detail::TensorDataContainer data, TensorOptions options) {
    // 1. Handle scalar (0-d tensor)
    Dtype desired = options.dtype_opt().value_or(compute_desired_dtype(data.dtype()));
    TensorOptions resolved = options.dtype(desired);
    
    if (data.is_scalar()) {
        return scalar_tensor(data.scalar(), resolved);
    }

    // Flatten all leaf scalars into a contiguous vector (row-major)
    std::vector<Scalar> flat;
    int64_t numel = compute_numel(data.sizes());
    flat.reserve(static_cast<size_t>(numel));
    data.flatten_into(flat);

    return tensor_from_scalars(flat, data.sizes(), resolved);
}

}