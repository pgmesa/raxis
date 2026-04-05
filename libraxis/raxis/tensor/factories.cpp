
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

namespace detail {
    
inline std::shared_ptr<TensorImpl> make_tensor_impl(
    DataPtr data, 
    const IntArray& shape, 
    std::optional<IntArray> strides, 
    Dtype dtype
) {
    size_t nbytes = strides 
        ? compute_numel_from_strides(shape, *strides) * dtype.itemsize()
        : compute_numel(shape) * dtype.itemsize();

    return std::make_shared<TensorImpl>(
        std::move(data), nbytes, shape, std::move(strides), dtype
    );
}

} // namespace detail


Tensor from_blob(
    void* data, 
    const IntArray& shape, 
    std::optional<IntArray> strides,
    Dtype dtype, 
    Device device
) {
    // Non-owning: Caller guarantees lifetime.
    return Tensor(detail::make_tensor_impl(
        DataPtr(data, device), shape, std::move(strides), dtype)
    );
}

Tensor from_blob(
    void* data,
    DeleterFnPtr deleter,
    const IntArray& shape,
    std::optional<IntArray> strides,
    Dtype dtype,
    Device device
) {
    return Tensor(detail::make_tensor_impl(
        DataPtr(data, data, deleter, device), shape, std::move(strides), dtype)
    );
}

Tensor from_blob(
    void* data,
    void* ctx,
    DeleterFnPtr deleter,
    const IntArray& shape,
    std::optional<IntArray> strides,
    Dtype dtype,
    Device device
) {
    return Tensor(detail::make_tensor_impl(
        DataPtr(data, ctx, deleter, device), shape, std::move(strides), dtype)
    );
}

Tensor empty(const IntArray& shape, TensorOptions options) {
    Dtype dtype = options.dtype();
    size_t nbytes = static_cast<size_t>(compute_numel(shape)) * dtype.itemsize();
    auto data_ptr = get_allocator(options.device().type())->allocate(nbytes);
    return Tensor(std::make_shared<detail::TensorImpl>(
        std::move(data_ptr), nbytes, shape, std::nullopt, dtype)
    );
}

Tensor empty_strided(const IntArray& shape, const IntArray& strides, TensorOptions options) {
    RAXIS_CHECK(shape.size() == strides.size(), 
        "Shape and strides must have the same number of dimensions");
    Dtype dtype = options.dtype();
    size_t nbytes = compute_numel_from_strides(shape, strides) * dtype.itemsize();
    auto data_ptr = get_allocator(options.device().type())->allocate(nbytes);
    return Tensor(std::make_shared<detail::TensorImpl>(
        std::move(data_ptr), nbytes, shape, strides, dtype)
    );
}

Tensor empty_like(const Tensor& other, TensorOptions options) {
    return empty(other.shape(), other.options().merge_in(options));
}

Tensor full(const IntArray& shape, Scalar value, TensorOptions options) {
    if (!options.has_dtype()) {
        options = options.dtype(value.dtype());
    }
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
    if (!options.has_dtype()) {
        options = options.dtype(default_float_dtype());
    }
    return full(shape, 1, options);
}

Tensor ones_like(const Tensor& other, TensorOptions options) {
    return ones(other.shape(), other.options().merge_in(options));
}

Tensor rand(const IntArray& shape, TensorOptions options) {
    if (!options.has_dtype()) {
        options = options.dtype(default_float_dtype());
    }
    RAXIS_CHECK(
        options.dtype().is_floating(),
        "rand requires a floating-point dtype, got ", options.dtype()
    );
    return empty(shape, options).uniform_(0.0, 1.0);
}

Tensor rand_like(const Tensor& other, TensorOptions options) {
    auto merged = other.options().merge_in(options);
    return rand(other.shape(), merged);
}

Tensor randn(const IntArray& shape, TensorOptions options) {
    if (!options.has_dtype()) {
        options = options.dtype(default_float_dtype());
    }
    RAXIS_CHECK(
        options.dtype().is_floating(),
        "randn requires a floating-point dtype, got ", options.dtype()
    );
    return empty(shape, options).normal_(0.0, 1.0);
}

Tensor randn_like(const Tensor& other, TensorOptions options) {
    auto merged = other.options().merge_in(options);
    return randn(other.shape(), merged);
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
    Dtype desired = options.dtype_opt().value_or(compute_desired_dtype(data.dtype()));
    TensorOptions resolved = options.dtype(desired);
    
    // Handle scalar (0-d tensor)
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