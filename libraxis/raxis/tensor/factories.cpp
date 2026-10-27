
#include <raxis/tensor/factories.h>

#include <memory>

#include <raxis/core/types.h>
#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>
#include <raxis/core/device.h>
#include <raxis/core/allocator.h>
#include <raxis/core/device_guard.h>

#include <raxis/util/data_ptr.h>
#include <raxis/util/exception.h>
#include <raxis/util/unique_void_ptr.h>

#include <raxis/tensor/detail/storage_impl.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_options.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/detail/tensor_data_container.h>

#include <raxis/ops/fill.h>
#include <raxis/ops/distribution_ops.h>


namespace raxis {

namespace detail {
    
inline std::shared_ptr<TensorImpl> make_tensor_impl(
    DataPtr data, 
    IntArrayRef shape, 
    std::optional<DimVector> strides, 
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
    IntArrayRef shape, 
    std::optional<DimVector> strides,
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
    IntArrayRef shape,
    std::optional<DimVector> strides,
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
    IntArrayRef shape,
    std::optional<DimVector> strides,
    Dtype dtype,
    Device device
) {
    return Tensor(detail::make_tensor_impl(
        DataPtr(data, ctx, deleter, device), shape, std::move(strides), dtype)
    );
}

Tensor from_storage(
    std::shared_ptr<detail::StorageImpl> storage,
    int64_t storage_offset,
    IntArrayRef shape,
    IntArrayRef strides,
    Dtype dtype
) {
    RAXIS_CHECK(storage, "from_storage: undefined storage");
    RAXIS_CHECK(shape.size() == strides.size(), "from_storage: shape and strides must have the same length");
    RAXIS_CHECK(storage_offset >= 0, "from_storage: negative storage_offset ", storage_offset);
    int64_t numel = 1;
    int64_t last = storage_offset;
    for (size_t d = 0; d < shape.size(); ++d) {
        RAXIS_CHECK(shape[d] >= 0 && strides[d] >= 0, "from_storage: negative size or stride");
        numel *= shape[d];
        last += (shape[d] - 1) * strides[d];
    }
    const auto storage_numel = static_cast<int64_t>(storage->nbytes() / dtype.itemsize());
    RAXIS_CHECK(numel == 0 || last < storage_numel,
        "from_storage: the view needs ", last + 1, " elements, but the storage has ", storage_numel);
    return Tensor(std::make_shared<detail::TensorImpl>(std::move(storage), storage_offset, shape, strides, dtype));
}

static void set_requires_grad(Tensor& t, const TensorOptions& options) {
    if (options.requires_grad()) {
        RAXIS_CHECK(t.dtype().is_floating(),
            "Only floating point tensors can require gradients");
        t.requires_grad_(true);
    }
}

Tensor empty(IntArrayRef shape, TensorOptions options) {
    Dtype dtype = options.dtype();
    size_t nbytes = static_cast<size_t>(compute_numel(shape)) * dtype.itemsize();
    // The CUDA allocator allocates on the current device
    DeviceGuard guard(options.device());
    auto data_ptr = get_allocator(options.device().type())->allocate(nbytes);
    Tensor t(std::make_shared<detail::TensorImpl>(
        std::move(data_ptr), nbytes, shape, std::nullopt, dtype)
    );
    set_requires_grad(t, options);
    return t;
}

Tensor empty_strided(IntArrayRef shape, IntArrayRef strides, TensorOptions options) {
    RAXIS_CHECK(shape.size() == strides.size(), 
        "Shape and strides must have the same number of dimensions");
    Dtype dtype = options.dtype();
    size_t nbytes = compute_numel_from_strides(shape, strides) * dtype.itemsize();
    DeviceGuard guard(options.device());
    auto data_ptr = get_allocator(options.device().type())->allocate(nbytes);
    Tensor t(std::make_shared<detail::TensorImpl>(
        std::move(data_ptr), nbytes, shape, strides, dtype)
    );
    set_requires_grad(t, options);
    return t;
}

// From: aten/src/ATen/native/TensorFactories.cpp
// Preserve keeps the layout of `other`: an elementwise op between the result
// and `other` then walks both in the same memory order, and a transposed
// gradient can be stored without a copy. A dense `other` (contiguous,
// transposed, permuted) gives its exact strides; one with gaps or repeats (a
// slice, a broadcast view) gives dense strides in its dimension order, since
// copying its strides would allocate the gaps or alias the repeats.
Tensor empty_like(const Tensor& other, TensorOptions options, MemoryFormat memory_format) {
    TensorOptions merged = other.options().merge_in(options);
    if (memory_format == MemoryFormat::Contiguous || other.is_contiguous()) {
        return empty(other.shape(), merged);
    }
    if (other.is_non_overlapping_and_dense()) {
        return empty_strided(other.shape(), other.strides(), merged);
    }
    return empty_strided(other.shape(), infer_dense_strides(other.shape(), other.strides()), merged);
}

Tensor full(IntArrayRef shape, Scalar value, TensorOptions options) {
    if (!options.has_dtype()) {
        options = options.dtype(value.dtype());
    }
    Tensor out = empty(shape, options);
    if (out.numel() > 0) {
        ops::fill_(out, value);
    }
    return out;
}

Tensor full_like(
    const Tensor& other, Scalar value, TensorOptions options, MemoryFormat memory_format
) {
    Tensor out = empty_like(other, options, memory_format);
    if (out.numel() > 0) {
        ops::fill_(out, value);
    }
    return out;
}

Tensor zeros(IntArrayRef shape, TensorOptions options) {
    Tensor out = empty(shape, options);
    if (out.numel() > 0) {
        ops::zero_(out);
    }
    return out;
}

Tensor zeros_like(
    const Tensor& other, TensorOptions options, MemoryFormat memory_format
) {
    Tensor out = empty_like(other, options, memory_format);
    if (out.numel() > 0) {
        ops::zero_(out);
    }
    return out;
}

Tensor ones(IntArrayRef shape, TensorOptions options) {
    if (!options.has_dtype()) {
        options = options.dtype(default_float_dtype());
    }
    return full(shape, 1, options);
}

Tensor ones_like(
    const Tensor& other, TensorOptions options, MemoryFormat memory_format
) {
    return full_like(other, 1, options, memory_format);
}

Tensor rand(IntArrayRef shape, TensorOptions options) {
    if (!options.has_dtype()) {
        options = options.dtype(default_float_dtype());
    }
    RAXIS_CHECK(
        options.dtype().is_floating(),
        "rand requires a floating-point dtype, got ", options.dtype()
    );
    Tensor out = empty(shape, options);
    return ops::uniform_(out, 0.0, 1.0);
}

Tensor rand_like(
    const Tensor& other, TensorOptions options, MemoryFormat memory_format
) {
    Tensor out = empty_like(other, options, memory_format);
    RAXIS_CHECK(
        out.dtype().is_floating(),
        "rand_like requires a floating-point dtype, got ", out.dtype()
    );
    return ops::uniform_(out, 0.0, 1.0);
}

Tensor randn(IntArrayRef shape, TensorOptions options) {
    if (!options.has_dtype()) {
        options = options.dtype(default_float_dtype());
    }
    RAXIS_CHECK(
        options.dtype().is_floating(),
        "randn requires a floating-point dtype, got ", options.dtype()
    );
    Tensor out = empty(shape, options);
    return ops::normal_(out, 0.0, 1.0);
}

Tensor randn_like(
    const Tensor& other, TensorOptions options, MemoryFormat memory_format
) {
    Tensor out = empty_like(other, options, memory_format);
    RAXIS_CHECK(
        out.dtype().is_floating(),
        "randn_like requires a floating-point dtype, got ", out.dtype()
    );
    return ops::normal_(out, 0.0, 1.0);
}

Tensor randint(int64_t low, int64_t high, IntArrayRef shape, TensorOptions options) {
    if (!options.has_dtype()) {
        options = options.dtype(default_int_dtype());
    }
    RAXIS_CHECK(
        options.dtype().is_integral(true),
        "randint requires an integral or bool dtype, got ", options.dtype()
    );
    Tensor out = empty(shape, options);
    return ops::randint_(out, low, high);
}

Tensor randint_like(
    const Tensor& other, int64_t low, int64_t high, TensorOptions options, MemoryFormat memory_format
) {
    Tensor out = empty_like(other, options, memory_format);
    RAXIS_CHECK(
        out.dtype().is_integral(true),
        "randint_like requires an integral or bool dtype, got ", out.dtype()
    );
    return ops::randint_(out, low, high);
}

Tensor scalar_tensor(Scalar s, TensorOptions options) {
    Tensor out = empty({}, options);
    return ops::fill_(out, s);
}

Tensor tensor_from_scalars(
    ArrayRef<Scalar> data,
    IntArrayRef shape,
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