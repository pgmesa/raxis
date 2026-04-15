#pragma once

#include <vector>
#include <functional>
#include <stdexcept>

#include <raxis/export.h>
#include <raxis/core/types.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>
#include <raxis/core/scalar.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_options.h>
#include <raxis/tensor/detail/tensor_data_container.h>

namespace raxis {

using Deleter = std::function<void(void*)>;

RAXIS_API Tensor from_blob(
    void* data, 
    IntArrayRef shape, 
    std::optional<DimVector> strides,
    Dtype dtype, 
    Device device = DeviceType::CPU
);

RAXIS_API Tensor from_blob(
    void* data,
    DeleterFnPtr deleter,
    IntArrayRef shape,
    std::optional<DimVector> strides,
    Dtype dtype,
    Device device = DeviceType::CPU
);

RAXIS_API Tensor from_blob(
    void* data,
    void* ctx,
    DeleterFnPtr deleter,
    IntArrayRef shape,
    std::optional<DimVector> strides,
    Dtype dtype,
    Device device = DeviceType::CPU
);

RAXIS_API Tensor empty(IntArrayRef shape, TensorOptions options = {});
RAXIS_API Tensor empty_like(const Tensor& other, TensorOptions options = {});
RAXIS_API Tensor empty_strided(
    IntArrayRef shape, IntArrayRef strides, TensorOptions options = {});

RAXIS_API Tensor full(IntArrayRef shape, Scalar value, TensorOptions options = {});
RAXIS_API Tensor full_like(const Tensor& other, Scalar value, TensorOptions options = {});

RAXIS_API Tensor zeros(IntArrayRef shape, TensorOptions options = {});
RAXIS_API Tensor zeros_like(const Tensor& other, TensorOptions options = {});

RAXIS_API Tensor ones(IntArrayRef shape, TensorOptions options = {});
RAXIS_API Tensor ones_like(const Tensor& other, TensorOptions options = {});

/// Returns a tensor filled with U[0, 1)
RAXIS_API Tensor rand(IntArrayRef shape, TensorOptions options = {});
RAXIS_API Tensor rand_like(const Tensor& other, TensorOptions options = {});

/// Returns a tensor filled with N(0, 1)
RAXIS_API Tensor randn(IntArrayRef shape, TensorOptions options = {});
RAXIS_API Tensor randn_like(const Tensor& other, TensorOptions options = {});

RAXIS_API Tensor scalar_tensor(Scalar s, TensorOptions options = {});


// ============================================================================
// tensor() - create from data
// ============================================================================
//
// Accepts nested braced-init-lists, scalars, and std::vector<T>.
//
//   raxis::tensor({{1, 2}, {3, 4}})                        -> int64, shape {2,2}
//   raxis::tensor({{1.0, 2.0}}, dtypes::Float64)           -> float64, shape {1,2}
//   raxis::tensor(3.14f)                                   -> float32, shape {}
//   raxis::tensor(std::vector<float>{1,2,3})               -> float32, shape {3}
//   raxis::tensor(std::vector<int>{1,2,3,4}, {{2,2}})      -> int32, shape {2,2}

inline Dtype compute_desired_dtype(Dtype inferred) {
    if (inferred.is_integral(/*include_bool=*/false)) {
        return default_int_dtype();
    }
    if (inferred.is_floating()) {
        return default_float_dtype();
    }
    return inferred;
}

RAXIS_API Tensor tensor_from_scalars(
    ArrayRef<Scalar> data,
    IntArrayRef shape,
    TensorOptions options = {}
);

RAXIS_API Tensor tensor(
    detail::TensorDataContainer data, TensorOptions options = {});

// Typed flat data + optional reshape
template <typename T>
Tensor tensor(ArrayRef<T> data, TensorOptions options = {}) {
    DimVector shape = {static_cast<int64_t>(data.size())};
    return tensor(data, shape, options);
}

template <typename T>
Tensor tensor(ArrayRef<T> data, IntArrayRef shape, TensorOptions options = {}) {
    int64_t expected = compute_numel(shape);
    RAXIS_CHECK(
        static_cast<int64_t>(data.size()) == expected,
        "tensor(): data size (", data.size(),
        ") does not match shape numel (", expected, ")"
    );

    Dtype src_dtype = Dtype::from_type<T>();
    if (!options.has_dtype()) {
        options = options.dtype(compute_desired_dtype(src_dtype));
    }

    // Allocate on CPU, bulk-copy, then move to target device if needed
    TensorOptions cpu_opts = options.device(DeviceType::CPU);
    Tensor out = empty(shape, cpu_opts);

    // Type-dispatch: cast source T -> target dtype element by element
    Dtype dst_dtype = out.dtype();
    dst_dtype.dispatch_all_types([&]<typename dest_T>(std::type_identity<dest_T>) {
        dest_T* dst = out.mutable_data_ptr<dest_T>();
        for (size_t i = 0; i < data.size(); ++i) {
            dst[i] = static_cast<dest_T>(data[i]);
        }
    });

    if (options.device().type() != DeviceType::CPU) {
        out = out.to(options.device());
    }
    return out;
}

}
