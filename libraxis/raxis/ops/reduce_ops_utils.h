#pragma once

#include <optional>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

using DimMask = TensorIterator::DimMask;

DimMask make_dim_mask(const std::optional<IntArrayRef>& opt_dims, int64_t ndim) {
    DimMask mask;

    if (!opt_dims.has_value() || opt_dims->empty()) {
        // No dims specified, reduce all dims
        for (int64_t i = 0; i < ndim; ++i) {
            mask.set(i);
        }
    } else {
        for (int64_t d : opt_dims.value()) {
            d = wrap_dim(d, ndim);
            RAXIS_CHECK(!mask.test(d),
                "dim ", d, " appears multiple times in the list of dims");
            mask.set(d);
        }
    }
    return mask;
}

DimVector shape_from_dim_mask(IntArrayRef input_shape, const DimMask& mask, bool keepdim) {
    DimVector shape;
    for (int64_t dim = 0; dim < static_cast<int64_t>(input_shape.size()); ++dim) {
        if (mask.test(dim)) {
            if (keepdim) {
                shape.push_back(1);
            }
            // else: dim is removed from output
        } else {
            shape.push_back(input_shape[dim]);
        }
    }
    return shape;
}

Dtype get_dtype_from_input(const Tensor& self, std::optional<Dtype> dtype, bool promote_integers) {
    if (dtype.has_value()) {
        return dtype.value();
    }

    Dtype src_dtype = self.dtype();
    if (promote_integers && src_dtype.is_integral(/*include_bool=*/true)) {
        return dtypes::Int64;
    }
    return src_dtype;
}

Dtype get_dtype_from_result(Tensor& result, std::optional<Dtype> dtype) {
    RAXIS_CHECK(result.defined(), 
        "Cannot create a new tensor inside a reduction op. You likely tried "
        "to call an operator with an out argument but the out argument was an undefined tensor.");
    if (dtype.has_value()) {
        return dtype.value();
    } else {
        return result.dtype();
    }
}

inline Tensor create_reduction_result(
  const Tensor& self, std::optional<IntArrayRef> dim, bool keepdim, Dtype dtype
) {
    DimMask mask = make_dim_mask(dim, self.ndim());
    auto shape = shape_from_dim_mask(self.shape(), mask, keepdim);
    return empty(shape, self.options().dtype(dtype));
}

Tensor review_reduce_result(
    const Tensor& result,
    int64_t ndim,
    const DimMask& mask,
    bool keepdim
) {
    if (keepdim) {
        return result;
    }
    // Build new shape and strides by re-inserting size-1 / stride-0 dims
    DimVector shape;
    DimVector strides;
    
    IntArrayRef result_shape = result.shape();
    IntArrayRef result_strides = result.strides();
    
    int64_t result_idx = 0;
    for (int64_t dim = 0; dim < ndim; ++dim) {
        if (mask.test(dim)) {
            shape.push_back(1);
            strides.push_back(0);
        } else {
            shape.push_back(result_shape[result_idx]);
            strides.push_back(result_strides[result_idx]);
            result_idx++;
        }
    }
    
    // Create a view that shares the same memory
    return from_blob(
        result.mutable_data_ptr(),
        shape,
        strides,
        result.dtype(),
        result.device()
    );
}

TensorIterator make_reduction(
    const char* name,
    const Tensor& self,
    Tensor& result,
    const std::optional<IntArrayRef>& opt_dims,
    bool keepdim,
    Dtype out_dtype)
{
    int64_t ndim = self.ndim();
    DimMask mask = make_dim_mask(opt_dims, ndim);
    DimVector out_shape = shape_from_dim_mask(self.shape(), mask, keepdim);

    // Allocate or validate the output tensor
    if (!result.defined()) {
        result = empty(out_shape, self.options().dtype(out_dtype));
    } else {
        RAXIS_CHECK(result.shape() == out_shape,
            name, ": result shape ", result.shape(), " doesn't match expected ", out_shape);
        // Use the result's dtype, the caller chose it deliberately
        out_dtype = result.dtype();
    }

    // Create the stride-0 view of the output, matching input rank
    Tensor viewed_result = review_reduce_result(result, ndim, mask, keepdim);

    // Build the iterator configured for reduction
    return TensorIterator::reduce_op(
        viewed_result, 
        self.dtype() == out_dtype ? self : self.to(out_dtype)
    );
}

inline void zero_numel_check_dims(const Tensor& self, int64_t dim, const char* fn_name){
    if (self.ndim() == 0) {
        RAXIS_CHECK(dim == 0 || dim == -1, fn_name,
            ": Expected reduction dim -1 or 0 for scalar but got ", dim);
    } else {
        RAXIS_CHECK(self.shape()[dim] != 0, fn_name,
            ": Expected reduction dim ", dim, " to have non-zero size.");
    }
}

inline void zero_numel_check_dims(const Tensor& self, IntArrayRef dim, const char* fn_name) {
    RAXIS_CHECK(!dim.empty(), fn_name,
        ": Expected reduction dim to be specified for input.numel() == 0.");
    for (int64_t d : dim) {
        zero_numel_check_dims(self, wrap_dim(d, self.ndim()), fn_name);
    }
}
    
} // namespace raxis
