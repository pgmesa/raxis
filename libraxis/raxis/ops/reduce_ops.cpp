#include <raxis/ops/reduce_ops.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <limits>
#include <vector>

#include <raxis/core/dtype.h>
#include <raxis/core/device.h>
#include <raxis/core/types.h>
#include <raxis/core/scalar.h>
#include <raxis/core/constants.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>

#include <raxis/ops/reduce_ops_utils.h>
#include <raxis/ops/binary_ops.h>


namespace raxis {

Tensor& sum_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype
) {
    Dtype dtype = get_dtype_from_result(result, opt_dtype);
    auto iter = make_reduction("sum", self, result, dim, keepdim, dtype);

    if (iter.numel() == 0) {
        result.zero_();
    } else {
        sum_stub(iter.device_type(), iter);
    }

    return result;
}

Tensor& nansum_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype
) {
    // For integral types, use existing sum as integral types don't have 'Nan'
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return sum_out(result, self, dim, keepdim, opt_dtype);
    }

    Dtype dtype = get_dtype_from_result(result, opt_dtype);
    auto iter = make_reduction("nansum", self, result, dim, keepdim, dtype);

    if (iter.numel() == 0) {
        result.zero_();
    } else {
        nansum_stub(iter.device_type(), iter);
    }

    return result;
}

Tensor& mean_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype)
{
    Dtype dtype = get_dtype_from_result(result, opt_dtype);
    RAXIS_CHECK(dtype.is_floating(),
        "mean(): could not infer output dtype. Input dtype must be a "
        "floating point dtype. Got: ", dtype.name());

    if (self.device().is_cpu()) {
        // Compute the number of elements being reduced
        int64_t dim_prod = 1;
        if (dim.empty()) {
            dim_prod = self.numel();
        } else {
            for (int64_t d : dim) {
                d = wrap_dim(d, self.ndim());
                dim_prod *= self.shape()[d];
            }
        }
        
        // mean = sum / count (TODO: Implement mean kernel for cpu)
        ops::div_(sum_out(result, self, dim, keepdim, dtype), dim_prod);
    } else {
        auto iter = make_reduction("mean", self, result, dim, keepdim, dtype);

        if (iter.numel() == 0) {
            result.fill_(std::numeric_limits<double>::quiet_NaN());
        } else {
            mean_stub(iter.device_type(), iter);
        }
    }

    return result;
}

Tensor& nanmean_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype)
{
    RAXIS_CHECK(self.dtype().is_floating(),
        "nanmean(): expected input to have floating point dtype but got ",
        self.dtype().name());

    Dtype out_dtype = opt_dtype.value_or(self.dtype());
    
    // nansum / count
    Tensor factor = self.isnan().logical_not_().sum(dim, keepdim);
    ops::div_(nansum_out(result, self, dim, keepdim, out_dtype), factor);
    return result;
}

Tensor& prod_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype
) {
    Dtype dtype = get_dtype_from_result(result, opt_dtype);
    auto iter = make_reduction("prod", self, result, dim, keepdim, dtype);

    if (iter.numel() == 0) {
        result.fill_(1);
    } else {
        prod_stub(iter.device_type(), iter);
    }

    return result;
}

namespace ops {

Tensor sum(
    const Tensor& self, 
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype
) {
    Dtype dtype = get_dtype_from_input(self, opt_dtype, /*promote_integers=*/true);
    Tensor result = create_reduction_result(self, dim, keepdim, dtype);
    return sum_out(result, self, dim, keepdim, dtype);
}

Tensor sum(const Tensor& self, std::optional<Dtype> dtype) {
    return ops::sum(self, {}, false, dtype);
}


Tensor nansum(
    const Tensor& self, 
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype
) {
    Dtype dtype = get_dtype_from_input(self, opt_dtype, /*promote_integers=*/true);
    Tensor result = create_reduction_result(self, dim, keepdim, dtype);
    return nansum_out(result, self, dim, keepdim, dtype);
}

Tensor nansum(const Tensor& self, std::optional<Dtype> dtype) {
    return ops::nansum(self, {}, false, dtype);
}


Tensor mean(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype)
{
    Dtype dtype = get_dtype_from_input(self, opt_dtype, /*promote_integers=*/true);
    Tensor result = create_reduction_result(self, dim, keepdim, dtype);
    return mean_out(result, self, dim, keepdim, dtype);
}

Tensor mean(const Tensor& self, std::optional<Dtype> dtype) {
    return ops::mean(self, {}, false, dtype);
}


Tensor nanmean(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype)
{
    Dtype dtype = get_dtype_from_input(self, opt_dtype, /*promote_integers=*/true);
    Tensor result = create_reduction_result(self, dim, keepdim, dtype);
    return nanmean_out(result, self, dim, keepdim, opt_dtype);
}

Tensor nanmean(const Tensor& self, std::optional<Dtype> dtype) {
    return ops::nanmean(self, {}, false, dtype);
}


Tensor prod(
    const Tensor& self, 
    IntArrayRef dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype
) {
    Dtype dtype = get_dtype_from_input(self, opt_dtype, /*promote_integers=*/true);
    Tensor result = create_reduction_result(self, dim, keepdim, dtype);
    return prod_out(result, self, dim, keepdim, dtype);
}

Tensor prod(const Tensor& self, std::optional<Dtype> dtype) {
    return ops::prod(self, {}, false, dtype);
}


Tensor max(const Tensor& self) {
    RAXIS_CHECK(self.numel() > 0,
        "max(): Expected reduction dim to be specified for input.numel() == 0.");
    Tensor result = empty({}, self.options());
    max_all_stub(self.device().type(), result, self.contiguous());
    return result;
}

Tensor min(const Tensor& self) {
    RAXIS_CHECK(self.numel() > 0,
        "min(): Expected reduction dim to be specified for input.numel() == 0.");
    Tensor result = empty({}, self.options());
    min_all_stub(self.device().type(), result, self.contiguous());
    return result;
}

Tensor amax(const Tensor& self, IntArrayRef dim, bool keepdim) {
    if (self.numel() == 0) {
        zero_numel_check_dims(self, dim, "amax()");
    }
    Tensor result;
    auto iter = make_reduction("amax", self, result, dim, keepdim, self.dtype());
    max_values_stub(iter.device_type(), iter);
    return result;
}

Tensor amin(const Tensor& self, IntArrayRef dim, bool keepdim) {
    if (self.numel() == 0) {
        zero_numel_check_dims(self, dim, "amin()");
    }
    Tensor result;
    auto iter = make_reduction("amin", self, result, dim, keepdim, self.dtype());
    min_values_stub(iter.device_type(), iter);
    return result;
}

// From: aten/src/ATen/native/ReduceOps.cpp (check_argmax_argmin)
static void check_argmax_argmin(const char* name, const Tensor& self, std::optional<int64_t> dim) {
    RAXIS_CHECK(!self.dtype().is_bool(), name, ": does not support bool input");
    if (dim.has_value()) {
        if (self.numel() == 0) {
            zero_numel_check_dims(self, wrap_dim(*dim, self.ndim()), name);
        }
    } else {
        RAXIS_CHECK(self.numel() != 0,
            name, ": Expected reduction dim to be specified for input.numel() == 0.");
    }
}

// From: aten/src/ATen/native/ReduceOps.cpp (argmax_argmin_impl)
// A reduction over `dim` whose output is the int64 index of the winner.
// Without dim, the flattened tensor is reduced to a single index.
template <typename Stub>
static Tensor argmax_argmin_impl(
    const char* name,
    const Tensor& self,
    std::optional<int64_t> dim,
    bool keepdim,
    Stub& stub
) {
    check_argmax_argmin(name, self, dim);

    DimVector dims;
    if (dim.has_value() && self.ndim() > 0) {
        dims.push_back(wrap_dim(*dim, self.ndim()));
    }
    Tensor result = create_reduction_result(self, dims, keepdim, dtypes::Int64);

    // A 0-d input, or a dim of size 1, has a single candidate
    if (self.ndim() == 0 || (dim.has_value() && self.shape()[dims[0]] == 1)) {
        return result.zero_();
    }

    Tensor in = self;
    Tensor out = result;
    if (!dim.has_value()) {
        in = self.reshape({-1});
        out = result.reshape({});
        keepdim = false;
    }
    auto iter = make_reduction(in, out, dims, keepdim, self.dtype());
    if (iter.numel() != 0) {
        stub(iter.device_type(), iter);
    }
    return result;
}

Tensor argmax(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    return argmax_argmin_impl("argmax()", self, dim, keepdim, argmax_stub);
}

Tensor argmin(const Tensor& self, std::optional<int64_t> dim, bool keepdim) {
    return argmax_argmin_impl("argmin()", self, dim, keepdim, argmin_stub);
}
    
} // namespace ops

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(nansum_stub);
DEFINE_DISPATCH(mean_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(max_all_stub);
DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_values_stub);
DEFINE_DISPATCH(min_values_stub);
DEFINE_DISPATCH(argmax_stub);
DEFINE_DISPATCH(argmin_stub);

} // namespace raxis