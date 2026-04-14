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
    return sum(self, {}, false, dtype);
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
    return nansum(self, {}, false, dtype);
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
        sum_out(result, self, dim, keepdim, dtype).div_(dim_prod);
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
    return mean(self, {}, false, dtype);
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
    nansum_out(result, self, dim, keepdim, out_dtype).div_(factor);
    return result;
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
    return nanmean(self, {}, false, dtype);
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
    return prod(self, {}, false, dtype);
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

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(nansum_stub);
DEFINE_DISPATCH(mean_stub);
DEFINE_DISPATCH(prod_stub);
DEFINE_DISPATCH(max_all_stub);
DEFINE_DISPATCH(min_all_stub);
DEFINE_DISPATCH(max_values_stub);
DEFINE_DISPATCH(min_values_stub);
// DEFINE_DISPATCH(argmax_stub);
// DEFINE_DISPATCH(argmin_stub);

} // namespace raxis