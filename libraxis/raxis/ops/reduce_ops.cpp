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
    const IntArray& dim,
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
    const IntArray& dim,
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
    const IntArray& dim,
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
    const IntArray& dim,
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
    const IntArray& dim,
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
    const IntArray& dim,
    bool keepdim,
    std::optional<Dtype> opt_dtype)
{
    Dtype dtype = get_dtype_from_input(self, opt_dtype, /*promote_integers=*/true);
    Tensor result = create_reduction_result(self, dim, keepdim, dtype);
    return mean_out(result, self, dim, keepdim, opt_dtype);
}

Tensor mean(const Tensor& self, std::optional<Dtype> dtype) {
    return mean(self, {}, false, dtype);
}


Tensor& nanmean_out(
    Tensor& result,
    const Tensor& self,
    const IntArray& dim,
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
    const IntArray& dim,
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

DEFINE_DISPATCH(sum_stub);
DEFINE_DISPATCH(nansum_stub);
DEFINE_DISPATCH(mean_stub);

} // namespace raxis