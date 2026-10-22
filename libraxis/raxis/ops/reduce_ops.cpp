#include <raxis/ops/reduce_ops.h>

#include <algorithm>
#include <cmath>
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
#include <raxis/util/parallel.h>

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

// From: aten/src/ATen/native/ReduceOps.cpp (std_var_all_cpu)
// Reducing everything to one value, a plain two-pass algorithm in double is
// faster than Welford (whose division per element does not vectorize) and
// accurate enough: mean first, then sum((x - mean)^2) in parallel.
static double std_var_all_cpu(const Tensor& self, double correction, bool take_sqrt) {
    double mean = ops::mean(self).item().to<double>();
    auto iter = TensorIteratorConfig()
        .add_input(self)
        .build();

    auto reduction = [&](int64_t begin, int64_t end, double thread_sum) {
        self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            iter.serial_for_each([&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
                const int64_t inner_stride = strides[0];
                const int64_t outer_stride = strides[1];
                double local_sum = 0.0;
                for (int64_t i = 0; i < size1; i++) {
                    const char* row = data[0] + outer_stride * i;
                    for (int64_t j = 0; j < size0; j++) {
                        double dx = static_cast<double>(
                            *reinterpret_cast<const scalar_t*>(row + inner_stride * j)) - mean;
                        local_sum += dx * dx;
                    }
                }
                thread_sum += local_sum;
            }, {begin, end});
        });
        return thread_sum;
    };

    double sum_dx2 = parallel_reduce(
        int64_t{0}, iter.numel(), detail::GRAIN_SIZE, 0.0, reduction, std::plus<>{});
    double var = sum_dx2 / std::max(0.0, static_cast<double>(self.numel()) - correction);
    return take_sqrt ? std::sqrt(var) : var;
}

// From: aten/src/ATen/native/ReduceOps.cpp (std_var_out)
static Tensor std_var(
    const char* name, const Tensor& self, IntArrayRef dim,
    std::optional<double> correction_opt, bool keepdim, bool take_sqrt
) {
    RAXIS_CHECK(self.dtype().is_floating(),
        "std and var only support floating point and complex dtypes");
    double correction = correction_opt.value_or(1.0);

    Tensor result;
    auto iter = make_reduction(name, self, result, dim, keepdim, self.dtype());
    if (iter.numel() == 0) {
        result.fill_(std::numeric_limits<double>::quiet_NaN());
    } else if (result.numel() == 1) {
        result.fill_(std_var_all_cpu(self, correction, take_sqrt));
    } else {
        std_var_stub(iter.device_type(), iter, correction, take_sqrt);
    }
    return result;
}

Tensor var(const Tensor& self, IntArrayRef dim, std::optional<double> correction, bool keepdim) {
    return std_var("var", self, dim, correction, keepdim, /*take_sqrt=*/false);
}

Tensor std(const Tensor& self, IntArrayRef dim, std::optional<double> correction, bool keepdim) {
    return std_var("std", self, dim, correction, keepdim, /*take_sqrt=*/true);
}

// From: aten/src/ATen/native/ReduceOps.cpp (allany_impl, get_allany_iter)
// On CPU the input is cast to bool first (make_reduction casts it to the
// output dtype), so one bool kernel serves every dtype. A uint8 input gives
// a uint8 result: the bool result is converted at the end.
template <bool identity, typename Stub>
static Tensor all_any_impl(const char* name, const Tensor& self, IntArrayRef dim, bool keepdim, Stub& stub) {
    Tensor result;
    auto iter = make_reduction(name, self, result, dim, keepdim, dtypes::Bool);
    if (self.numel() == 0) {
        result.fill_(identity);
    } else {
        stub(iter.device_type(), iter);
    }
    return self.dtype() == dtypes::Uint8 ? result.to(dtypes::Uint8) : result;
}

// From: aten/src/ATen/native/ReduceOps.cpp (cumsum, cumsum_out)
static Tensor cumsum_impl(const Tensor& self, int64_t dim, std::optional<Dtype> dtype, bool reverse) {
    const int64_t wrapped = wrap_dim(dim, std::max<int64_t>(self.ndim(), 1));
    const Dtype out_dtype = dtype.value_or(
        self.dtype().is_integral(/*include_bool=*/true) ? dtypes::Int64 : self.dtype());
    const Tensor input = self.dtype() == out_dtype ? self : self.to(out_dtype);
    Tensor result = empty(self.shape(), self.options().dtype(out_dtype));
    if (self.ndim() == 0) {
        result.copy_(input);
    } else if (self.numel() > 0) {
        cumsum_stub(self.device().type(), result, input, wrapped, reverse);
    }
    return result;
}

Tensor cumsum(const Tensor& self, int64_t dim, std::optional<Dtype> dtype) {
    return cumsum_impl(self, dim, dtype, /*reverse=*/false);
}

Tensor reversed_cumsum(const Tensor& self, int64_t dim) {
    return cumsum_impl(self, dim, std::nullopt, /*reverse=*/true);
}

Tensor all(const Tensor& self, IntArrayRef dim, bool keepdim) {
    return all_any_impl<true>("all", self, dim, keepdim, and_stub);
}

Tensor any(const Tensor& self, IntArrayRef dim, bool keepdim) {
    return all_any_impl<false>("any", self, dim, keepdim, or_stub);
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
DEFINE_DISPATCH(and_stub);
DEFINE_DISPATCH(or_stub);
DEFINE_DISPATCH(std_var_stub);
DEFINE_DISPATCH(cumsum_stub);

} // namespace raxis