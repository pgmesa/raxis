#pragma once

#include <bitset>
#include <optional>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>

#include <raxis/export.h>

#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/types.h>
#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

using reduce_fn = void(*)(TensorIterator&);
using reduce_all_fn = void(*)(Tensor& result, const Tensor& self);

DECLARE_DISPATCH(reduce_fn, sum_stub);
DECLARE_DISPATCH(reduce_fn, nansum_stub);
DECLARE_DISPATCH(reduce_fn, mean_stub);
DECLARE_DISPATCH(reduce_fn, nanmean_stub);
DECLARE_DISPATCH(reduce_fn, prod_stub);
DECLARE_DISPATCH(reduce_all_fn, max_all_stub);
DECLARE_DISPATCH(reduce_all_fn, min_all_stub);
DECLARE_DISPATCH(reduce_fn, max_values_stub);
DECLARE_DISPATCH(reduce_fn, min_values_stub);
DECLARE_DISPATCH(reduce_fn, argmax_stub);
DECLARE_DISPATCH(reduce_fn, argmin_stub);
DECLARE_DISPATCH(reduce_fn, and_stub);
DECLARE_DISPATCH(reduce_fn, or_stub);

using std_var_fn = void(*)(TensorIterator&, double correction, bool take_sqrt);
DECLARE_DISPATCH(std_var_fn, std_var_stub);

// result[..., i, ...] = sum of self[..., j, ...] for j <= i along dim, or for
// j >= i when reverse. result and self have the same shape and dtype.
using cumsum_fn = void(*)(const Tensor& result, const Tensor& self, int64_t dim, bool reverse);
DECLARE_DISPATCH(cumsum_fn, cumsum_stub);

RAXIS_API Tensor sum(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor sum(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& sum_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor nansum(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor nansum(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& nansum_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor mean(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor mean(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& mean_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor nanmean(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor nanmean(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& nanmean_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor prod(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor prod(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& prod_out(
    Tensor& result,
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor max(const Tensor& self);
RAXIS_API Tensor min(const Tensor& self);

RAXIS_API Tensor amax(
    const Tensor& self,
    IntArrayRef dim = {},
    bool keepdim = false
);
RAXIS_API Tensor amin(
    const Tensor& self, 
    IntArrayRef dim = {},
    bool keepdim = false
);

RAXIS_API Tensor argmax(
    const Tensor& self,
    std::optional<int64_t> dim = std::nullopt,
    bool keepdim = false
);
RAXIS_API Tensor argmin(
    const Tensor& self,
    std::optional<int64_t> dim = std::nullopt,
    bool keepdim = false
);

/// Cumulative sum along dim. Integral and bool inputs give int64, as sum.
RAXIS_API Tensor cumsum(
    const Tensor& self,
    int64_t dim,
    std::optional<Dtype> dtype = std::nullopt
);

/// Whether all / any elements over dim (all dims if empty) are nonzero.
/// The result is bool, or uint8 for uint8 input (as PyTorch, for
/// compatibility with the time before the bool dtype).
RAXIS_API Tensor all(
    const Tensor& self,
    IntArrayRef dim = {},
    bool keepdim = false
);
RAXIS_API Tensor any(
    const Tensor& self,
    IntArrayRef dim = {},
    bool keepdim = false
);

/// Variance and standard deviation over dim (all dims if empty), dividing
/// by n - correction (1 by default: the unbiased estimator)
RAXIS_API Tensor var(
    const Tensor& self,
    IntArrayRef dim = {},
    std::optional<double> correction = std::nullopt,
    bool keepdim = false
);
RAXIS_API Tensor std(
    const Tensor& self,
    IntArrayRef dim = {},
    std::optional<double> correction = std::nullopt,
    bool keepdim = false
);


namespace ops {

RAXIS_API Tensor sum(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor sum(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor nansum(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor nansum(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor mean(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor mean(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor nanmean(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor nanmean(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor prod(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor prod(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor max(const Tensor& self);
RAXIS_API Tensor min(const Tensor& self);

RAXIS_API Tensor amax(
    const Tensor& self,
    IntArrayRef dim = {},
    bool keepdim = false
);
RAXIS_API Tensor amin(
    const Tensor& self, 
    IntArrayRef dim = {},
    bool keepdim = false
);

RAXIS_API Tensor argmax(
    const Tensor& self,
    std::optional<int64_t> dim = std::nullopt,
    bool keepdim = false
);
RAXIS_API Tensor argmin(
    const Tensor& self,
    std::optional<int64_t> dim = std::nullopt,
    bool keepdim = false
);

/// Cumulative sum along dim. Integral and bool inputs give int64, as sum.
RAXIS_API Tensor cumsum(
    const Tensor& self,
    int64_t dim,
    std::optional<Dtype> dtype = std::nullopt
);

/// Cumulative sum from the end of dim: self.flip(dim).cumsum(dim).flip(dim)
/// in one pass, without the two flipped copies (the backward of cumsum).
RAXIS_API Tensor reversed_cumsum(const Tensor& self, int64_t dim);

/// Whether all / any elements over dim (all dims if empty) are nonzero.
/// The result is bool, or uint8 for uint8 input (as PyTorch, for
/// compatibility with the time before the bool dtype).
RAXIS_API Tensor all(
    const Tensor& self,
    IntArrayRef dim = {},
    bool keepdim = false
);
RAXIS_API Tensor any(
    const Tensor& self,
    IntArrayRef dim = {},
    bool keepdim = false
);

/// Variance and standard deviation over dim (all dims if empty), dividing
/// by n - correction (1 by default: the unbiased estimator)
RAXIS_API Tensor var(
    const Tensor& self,
    IntArrayRef dim = {},
    std::optional<double> correction = std::nullopt,
    bool keepdim = false
);
RAXIS_API Tensor std(
    const Tensor& self,
    IntArrayRef dim = {},
    std::optional<double> correction = std::nullopt,
    bool keepdim = false
);
    
} // namespace ops


} // namespace raxis