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
// DECLARE_DISPATCH(reduce_fn, argmax_stub);
// DECLARE_DISPATCH(reduce_fn, argmin_stub);

RAXIS_API Tensor sum(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor sum(
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& sum_out(
    Tensor& result,
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor nansum(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor nansum(
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& nansum_out(
    Tensor& result,
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor mean(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor mean(
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& mean_out(
    Tensor& result,
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor nanmean(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor nanmean(
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& nanmean_out(
    Tensor& result,
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor prod(
    const Tensor& self,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor prod(
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);

RAXIS_API Tensor& prod_out(
    Tensor& result,
    const Tensor& self,
    const IntArray& dim,
    bool keepdim = false,
    std::optional<Dtype> dtype = std::nullopt
);


RAXIS_API Tensor max(const Tensor& self);
RAXIS_API Tensor min(const Tensor& self);

RAXIS_API Tensor amax(
    const Tensor& self,
    const IntArray& dim = {},
    bool keepdim = false
);
RAXIS_API Tensor amin(
    const Tensor& self, 
    const IntArray& dim = {},
    bool keepdim = false
);

// RAXIS_API std::pair<Tensor, Tensor> max(const Tensor& self, int64_t dim, bool keepdim = false);
// RAXIS_API std::pair<Tensor, Tensor> min(const Tensor& self, int64_t dim, bool keepdim = false);

// RAXIS_API Tensor argmax(
//     const Tensor& self,
//     std::optional<int64_t> dim = std::nullopt,
//     bool keepdim = false
// );
// RAXIS_API Tensor argmin(
//     const Tensor& self,
//     std::optional<int64_t> dim = std::nullopt,
//     bool keepdim = false
// );

} // namespace raxis