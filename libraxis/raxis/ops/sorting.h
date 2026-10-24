#pragma once

#include <cstdint>
#include <tuple>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// Sort every slice of self along dim into values / indices (same shape as
// self). Ties keep their original order, so every sort is stable.
using sort_fn = void(*)(const Tensor& values, const Tensor& indices, const Tensor& self,
                        int64_t dim, bool descending);
// The k largest (smallest) of every slice along dim; values / indices have
// size k along dim
using topk_fn = void(*)(const Tensor& values, const Tensor& indices, const Tensor& self,
                        int64_t k, int64_t dim, bool largest, bool sorted);

DECLARE_DISPATCH(sort_fn, sort_stub);
DECLARE_DISPATCH(topk_fn, topk_stub);

namespace ops {

RAXIS_API std::tuple<Tensor, Tensor> sort(
    const Tensor& self, int64_t dim = -1, bool descending = false, bool stable = false);
RAXIS_API std::tuple<Tensor, Tensor> topk(
    const Tensor& self, int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true);

} // namespace ops

/// Sorted values and their positions along dim. NaN sorts as the largest
/// value, as in torch. `stable` is accepted for the torch API: raxis always
/// keeps equal values in their original order.
RAXIS_API std::tuple<Tensor, Tensor> sort(
    const Tensor& self, int64_t dim = -1, bool descending = false, bool stable = false);

/// The positions that sort self along dim
RAXIS_API Tensor argsort(
    const Tensor& self, int64_t dim = -1, bool descending = false, bool stable = false);

/// The k largest (or smallest) values along dim and their positions,
/// in sorted order when `sorted`
RAXIS_API std::tuple<Tensor, Tensor> topk(
    const Tensor& self, int64_t k, int64_t dim = -1, bool largest = true, bool sorted = true);

} // namespace raxis
