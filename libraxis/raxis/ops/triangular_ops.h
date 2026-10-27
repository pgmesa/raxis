#pragma once

#include <cstdint>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// result: contiguous [..., n, m]; self: same shape, with its own strides in
// the last two dims and contiguous over the batch dims. k is clamped to
// [-n, m]
using triu_tril_fn = void(*)(const Tensor& result, const Tensor& self, int64_t k, bool upper);
DECLARE_DISPATCH(triu_tril_fn, triu_tril_stub);

namespace ops {

RAXIS_API Tensor tril(const Tensor& self, int64_t diagonal = 0);
RAXIS_API Tensor triu(const Tensor& self, int64_t diagonal = 0);

} // namespace ops

/// The lower triangle of the matrices in the last two dims: elements above
/// the diagonal-th diagonal are zeroed (diagonal > 0 keeps diagonals above
/// the main one, < 0 removes diagonals below it)
RAXIS_API Tensor tril(const Tensor& self, int64_t diagonal = 0);
/// The upper triangle: elements below the diagonal-th diagonal are zeroed
RAXIS_API Tensor triu(const Tensor& self, int64_t diagonal = 0);

} // namespace raxis
