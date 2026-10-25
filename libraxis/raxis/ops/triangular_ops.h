#pragma once

#include <cstdint>

#include <raxis/export.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

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
