#pragma once

#include <cstdint>

#include <raxis/export.h>
#include <raxis/core/scalar.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_options.h>


namespace raxis {

// Fills the contiguous result with start + step * i
using arange_fn = void(*)(const Tensor& result, Scalar start, Scalar step);
DECLARE_DISPATCH(arange_fn, arange_stub);

namespace ops {

RAXIS_API Tensor arange(Scalar start, Scalar end, Scalar step = 1, TensorOptions options = {});

} // namespace ops

/// Values in [start, end) spaced by step. Without a dtype, int64 when start,
/// end and step are all integers, the default float dtype otherwise
RAXIS_API Tensor arange(Scalar start, Scalar end, Scalar step = 1, TensorOptions options = {});
RAXIS_API Tensor arange(Scalar end, TensorOptions options = {});

} // namespace raxis
