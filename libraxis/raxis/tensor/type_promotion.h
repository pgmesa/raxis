#pragma once

#include <raxis/export.h>
#include <raxis/core/dtype.h>
#include <raxis/util/array_ref.h>

namespace raxis {

class Tensor;

struct PromotionState {
    Dtype tensor_dtype  = dtypes::Undefined;  // from dim > 0 tensors
    Dtype zero_dim_dtype = dtypes::Undefined;  // from explicit 0-dim tensors
    Dtype wrapped_dtype  = dtypes::Undefined;  // from wrapped scalars
};

PromotionState update_promotion_state(const Tensor& tensor, const PromotionState& state);
Dtype compute_promoted_dtype(const PromotionState& state);

// From: aten/src/ATen/native/TypeProperties.cpp
// The dtype an elementwise op over these tensors computes in
RAXIS_API Dtype result_type(ArrayRef<Tensor> tensors);

} // namespace raxis