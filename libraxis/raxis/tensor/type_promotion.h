#pragma once

#include <raxis/core/dtype.h>

namespace raxis {

class Tensor;

struct PromotionState {
    Dtype tensor_dtype  = dtypes::Undefined;  // from dim > 0 tensors
    Dtype zero_dim_dtype = dtypes::Undefined;  // from explicit 0-dim tensors
    Dtype wrapped_dtype  = dtypes::Undefined;  // from wrapped scalars
};

PromotionState update_promotion_state(const Tensor& tensor, const PromotionState& state);
Dtype compute_promoted_dtype(const PromotionState& state);

} // namespace raxis