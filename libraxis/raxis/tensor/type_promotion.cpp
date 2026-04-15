
#include <raxis/tensor/type_promotion.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

namespace {

Dtype promote_skip_undefined(Dtype a, Dtype b) {
    if (a == dtypes::Undefined) return b;
    if (b == dtypes::Undefined) return a;
    return promote_types(a, b);
}

// Higher-priority dtype dominates lower, with special cases
// for crossing category boundaries (e.g., integral + floating).
Dtype combine_categories(Dtype higher, Dtype lower) {
    if (higher.is_floating()) return higher;

    if (higher == dtypes::Bool || lower.is_floating()) {
        return promote_skip_undefined(higher, lower);
    }

    if (higher.defined()) return higher;

    return lower;
}

} // anonymous namespace

PromotionState update_promotion_state(const Tensor& tensor, const PromotionState& state) {
    if (!tensor.defined()) return state;

    PromotionState new_state = state;
    Dtype current = tensor.dtype();

    // Wrapped scalars get remapped to the default dtype so they
    // don't force precision (e.g., Python float64 -> default float32)
    bool wrapped = tensor.is_wrapped_number();
    if (wrapped && current.is_floating()) {
        current = default_float_dtype();
    }

    if (tensor.ndim() > 0) {
        new_state.tensor_dtype = promote_skip_undefined(state.tensor_dtype, current);
    } else if (wrapped) {
        new_state.wrapped_dtype = promote_skip_undefined(state.wrapped_dtype, current);
    } else {
        new_state.zero_dim_dtype = promote_skip_undefined(state.zero_dim_dtype, current);
    }

    return new_state;
}

Dtype compute_promoted_dtype(const PromotionState& state) {
    return combine_categories(
        state.tensor_dtype,
        combine_categories(state.zero_dim_dtype, state.wrapped_dtype)
    );
}

} // namespace raxis