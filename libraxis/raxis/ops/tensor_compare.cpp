#include <raxis/ops/tensor_compare.h>

#include <cmath>
#include <limits>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/type_promotion.h>

#include <raxis/ops/scalar_ops.h>


namespace raxis {

namespace ops {

// From: aten/src/ATen/native/TensorCompare.cpp (where_self_out)
// self and other are promoted to their common dtype first, so the kernel
// only selects between two values of the same type
Tensor where(const Tensor& condition, const Tensor& self, const Tensor& other) {
    RAXIS_CHECK(condition.dtype() == dtypes::Bool,
        "where expected condition to be a boolean tensor, but got a tensor with dtype ",
        condition.dtype());

    Dtype dtype = result_type({self, other});
    Tensor out;
    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .declare_static_dtype(dtype)
        .add_output(out)
        .add_input(condition)
        .add_owned_input(self.to(dtype))
        .add_owned_input(other.to(dtype))
        .build();

    where_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor where(const Tensor& condition, const Tensor& self, Scalar other) {
    return ops::where(condition, self, wrapped_scalar_tensor(other, self.device()));
}

Tensor where(const Tensor& condition, Scalar self, const Tensor& other) {
    return ops::where(condition, wrapped_scalar_tensor(self, other.device()), other);
}


// From: aten/src/ATen/native/TensorCompare.cpp (clamp meta function)
// Bounds take part in type promotion like the scalars of a binary op: an
// integer tensor clamped to 2.5 becomes float
static Dtype clamp_result_type(
    const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max
) {
    if (self.dtype().is_floating()) {
        return self.dtype();
    }
    PromotionState state = update_promotion_state(self, {});
    if (min) state = update_promotion_state(wrapped_scalar_tensor(*min), state);
    if (max) state = update_promotion_state(wrapped_scalar_tensor(*max), state);
    return compute_promoted_dtype(state);
}

// From: aten/src/ATen/native/TensorCompare.cpp (clamp_out)
static void clamp_impl(TensorIterator& iter, std::optional<Scalar> min, std::optional<Scalar> max) {
    // A NaN bound makes every element NaN
    bool min_is_nan = min && std::isnan(min->to<double>());
    bool max_is_nan = max && std::isnan(max->to<double>());
    if (min_is_nan || max_is_nan) {
        iter.get_output(0).fill_(std::numeric_limits<double>::quiet_NaN());
    } else if (min && max) {
        clamp_scalar_stub(iter.device_type(), iter, *min, *max);
    } else if (max) {
        clamp_max_scalar_stub(iter.device_type(), iter, *max);
    } else {
        clamp_min_scalar_stub(iter.device_type(), iter, *min);
    }
}

Tensor clamp(const Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    RAXIS_CHECK(min || max, "clamp: At least one of 'min' or 'max' must not be None");
    RAXIS_CHECK(!self.dtype().is_bool(), "clamp is not supported for bool tensors");

    Tensor out;
    auto iter = iterator::unary_op_config()
        .add_output(out)
        .add_owned_input(self.to(clamp_result_type(self, min, max)))
        .build();
    clamp_impl(iter, min, max);
    return iter.get_output(0);
}

Tensor& clamp_(Tensor& self, std::optional<Scalar> min, std::optional<Scalar> max) {
    RAXIS_CHECK(min || max, "clamp: At least one of 'min' or 'max' must not be None");
    RAXIS_CHECK(!self.dtype().is_bool(), "clamp is not supported for bool tensors");
    Dtype dtype = clamp_result_type(self, min, max);
    RAXIS_CHECK(dtype == self.dtype(),
        "result type ", dtype, " can't be cast to the desired output type ", self.dtype());

    auto iter = iterator::unary_op_config()
        .add_output(self)
        .add_input(self)
        .build();
    clamp_impl(iter, min, max);
    return self;
}

} // namespace ops

DEFINE_DISPATCH(where_stub);
DEFINE_DISPATCH(clamp_scalar_stub);
DEFINE_DISPATCH(clamp_min_scalar_stub);
DEFINE_DISPATCH(clamp_max_scalar_stub);

} // namespace raxis
