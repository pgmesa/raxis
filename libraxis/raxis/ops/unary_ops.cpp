#include <raxis/ops/unary_ops.h>

#include <raxis/ops/copy.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/resize.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/detail/storage_impl.h>


namespace raxis {

#define RAXIS_UNARY_OP(name, config, stub)                  \
    Tensor& name##_out(const Tensor& self, Tensor& out) {   \
        auto iter = iterator::config()                      \
            .resize_outputs(true)                           \
            .add_output(out)                                \
            .add_input(self)                                \
            .build();                                       \
        stub(iter.device_type(), iter);                     \
        out = iter.get_output(0);                           \
        return out;                                         \
    }                                                       \
    Tensor name(const Tensor& self) {                       \
        Tensor out;                                         \
        return ops::name##_out(self, out);                  \
    }                                                       \
    Tensor& name##_(Tensor& self) {                         \
        auto iter = iterator::config()                      \
            .add_output(self)                               \
            .add_input(self)                                \
            .build();                                       \
        stub(iter.device_type(), iter);                     \
        return self;                                        \
    }

namespace ops {

namespace {

// ceil, floor and round of an integer tensor are the tensor itself
Tensor& copy_into_out(const Tensor& self, Tensor& out) {
    if (!out.defined()) {
        out = self.clone();
        return out;
    }
    resize_output(out, self.shape());
    copy(self, out);
    return out;
}

} // anonymous namespace

Tensor& neg_out(const Tensor& self, Tensor& out) {
    RAXIS_CHECK(!self.dtype().is_bool(),
        "Negation, the `-` operator, on a bool tensor is not supported. "
        "Use `logical_not()` instead.");

    auto iter = iterator::unary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(self)
        .build();

    neg_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor neg(const Tensor& self) {
    Tensor out;
    return ops::neg_out(self, out);
}

Tensor& neg_(Tensor& self) {
    RAXIS_CHECK(!self.dtype().is_bool(),
        "Negation, the `-` operator, on a bool tensor is not supported. "
        "Use `logical_not()` instead.");

    auto iter = iterator::unary_op_config()
        .add_output(self)
        .add_input(self)
        .build();

    neg_stub(iter.device_type(), iter);
    return self;
}

RAXIS_UNARY_OP(abs, unary_op_config, abs_stub)
RAXIS_UNARY_OP(sign, unary_op_config, sign_stub)

// Integer identity ops
Tensor& ceil_out(const Tensor& self, Tensor& out) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return copy_into_out(self, out);
    }
    auto iter = iterator::unary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(self)
        .build();
    ceil_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor ceil(const Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self.clone();
    }
    Tensor out;
    return ops::ceil_out(self, out);
}

Tensor& ceil_(Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self;
    }
    auto iter = iterator::unary_op_config()
        .add_output(self)
        .add_input(self)
        .build();
    ceil_stub(iter.device_type(), iter);
    return self;
}

Tensor& floor_out(const Tensor& self, Tensor& out) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return copy_into_out(self, out);
    }
    auto iter = iterator::unary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(self)
        .build();
    floor_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor floor(const Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self.clone();
    }
    Tensor out;
    return ops::floor_out(self, out);
}

Tensor& floor_(Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self;
    }
    auto iter = iterator::unary_op_config()
        .add_output(self)
        .add_input(self)
        .build();
    floor_stub(iter.device_type(), iter);
    return self;
}

Tensor& round_out(const Tensor& self, Tensor& out) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return copy_into_out(self, out);
    }
    auto iter = iterator::unary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(self)
        .build();
    round_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor round(const Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self.clone();
    }
    Tensor out;
    return ops::round_out(self, out);
}

Tensor& round_(Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self;
    }
    auto iter = iterator::unary_op_config()
        .add_output(self)
        .add_input(self)
        .build();
    round_stub(iter.device_type(), iter);
    return self;
}

Tensor& trunc_out(const Tensor& self, Tensor& out) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return copy_into_out(self, out);
    }
    auto iter = iterator::unary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(self)
        .build();
    trunc_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor trunc(const Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self.clone();
    }
    Tensor out;
    return ops::trunc_out(self, out);
}

Tensor& trunc_(Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self;
    }
    auto iter = iterator::unary_op_config()
        .add_output(self)
        .add_input(self)
        .build();
    trunc_stub(iter.device_type(), iter);
    return self;
}

// x - trunc(x): floating tensors only, as torch (an integer has no fraction,
// but torch raises instead of returning zeros)
Tensor& frac_out(const Tensor& self, Tensor& out) {
    RAXIS_CHECK_NOT_IMPLEMENTED(self.dtype().is_floating(),
        "\"frac\" not implemented for '", self.dtype().name(), "'");
    auto iter = iterator::unary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(self)
        .build();
    frac_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor frac(const Tensor& self) {
    Tensor out;
    return ops::frac_out(self, out);
}

Tensor& frac_(Tensor& self) {
    return ops::frac_out(self, self);
}

RAXIS_UNARY_OP(exp, unary_float_op_config, exp_stub)
RAXIS_UNARY_OP(expm1, unary_float_op_config, expm1_stub)
RAXIS_UNARY_OP(log, unary_float_op_config, log_stub)
RAXIS_UNARY_OP(log2, unary_float_op_config, log2_stub)
RAXIS_UNARY_OP(log10, unary_float_op_config, log10_stub)
RAXIS_UNARY_OP(log1p, unary_float_op_config, log1p_stub)
RAXIS_UNARY_OP(sqrt, unary_float_op_config, sqrt_stub)
RAXIS_UNARY_OP(rsqrt, unary_float_op_config, rsqrt_stub)
RAXIS_UNARY_OP(reciprocal, unary_float_op_config, reciprocal_stub)
RAXIS_UNARY_OP(sigmoid, unary_float_op_config, sigmoid_stub)

RAXIS_UNARY_OP(sin, unary_float_op_config, sin_stub)
RAXIS_UNARY_OP(cos, unary_float_op_config, cos_stub)
RAXIS_UNARY_OP(tan, unary_float_op_config, tan_stub)
RAXIS_UNARY_OP(asin, unary_float_op_config, asin_stub)
RAXIS_UNARY_OP(acos, unary_float_op_config, acos_stub)
RAXIS_UNARY_OP(atan, unary_float_op_config, atan_stub)
RAXIS_UNARY_OP(sinh, unary_float_op_config, sinh_stub)
RAXIS_UNARY_OP(cosh, unary_float_op_config, cosh_stub)
RAXIS_UNARY_OP(tanh, unary_float_op_config, tanh_stub)
RAXIS_UNARY_OP(erf, unary_float_op_config, erf_stub)
RAXIS_UNARY_OP(erfc, unary_float_op_config, erfc_stub)

} // namespace ops

#undef RAXIS_UNARY_OP

Tensor isnan(const Tensor& self) {
    return self != self;
}

namespace {

Tensor& logical_not_into(const Tensor& self, Tensor& out) {
    // The kernel writes bools: an out of another dtype gets them converted
    if (out.defined() && !out.dtype().is_bool()) {
        Tensor result;
        logical_not_into(self, result);
        resize_output(out, result.shape());
        copy(result, out);
        return out;
    }
    auto iter = iterator::comparison_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(self)
        .build();
    logical_not_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

} // anonymous namespace

Tensor logical_not(const Tensor& self) {
    Tensor out;
    return logical_not_into(self, out);
}

// Not differentiable: only the version of out moves
Tensor& logical_not_out(const Tensor& self, Tensor& out) {
    logical_not_into(self, out);
    out.unsafe_get_tensor_impl()->storage()->bump_version();
    return out;
}

Tensor& logical_not_(Tensor& self) {
    auto iter = iterator::comparison_op_config()
        .add_output(self)
        .add_input(self)
        .build();
    logical_not_stub(iter.device_type(), iter);
    // Not differentiable, so no autograd wrapper: the in-place write still
    // makes saved copies of self stale
    self.unsafe_get_tensor_impl()->storage()->bump_version();
    return self;
}

namespace {

Tensor& bitwise_not_into(const Tensor& self, Tensor& out) {
    RAXIS_CHECK_NOT_IMPLEMENTED(self.dtype().is_integral(/*include_bool=*/true),
        "\"bitwise_not\" not implemented for '", self.dtype().name(), "'");
    auto iter = iterator::unary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(self)
        .build();
    bitwise_not_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

} // anonymous namespace

// Integer and bool tensors cannot require grad: no autograd wrapper
Tensor bitwise_not(const Tensor& self) {
    Tensor out;
    return bitwise_not_into(self, out);
}

Tensor& bitwise_not_out(const Tensor& self, Tensor& out) {
    bitwise_not_into(self, out);
    out.unsafe_get_tensor_impl()->storage()->bump_version();
    return out;
}

Tensor& bitwise_not_(Tensor& self) {
    bitwise_not_into(self, self);
    self.unsafe_get_tensor_impl()->storage()->bump_version();
    return self;
}

DEFINE_DISPATCH(neg_stub);
DEFINE_DISPATCH(abs_stub);
DEFINE_DISPATCH(sign_stub);
DEFINE_DISPATCH(ceil_stub);
DEFINE_DISPATCH(floor_stub);
DEFINE_DISPATCH(round_stub);
DEFINE_DISPATCH(trunc_stub);
DEFINE_DISPATCH(frac_stub);
DEFINE_DISPATCH(exp_stub);
DEFINE_DISPATCH(expm1_stub);
DEFINE_DISPATCH(log_stub);
DEFINE_DISPATCH(log2_stub);
DEFINE_DISPATCH(log10_stub);
DEFINE_DISPATCH(log1p_stub);
DEFINE_DISPATCH(sqrt_stub);
DEFINE_DISPATCH(rsqrt_stub);
DEFINE_DISPATCH(reciprocal_stub);
DEFINE_DISPATCH(sigmoid_stub);
DEFINE_DISPATCH(logical_not_stub);
DEFINE_DISPATCH(bitwise_not_stub);
DEFINE_DISPATCH(sin_stub);
DEFINE_DISPATCH(cos_stub);
DEFINE_DISPATCH(tan_stub);
DEFINE_DISPATCH(asin_stub);
DEFINE_DISPATCH(acos_stub);
DEFINE_DISPATCH(atan_stub);
DEFINE_DISPATCH(sinh_stub);
DEFINE_DISPATCH(cosh_stub);
DEFINE_DISPATCH(tanh_stub);
DEFINE_DISPATCH(erf_stub);
DEFINE_DISPATCH(erfc_stub);

} // namespace raxis