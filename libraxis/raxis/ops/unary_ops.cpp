#include <raxis/ops/unary_ops.h>

#include <raxis/ops/dispatch_stub.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

#define RAXIS_UNARY_OP(name, config, stub)                  \
    Tensor name(const Tensor& self) {                       \
        Tensor out;                                         \
        auto iter = iterator::config()                      \
            .add_output(out)                                \
            .add_input(self)                                \
            .build();                                       \
        stub(iter.device_type(), iter);                     \
        return iter.get_output(0);                          \
    }                                                       \
    Tensor& name##_(Tensor& self) {                         \
        auto iter = iterator::config()                      \
            .add_output(self)                               \
            .add_input(self)                                \
            .build();                                       \
        stub(iter.device_type(), iter);                     \
        return self;                                        \
    }

Tensor isnan(const Tensor& self) {
    return self != self;
}

Tensor neg(const Tensor& self) {
    RAXIS_CHECK(!self.dtype().is_bool(),
        "Negation, the `-` operator, on a bool tensor is not supported. "
        "Use `logical_not()` instead.");

    Tensor out;
    auto iter = iterator::unary_op_config()
        .add_output(out)
        .add_input(self)
        .build();

    neg_stub(iter.device_type(), iter);
    return iter.get_output(0);
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

// Integer identity ops
Tensor ceil(const Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self.clone();
    }
    Tensor out;
    auto iter = iterator::unary_op_config()
        .add_output(out)
        .add_input(self)
        .build();
    ceil_stub(iter.device_type(), iter);
    return iter.get_output(0);
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

Tensor floor(const Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self.clone();
    }
    Tensor out;
    auto iter = iterator::unary_op_config()
        .add_output(out)
        .add_input(self)
        .build();
    floor_stub(iter.device_type(), iter);
    return iter.get_output(0);
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

Tensor round(const Tensor& self) {
    if (self.dtype().is_integral(/*include_bool=*/true)) {
        return self.clone();
    }
    Tensor out;
    auto iter = iterator::unary_op_config()
        .add_output(out)
        .add_input(self)
        .build();
    round_stub(iter.device_type(), iter);
    return iter.get_output(0);
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

RAXIS_UNARY_OP(exp,        unary_float_op_config, exp_stub)
RAXIS_UNARY_OP(expm1,      unary_float_op_config, expm1_stub)
RAXIS_UNARY_OP(log,        unary_float_op_config, log_stub)
RAXIS_UNARY_OP(log2,       unary_float_op_config, log2_stub)
RAXIS_UNARY_OP(log10,      unary_float_op_config, log10_stub)
RAXIS_UNARY_OP(log1p,      unary_float_op_config, log1p_stub)
RAXIS_UNARY_OP(sqrt,       unary_float_op_config, sqrt_stub)
RAXIS_UNARY_OP(rsqrt,      unary_float_op_config, rsqrt_stub)
RAXIS_UNARY_OP(reciprocal, unary_float_op_config, reciprocal_stub)
RAXIS_UNARY_OP(sigmoid,    unary_float_op_config, sigmoid_stub)

RAXIS_UNARY_OP(sin,  unary_float_op_config, sin_stub)
RAXIS_UNARY_OP(cos,  unary_float_op_config, cos_stub)
RAXIS_UNARY_OP(tan,  unary_float_op_config, tan_stub)
RAXIS_UNARY_OP(asin, unary_float_op_config, asin_stub)
RAXIS_UNARY_OP(acos, unary_float_op_config, acos_stub)
RAXIS_UNARY_OP(atan, unary_float_op_config, atan_stub)
RAXIS_UNARY_OP(sinh, unary_float_op_config, sinh_stub)
RAXIS_UNARY_OP(cosh, unary_float_op_config, cosh_stub)
RAXIS_UNARY_OP(tanh, unary_float_op_config, tanh_stub)
RAXIS_UNARY_OP(erf,  unary_float_op_config, erf_stub)
RAXIS_UNARY_OP(erfc, unary_float_op_config, erfc_stub)


Tensor logical_not(const Tensor& self) {
    Tensor out;
    auto iter = iterator::comparison_op_config()
        .add_output(out)
        .add_input(self)
        .build();
    logical_not_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor& logical_not_(Tensor& self) {
    auto iter = iterator::comparison_op_config()
        .add_output(self)
        .add_input(self)
        .build();
    logical_not_stub(iter.device_type(), iter);
    return self;
}

#undef RAXIS_UNARY_OP


DEFINE_DISPATCH(neg_stub);
DEFINE_DISPATCH(abs_stub);
DEFINE_DISPATCH(ceil_stub);
DEFINE_DISPATCH(floor_stub);
DEFINE_DISPATCH(round_stub);
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