#include <raxis/ops/unary_ops.h>

#include <raxis/ops/dispatch_stub.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

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


Tensor abs(const Tensor& self) {
    Tensor out;
    auto iter = iterator::unary_op_config()
        .add_output(out)
        .add_input(self)
        .build();

    abs_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor& abs_(Tensor& self) {
    auto iter = iterator::unary_op_config()
        .add_output(self)
        .add_input(self)
        .build();

    abs_stub(iter.device_type(), iter);
    return self;
}


Tensor ceil(const Tensor& self) {
    // Integral types: ceil is identity
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


Tensor exp(const Tensor& self) {
    Tensor out;
    auto iter = iterator::unary_float_op_config()
        .add_output(out)
        .add_input(self)
        .build();

    exp_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor& exp_(Tensor& self) {
    auto iter = iterator::unary_float_op_config()
        .add_output(self)
        .add_input(self)
        .build();

    exp_stub(iter.device_type(), iter);
    return self;
}


Tensor log(const Tensor& self) {
    Tensor out;
    auto iter = iterator::unary_float_op_config()
        .add_output(out)
        .add_input(self)
        .build();

    log_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor& log_(Tensor& self) {
    auto iter = iterator::unary_float_op_config()
        .add_output(self)
        .add_input(self)
        .build();

    log_stub(iter.device_type(), iter);
    return self;
}


Tensor sqrt(const Tensor& self) {
    Tensor out;
    auto iter = iterator::unary_float_op_config()
        .add_output(out)
        .add_input(self)
        .build();

    sqrt_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor& sqrt_(Tensor& self) {
    auto iter = iterator::unary_float_op_config()
        .add_output(self)
        .add_input(self)
        .build();

    sqrt_stub(iter.device_type(), iter);
    return self;
}

Tensor rsqrt(const Tensor& self) {
    Tensor out;
    auto iter = iterator::unary_float_op_config()
        .add_output(out)
        .add_input(self)
        .build();

    rsqrt_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor& rsqrt_(Tensor& self) {
    auto iter = iterator::unary_float_op_config()
        .add_output(self)
        .add_input(self)
        .build();

    rsqrt_stub(iter.device_type(), iter);
    return self;
}

Tensor reciprocal(const Tensor& self) {
    Tensor out;
    auto iter = iterator::unary_float_op_config()
        .add_output(out)
        .add_input(self)
        .build();

    reciprocal_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor& reciprocal_(Tensor& self) {
    auto iter = iterator::unary_float_op_config()
        .add_output(self)
        .add_input(self)
        .build();

    reciprocal_stub(iter.device_type(), iter);
    return self;
}

Tensor sigmoid(const Tensor& self) {
    Tensor out;
    auto iter = iterator::unary_float_op_config()
        .add_output(out)
        .add_input(self)
        .build();

    sigmoid_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor& sigmoid_(Tensor& self) {
    auto iter = iterator::unary_float_op_config()
        .add_output(self)
        .add_input(self)
        .build();

    sigmoid_stub(iter.device_type(), iter);
    return self;
}


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


// Dispatch definitions
DEFINE_DISPATCH(neg_stub);
DEFINE_DISPATCH(abs_stub);
DEFINE_DISPATCH(ceil_stub);
DEFINE_DISPATCH(floor_stub);
DEFINE_DISPATCH(round_stub);
DEFINE_DISPATCH(exp_stub);
DEFINE_DISPATCH(log_stub);
DEFINE_DISPATCH(sqrt_stub);
DEFINE_DISPATCH(rsqrt_stub);
DEFINE_DISPATCH(reciprocal_stub);
DEFINE_DISPATCH(sigmoid_stub);
DEFINE_DISPATCH(logical_not_stub);

} // namespace raxis