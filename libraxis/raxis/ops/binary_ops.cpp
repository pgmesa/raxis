
#include <raxis/ops/binary_ops.h>

#include <iostream>

#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/scalar_ops.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

Tensor& add(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    add_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor& sub(const Tensor& a, const Tensor& b, Tensor& out) {
    bool is_boolean_op = a.dtype().is_bool() && b.dtype().is_bool();
    RAXIS_CHECK(!is_boolean_op, 
        "Subtraction, the `-` operator, with two bool tensors is not supported.");

    RAXIS_CHECK(!a.dtype().is_bool() && !b.dtype().is_bool(), 
        "Subtraction, the `-` operator, with a bool tensor is not supported. "
        "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");

    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    sub_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor& mul(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    mul_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor& div(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_float_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    div_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}


Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out;
    add(a, b, out);
    return out;
}

Tensor add(const Tensor& a, Scalar s) {
    Tensor out;
    add(a, wrapped_scalar_tensor(s), out);
    return out;
}


Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor out;
    sub(a, b, out);
    return out;
}

Tensor sub(const Tensor& a, Scalar s) {
    Tensor out;
    sub(a, wrapped_scalar_tensor(s), out);
    return out;
}


Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out;
    mul(a, b, out);
    return out;
}

Tensor mul(const Tensor& a, Scalar s) {
    Tensor out;
    mul(a, wrapped_scalar_tensor(s), out);
    return out;
}


Tensor div(const Tensor& a, const Tensor& b) {
    Tensor out;
    div(a, b, out);
    return out;
}

Tensor div(const Tensor& a, Scalar s) {
    Tensor out;
    div(a, wrapped_scalar_tensor(s), out);
    return out;
}

// -- Reversed
Tensor rsub(const Tensor& a, const Tensor& b) {
    Tensor out;
    sub(b, a, out);
    return out;
}

Tensor rsub(const Tensor& a, Scalar s) {
    Tensor out;
    sub(wrapped_scalar_tensor(s), a, out);
    return out;
}


Tensor rdiv(const Tensor& a, const Tensor& b) {
    Tensor out;
    div(b, a, out);
    return out;
}

Tensor rdiv(const Tensor& a, Scalar s) {
    Tensor out;
    div(wrapped_scalar_tensor(s), a, out);
    return out;
}


// In-place
Tensor& add_(Tensor& a, const Tensor& b) {
    return add(a, b, a);
}

Tensor& add_(Tensor& a, Scalar s) {
    return add(a, wrapped_scalar_tensor(s), a);
}


Tensor& sub_(Tensor& a, const Tensor& b) {
    return sub(a, b, a);
}

Tensor& sub_(Tensor& a, Scalar s) {
    return sub(a, wrapped_scalar_tensor(s), a);
}


Tensor& mul_(Tensor& a, const Tensor& b) {
    return mul(a, b, a);
}

Tensor& mul_(Tensor& a, Scalar s) {
    return mul(a, wrapped_scalar_tensor(s), a);
}


Tensor& div_(Tensor& a, const Tensor& b) {
    return div(a, b, a);
}

Tensor& div_(Tensor& a, Scalar s) {
    return div(a, wrapped_scalar_tensor(s), a);
}

// -- Reversed
Tensor& rsub_(Tensor& a, const Tensor& b) {
    return sub(b, a, a);
}

Tensor& rsub_(Tensor& a, Scalar s) {
    return sub(wrapped_scalar_tensor(s), a, a);
}


Tensor& rdiv_(Tensor& a, const Tensor& b) {
    return div(b, a, a);
}

Tensor& rdiv_(Tensor& a, Scalar s) {
    return div(wrapped_scalar_tensor(s), a, a);
}


DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_stub);


namespace {
 
template<typename Stub>
Tensor comparison_op(const Tensor& a, const Tensor& b, Stub& stub) {
    Tensor out;
    auto iter = iterator::comparison_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();
 
    stub(iter.device_type(), iter);
    return iter.get_output(0);
}
 
} // namespace
 
Tensor gt(const Tensor& a, const Tensor& b) { return comparison_op(a, b, gt_stub); }
Tensor gt(const Tensor& a, Scalar s)        { return gt(a, wrapped_scalar_tensor(s)); }
Tensor ge(const Tensor& a, const Tensor& b) { return comparison_op(a, b, ge_stub); }
Tensor ge(const Tensor& a, Scalar s)        { return ge(a, wrapped_scalar_tensor(s)); }
Tensor lt(const Tensor& a, const Tensor& b) { return comparison_op(a, b, lt_stub); }
Tensor lt(const Tensor& a, Scalar s)        { return lt(a, wrapped_scalar_tensor(s)); }
Tensor le(const Tensor& a, const Tensor& b) { return comparison_op(a, b, le_stub); }
Tensor le(const Tensor& a, Scalar s)        { return le(a, wrapped_scalar_tensor(s)); }
Tensor eq(const Tensor& a, const Tensor& b) { return comparison_op(a, b, eq_stub); }
Tensor eq(const Tensor& a, Scalar s)        { return eq(a, wrapped_scalar_tensor(s)); }
Tensor ne(const Tensor& a, const Tensor& b) { return comparison_op(a, b, ne_stub); }
Tensor ne(const Tensor& a, Scalar s)        { return ne(a, wrapped_scalar_tensor(s)); }
 

Tensor maximum(const Tensor& a, const Tensor& b) {
    Tensor out;
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();
 
    maximum_stub(iter.device_type(), iter);
    return iter.get_output(0);
}
 
Tensor minimum(const Tensor& a, const Tensor& b) {
    Tensor out;
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();
 
    minimum_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor logical_and(const Tensor& a, const Tensor& b) {
    Tensor out;
    auto iter = iterator::comparison_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();
 
    logical_and_stub(iter.device_type(), iter);
    return iter.get_output(0);
}
 
Tensor logical_or(const Tensor& a, const Tensor& b) {
    Tensor out;
    auto iter = iterator::comparison_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();
 
    logical_or_stub(iter.device_type(), iter);
    return iter.get_output(0);
}
 

DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(ge_stub);
DEFINE_DISPATCH(lt_stub);
DEFINE_DISPATCH(le_stub);
DEFINE_DISPATCH(eq_stub);
DEFINE_DISPATCH(ne_stub);
DEFINE_DISPATCH(maximum_stub);
DEFINE_DISPATCH(minimum_stub);
DEFINE_DISPATCH(logical_and_stub);
DEFINE_DISPATCH(logical_or_stub);

}