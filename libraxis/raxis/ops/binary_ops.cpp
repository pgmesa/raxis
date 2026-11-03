
#include <raxis/ops/binary_ops.h>

#include <iostream>

#include <raxis/ops/copy.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/scalar_ops.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/resize.h>

#include <raxis/autograd/variable.h>


namespace raxis {

namespace ops {

Tensor& add_out(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    add_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor& sub_out(const Tensor& a, const Tensor& b, Tensor& out) {
    bool is_boolean_op = a.dtype().is_bool() && b.dtype().is_bool();
    RAXIS_CHECK(!is_boolean_op, 
        "Subtraction, the `-` operator, with two bool tensors is not supported.");

    RAXIS_CHECK(!a.dtype().is_bool() && !b.dtype().is_bool(), 
        "Subtraction, the `-` operator, with a bool tensor is not supported. "
        "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");

    auto iter = iterator::binary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    sub_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor& mul_out(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    mul_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor& div_out(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_float_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    div_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor& add_out(const Tensor& a, Scalar s, Tensor& out) { return ops::add_out(a, wrapped_scalar_tensor(s), out); }
Tensor& sub_out(const Tensor& a, Scalar s, Tensor& out) { return ops::sub_out(a, wrapped_scalar_tensor(s), out); }
Tensor& mul_out(const Tensor& a, Scalar s, Tensor& out) { return ops::mul_out(a, wrapped_scalar_tensor(s), out); }
Tensor& div_out(const Tensor& a, Scalar s, Tensor& out) { return ops::div_out(a, wrapped_scalar_tensor(s), out); }

Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out;
    ops::add_out(a, b, out);
    return out;
}

Tensor add(const Tensor& a, Scalar s) {
    Tensor out;
    ops::add_out(a, wrapped_scalar_tensor(s), out);
    return out;
}


Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor out;
    ops::sub_out(a, b, out);
    return out;
}

Tensor sub(const Tensor& a, Scalar s) {
    Tensor out;
    ops::sub_out(a, wrapped_scalar_tensor(s), out);
    return out;
}


Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out;
    ops::mul_out(a, b, out);
    return out;
}

Tensor mul(const Tensor& a, Scalar s) {
    Tensor out;
    ops::mul_out(a, wrapped_scalar_tensor(s), out);
    return out;
}


Tensor div(const Tensor& a, const Tensor& b) {
    Tensor out;
    ops::div_out(a, b, out);
    return out;
}

Tensor div(const Tensor& a, Scalar s) {
    Tensor out;
    ops::div_out(a, wrapped_scalar_tensor(s), out);
    return out;
}

// -- Reversed
Tensor rsub(const Tensor& a, const Tensor& b) {
    Tensor out;
    ops::sub_out(b, a, out);
    return out;
}

Tensor rsub(const Tensor& a, Scalar s) {
    Tensor out;
    ops::sub_out(wrapped_scalar_tensor(s), a, out);
    return out;
}


Tensor rdiv(const Tensor& a, const Tensor& b) {
    Tensor out;
    ops::div_out(b, a, out);
    return out;
}

Tensor rdiv(const Tensor& a, Scalar s) {
    Tensor out;
    ops::div_out(wrapped_scalar_tensor(s), a, out);
    return out;
}


// In-place
Tensor& add_(Tensor& a, const Tensor& b) {
    return ops::add_out(a, b, a);
}

Tensor& add_(Tensor& a, Scalar s) {
    return ops::add_out(a, wrapped_scalar_tensor(s), a);
}


Tensor& sub_(Tensor& a, const Tensor& b) {
    return ops::sub_out(a, b, a);
}

Tensor& sub_(Tensor& a, Scalar s) {
    return ops::sub_out(a, wrapped_scalar_tensor(s), a);
}


Tensor& mul_(Tensor& a, const Tensor& b) {
    return ops::mul_out(a, b, a);
}

Tensor& mul_(Tensor& a, Scalar s) {
    return ops::mul_out(a, wrapped_scalar_tensor(s), a);
}


Tensor& div_(Tensor& a, const Tensor& b) {
    return ops::div_out(a, b, a);
}

Tensor& div_(Tensor& a, Scalar s) {
    return ops::div_out(a, wrapped_scalar_tensor(s), a);
}

// -- Reversed
Tensor& rsub_(Tensor& a, const Tensor& b) {
    return ops::sub_out(b, a, a);
}

Tensor& rsub_(Tensor& a, Scalar s) {
    return ops::sub_out(wrapped_scalar_tensor(s), a, a);
}


Tensor& rdiv_(Tensor& a, const Tensor& b) {
    return ops::div_out(b, a, a);
}

Tensor& rdiv_(Tensor& a, Scalar s) {
    return ops::div_out(wrapped_scalar_tensor(s), a, a);
}


Tensor& maximum_out(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    maximum_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor& minimum_out(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    minimum_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor maximum(const Tensor& a, const Tensor& b) {
    Tensor out;
    return ops::maximum_out(a, b, out);
}

Tensor minimum(const Tensor& a, const Tensor& b) {
    Tensor out;
    return ops::minimum_out(a, b, out);
}

} // namespace ops

DEFINE_DISPATCH(add_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(div_stub);

DEFINE_DISPATCH(maximum_stub);
DEFINE_DISPATCH(minimum_stub);

namespace {
 
template<typename Stub>
Tensor& comparison_op_out(const Tensor& a, const Tensor& b, Tensor& out, Stub& stub) {
    // The kernels write bools: an out of another dtype gets them converted
    // (PyTorch's kernels compute in the out dtype instead)
    if (out.defined() && !out.dtype().is_bool()) {
        Tensor result;
        comparison_op_out(a, b, result, stub);
        resize_output(out, result.shape());
        copy(result, out);
        return out;
    }
    auto iter = iterator::comparison_op_config()
        .resize_outputs(true)
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

template<typename Stub>
Tensor comparison_op(const Tensor& a, const Tensor& b, Stub& stub) {
    Tensor out;
    return comparison_op_out(a, b, out, stub);
}

// The out= variant of a non-differentiable op: only the version moves
Tensor& written(Tensor& out) {
    autograd::bump_version(out);
    return out;
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

Tensor& gt_out(const Tensor& a, const Tensor& b, Tensor& out) { return written(comparison_op_out(a, b, out, gt_stub)); }
Tensor& gt_out(const Tensor& a, Scalar s, Tensor& out)        { return gt_out(a, wrapped_scalar_tensor(s), out); }
Tensor& ge_out(const Tensor& a, const Tensor& b, Tensor& out) { return written(comparison_op_out(a, b, out, ge_stub)); }
Tensor& ge_out(const Tensor& a, Scalar s, Tensor& out)        { return ge_out(a, wrapped_scalar_tensor(s), out); }
Tensor& lt_out(const Tensor& a, const Tensor& b, Tensor& out) { return written(comparison_op_out(a, b, out, lt_stub)); }
Tensor& lt_out(const Tensor& a, Scalar s, Tensor& out)        { return lt_out(a, wrapped_scalar_tensor(s), out); }
Tensor& le_out(const Tensor& a, const Tensor& b, Tensor& out) { return written(comparison_op_out(a, b, out, le_stub)); }
Tensor& le_out(const Tensor& a, Scalar s, Tensor& out)        { return le_out(a, wrapped_scalar_tensor(s), out); }
Tensor& eq_out(const Tensor& a, const Tensor& b, Tensor& out) { return written(comparison_op_out(a, b, out, eq_stub)); }
Tensor& eq_out(const Tensor& a, Scalar s, Tensor& out)        { return eq_out(a, wrapped_scalar_tensor(s), out); }
Tensor& ne_out(const Tensor& a, const Tensor& b, Tensor& out) { return written(comparison_op_out(a, b, out, ne_stub)); }
Tensor& ne_out(const Tensor& a, Scalar s, Tensor& out)        { return ne_out(a, wrapped_scalar_tensor(s), out); }

Tensor logical_and(const Tensor& a, const Tensor& b) { return comparison_op(a, b, logical_and_stub); }
Tensor logical_or(const Tensor& a, const Tensor& b)  { return comparison_op(a, b, logical_or_stub); }

Tensor& logical_and_out(const Tensor& a, const Tensor& b, Tensor& out) {
    return written(comparison_op_out(a, b, out, logical_and_stub));
}

Tensor& logical_or_out(const Tensor& a, const Tensor& b, Tensor& out) {
    return written(comparison_op_out(a, b, out, logical_or_stub));
}


DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(ge_stub);
DEFINE_DISPATCH(lt_stub);
DEFINE_DISPATCH(le_stub);
DEFINE_DISPATCH(eq_stub);
DEFINE_DISPATCH(ne_stub);
DEFINE_DISPATCH(logical_and_stub);
DEFINE_DISPATCH(logical_or_stub);

}