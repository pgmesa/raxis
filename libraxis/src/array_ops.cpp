
#include <raxis/array_ops.hpp>

#include <iostream>

#include <fmt/format.h>
#include <raxis/array.hpp>
#include <raxis/factories.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/format/array_fmt.hpp>
#include <raxis/core/array_iterator.hpp>
#include <raxis/kernels/cpu/loops.hpp>
#include <raxis/kernels/cpu/copy_kernel.hpp>
#include <raxis/kernels/cpu/binary_ops_kernel.hpp>


namespace raxis {

NDArray& copy(const NDArray& a, NDArray& out) {
    auto iter = ArrayIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_input(a)
        .build();
    
    kernels::cpu::copy_kernel(iter);
    out = iter.get_output(0);
    return out;
}

NDArray clone(const NDArray& a) {
    NDArray out;
    copy(a, out);
    return out;
}

NDArray to(const NDArray& a, Dtype dtype) {
    if (a.dtype() == dtype) return a;

    NDArray out = empty(a.shape(), dtype);
    copy(a, out);
    return out;
}

NDArray& add(const NDArray& a, const NDArray& b, NDArray& out) {
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    // iter.print_state();
    kernels::cpu::add_kernel(iter);

    out = iter.get_output(0);
    return out;
}

NDArray& sub(const NDArray& a, const NDArray& b, NDArray& out) {
    if (a.dtype().is_bool() && b.dtype().is_bool()) {
        std::runtime_error(
            "Subtraction, the `-` operator, with two bool arrays is not supported."
        );
    }

    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    kernels::cpu::sub_kernel(iter);
    out = iter.get_output(0);
    return out;
}

NDArray& mul(const NDArray& a, const NDArray& b, NDArray& out) {
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    kernels::cpu::mul_kernel(iter);
    out = iter.get_output(0);
    return out;
}

NDArray& div(const NDArray& a, const NDArray& b, NDArray& out) {
    auto iter = iterator::binary_float_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    kernels::cpu::div_kernel(iter);
    out = iter.get_output(0);
    return out;
}


NDArray add(const NDArray& a, const NDArray& b) {
    NDArray out;
    add(a, b, out);
    return out;
}

NDArray& add_(NDArray& a, const NDArray& b) {
    return add(a, b, a);
}


NDArray sub(const NDArray& a, const NDArray& b) {
    NDArray out;
    sub(a, b, out);
    return out;
}

NDArray& sub_(NDArray& a, const NDArray& b) {
    return sub(a, b, a);
}


NDArray mul(const NDArray& a, const NDArray& b) {
    NDArray out;
    mul(a, b, out);
    return out;
}

NDArray& mul_(NDArray& a, const NDArray& b) {
    return mul(a, b, a);
}


NDArray div(const NDArray& a, const NDArray& b) {
    NDArray out;
    div(a, b, out);
    return out;
}

NDArray& div_(NDArray& a, const NDArray& b) {
    return div(a, b, a);
}

// Reversed
NDArray rsub(const NDArray& a, const NDArray& b) {
    NDArray out;
    sub(b, a, out);
    return out;
}

NDArray rdiv(const NDArray& a, const NDArray& b) {
    NDArray out;
    div(b, a, out);
    return out;
}

NDArray& rsub_(NDArray& a, const NDArray& b) {
    return sub(b, a, a);
}

NDArray& rdiv_(NDArray& a, const NDArray& b) {
    return div(b, a, a);
}
    
}