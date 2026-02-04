
#include <raxis/array_ops.hpp>

#include <iostream>

#include <fmt/format.h>
#include <raxis/array.hpp>
#include <raxis/factories.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/format/array_fmt.hpp>
#include <raxis/core/array_iterator.hpp>
#include <raxis/kernels/cpu/binary_ops.hpp>


namespace raxis {

NDArray add(const NDArray& a, const NDArray& b) {
    NDArray out;

    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    // iter.print_state();
    kernels::cpu::add_kernel(iter);
    return out;
}

NDArray sub(const NDArray& a, const NDArray& b) {
    if (a.dtype().is_bool() && b.dtype().is_bool()) {
        std::runtime_error(
            "Subtraction, the `-` operator, with two bool arrays is not supported."
        );
    }

    NDArray out;

    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    kernels::cpu::sub_kernel(iter);
    return out;
}

NDArray mul(const NDArray& a, const NDArray& b) {
    NDArray out;

    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    kernels::cpu::mul_kernel(iter);
    return out;
}

NDArray div(const NDArray& a, const NDArray& b) {
    NDArray out;
    
    auto iter = iterator::binary_float_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    kernels::cpu::div_kernel(iter);
    return out;
}
    
}