
#include <raxis/ops/binary_ops.h>

#include <iostream>

#include <fmt/format.h>

#include <raxis/format.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_iterator.h>

#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/copy_kernel.h>
#include <raxis/ops/cpu/binary_ops_kernel.h>


namespace raxis {

Tensor& copy(const Tensor& a, Tensor& out) {
    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_input(a)
        .build();
    
    kernels::cpu::copy_kernel(iter);
    out = iter.get_output(0);
    return out;
}

Tensor clone(const Tensor& a) {
    Tensor out;
    copy(a, out);
    return out;
}

Tensor to(const Tensor& a, Dtype dtype) {
    if (a.dtype() == dtype) return a;

    Tensor out = empty(a.shape(), dtype);
    copy(a, out);
    return out;
}

// Tensor& fill_(Tensor& self, Scalar value) {
//     if (self.device() == CPU && self.numel() == 1) {
//         scalar_fill()
//     }

//     auto iter = TensorIteratorConfig()
//     .check_all_same_dtype(false)
//     .add_output(self)
//     .build();

//     kernels::fill();
//     return self;
// }

Tensor& add(const Tensor& a, const Tensor& b, Tensor& out) {
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

Tensor& sub(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.dtype().is_bool() && b.dtype().is_bool()) {
        std::runtime_error(
            "Subtraction, the `-` operator, with two bool tensors is not supported."
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

Tensor& mul(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    kernels::cpu::mul_kernel(iter);
    out = iter.get_output(0);
    return out;
}

Tensor& div(const Tensor& a, const Tensor& b, Tensor& out) {
    auto iter = iterator::binary_float_op_config()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    kernels::cpu::div_kernel(iter);
    out = iter.get_output(0);
    return out;
}


Tensor add(const Tensor& a, const Tensor& b) {
    Tensor out;
    add(a, b, out);
    return out;
}

Tensor& add_(Tensor& a, const Tensor& b) {
    return add(a, b, a);
}


Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor out;
    sub(a, b, out);
    return out;
}

Tensor& sub_(Tensor& a, const Tensor& b) {
    return sub(a, b, a);
}


Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor out;
    mul(a, b, out);
    return out;
}

Tensor& mul_(Tensor& a, const Tensor& b) {
    return mul(a, b, a);
}


Tensor div(const Tensor& a, const Tensor& b) {
    Tensor out;
    div(a, b, out);
    return out;
}

Tensor& div_(Tensor& a, const Tensor& b) {
    return div(a, b, a);
}

// Reversed
Tensor rsub(const Tensor& a, const Tensor& b) {
    Tensor out;
    sub(b, a, out);
    return out;
}

Tensor rdiv(const Tensor& a, const Tensor& b) {
    Tensor out;
    div(b, a, out);
    return out;
}

Tensor& rsub_(Tensor& a, const Tensor& b) {
    return sub(b, a, a);
}

Tensor& rdiv_(Tensor& a, const Tensor& b) {
    return div(b, a, a);
}
    
}