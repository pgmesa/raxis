
#include <raxis/ops/pow.h>

#include <raxis/ops/scalar_ops.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace ops {

Tensor pow(const Tensor& base, const Tensor& exp) {
    Tensor out;
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(base)
        .add_input(exp)
        .build();

    pow_tensor_tensor_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor pow(const Tensor& base, Scalar exp) {
    Tensor out;
    auto iter = iterator::unary_float_op_config()
        .add_output(out)
        .add_input(base)
        .build();

    if (exp.to<double>() == 0.0) {
        iter.get_output(0).fill_(1);
    } else if (exp.to<double>() == 1.0) {
        iter.get_output(0).copy_(base);
    } else {
        pow_tensor_scalar_stub(iter.device_type(), iter, exp);
    }
    return iter.get_output(0);
}

Tensor pow(Scalar base, const Tensor& exp) {
    if (base.to<double>() == 1.0) {
        Tensor out;
        auto iter = iterator::unary_float_op_config()
            .add_output(out)
            .add_input(exp)
            .build();

        iter.get_output(0).fill_(1);
        return iter.get_output(0);
    }
    return ops::pow(wrapped_scalar_tensor(base), exp);
}

} // namespace ops

DEFINE_DISPATCH(pow_tensor_tensor_stub);
DEFINE_DISPATCH(pow_tensor_scalar_stub);

} // namespace raxis