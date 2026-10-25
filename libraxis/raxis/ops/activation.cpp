#include <raxis/ops/activation.h>

#include <raxis/util/exception.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace ops {

Tensor relu(const Tensor& self) {
    RAXIS_CHECK(self.dtype() != dtypes::Bool, "relu: boolean tensors are not supported");
    Tensor out;
    auto iter = iterator::unary_op_config()
        .add_output(out)
        .add_input(self)
        .build();
    relu_stub(iter.device_type(), iter);
    return iter.get_output(0);
}

Tensor& relu_(Tensor& self) {
    RAXIS_CHECK(self.dtype() != dtypes::Bool, "relu_: boolean tensors are not supported");
    auto iter = iterator::unary_op_config()
        .add_output(self)
        .add_input(self)
        .build();
    relu_stub(iter.device_type(), iter);
    return self;
}

Tensor threshold_backward(const Tensor& grad, const Tensor& self, Scalar threshold) {
    Tensor out;
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(self)
        .add_input(grad)
        .build();
    threshold_backward_stub(iter.device_type(), iter, threshold);
    return iter.get_output(0);
}

Tensor gelu(const Tensor& self, GeluType approximate) {
    RAXIS_CHECK(self.dtype().is_floating(), "\"GeluKernelImpl\" not implemented for '", self.dtype(), "'");
    Tensor out;
    auto iter = iterator::unary_op_config()
        .add_output(out)
        .add_input(self)
        .build();
    gelu_stub(iter.device_type(), iter, approximate);
    return iter.get_output(0);
}

Tensor gelu_backward(const Tensor& grad, const Tensor& self, GeluType approximate) {
    Tensor out;
    auto iter = iterator::binary_op_config()
        .add_output(out)
        .add_input(grad)
        .add_input(self)
        .build();
    RAXIS_CHECK(iter.common_dtype().is_floating(),
        "\"GeluBackwardKernelImpl\" not implemented for '", iter.common_dtype(), "'");
    gelu_backward_stub(iter.device_type(), iter, approximate);
    return iter.get_output(0);
}

} // namespace ops

GeluType gelu_type_from_string(std::string_view approximate) {
    if (approximate == "none") {
        return GeluType::None;
    }
    RAXIS_CHECK(approximate == "tanh", "approximate argument must be either none or tanh.");
    return GeluType::Tanh;
}

DEFINE_DISPATCH(relu_stub);
DEFINE_DISPATCH(threshold_backward_stub);
DEFINE_DISPATCH(gelu_stub);
DEFINE_DISPATCH(gelu_backward_stub);

} // namespace raxis
