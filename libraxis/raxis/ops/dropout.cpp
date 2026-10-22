#include <raxis/ops/dropout.h>

#include <cstdint>

#include <raxis/core/generator.h>
#include <raxis/util/exception.h>
#include <raxis/tensor/factories.h>


namespace raxis {

namespace ops {

// PyTorch's CPU dropout is not this fused op: it fills a noise tensor with
// bernoulli_(1 - p), divides it by 1 - p and multiplies (MulBackward0), and
// bernoulli_ draws serially from mt19937 under the generator lock. Its CUDA
// dropout is this one: native_dropout with Philox, which parallelizes.
std::tuple<Tensor, Tensor> native_dropout(const Tensor& input_, double p, bool train) {
    RAXIS_CHECK(p >= 0.0 && p <= 1.0,
        "dropout probability has to be between 0 and 1, but got ", p);
    if (!train) {
        return {input_, ones(input_.shape(), input_.options().dtype(dtypes::Bool))};
    }
    RAXIS_CHECK(input_.dtype().is_floating(),
        "dropout: expected a floating point input, but got ", input_.dtype());

    const Tensor input = input_.contiguous();
    Tensor output = empty(input.shape(), input.options());
    Tensor mask = empty(input.shape(), input.options().dtype(dtypes::Bool));
    if (input.numel() > 0) {
        const auto [seed, offset] =
            get_default_generator().philox_engine_inputs(dropout_philox_blocks(input.numel()));
        fused_dropout_stub(DeviceType::CPU, output, mask, input, 1.0 - p, seed, offset);
    }
    return {output, mask};
}

Tensor native_dropout_backward(const Tensor& grad, const Tensor& mask, double scale) {
    const Tensor grad_c = grad.contiguous();
    Tensor grad_input = empty(grad_c.shape(), grad_c.options());
    if (grad_input.numel() > 0) {
        dropout_backward_stub(DeviceType::CPU, grad_input, grad_c, mask.contiguous(), scale);
    }
    return grad_input;
}

} // namespace ops

DEFINE_DISPATCH(fused_dropout_stub);
DEFINE_DISPATCH(dropout_backward_stub);

} // namespace raxis
