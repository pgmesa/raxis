#include <raxis/ops/dropout.h>

#include <raxis/util/exception.h>
#include <raxis/tensor/factories.h>
#include <raxis/ops/binary_ops.h>


namespace raxis {

// Composite: the mask does not require grad, so autograd only sees
// input * mask and the gradient is grad * mask.
// From: aten/src/ATen/native/Dropout.cpp
Tensor dropout(const Tensor& input, double p, bool train) {
    RAXIS_CHECK(p >= 0.0 && p <= 1.0,
        "dropout probability has to be between 0 and 1, but got ", p);
    if (!train || p == 0.0) {
        return input;
    }
    if (p == 1.0) {
        return input * 0.0;
    }
    // Keep where u >= p, u ~ U[0, 1): P(keep) = 1 - p. The 1 / (1 - p) scale
    // is folded into the mask so the forward is a single multiply.
    Tensor keep = rand_like(input) >= p;
    Tensor mask = keep.to(input.dtype()) * (1.0 / (1.0 - p));
    return input * mask;
}

} // namespace raxis
