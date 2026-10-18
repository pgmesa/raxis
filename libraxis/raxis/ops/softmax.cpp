#include <raxis/ops/softmax.h>

#include <raxis/util/exception.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/ops/unary_ops.h>
#include <raxis/ops/binary_ops.h>
#include <raxis/ops/reduce_ops.h>


namespace raxis {

namespace {

// A 0-d tensor is treated as a 1-element vector (as in PyTorch), so dim must
// be 0 or -1 and the result is 1 (softmax) or 0 (log_softmax).
template <typename Stub>
Tensor softmax_impl(const Tensor& self, int64_t dim, Stub& stub, const char* name) {
    RAXIS_CHECK(self.dtype().is_floating(),
        name, ": expected a floating point tensor, but got ", self.dtype());
    int64_t ndim = std::max<int64_t>(self.ndim(), 1);
    dim = wrap_dim(dim, ndim);

    Tensor input = self.contiguous();
    Tensor result = empty(self.shape(), self.options());
    if (result.numel() == 0) {
        return result;
    }
    if (self.ndim() == 0) {
        stub(self.device().type(), result.reshape({1}), input.reshape({1}), 0);
        return result;
    }
    stub(self.device().type(), result, input, dim);
    return result;
}

} // namespace

namespace ops {

Tensor softmax(const Tensor& self, int64_t dim) {
    return softmax_impl(self, dim, softmax_stub, "softmax");
}

Tensor log_softmax(const Tensor& self, int64_t dim) {
    return softmax_impl(self, dim, log_softmax_stub, "log_softmax");
}

// Composed from existing ops for now (3-4 kernel launches instead of one
// fused loop); fusing them is a pending optimization.

// d softmax: y * (g - sum(g * y))
Tensor softmax_backward_data(const Tensor& grad, const Tensor& output, int64_t dim) {
    if (output.ndim() == 0) {
        return zeros_like(output);
    }
    Tensor dot = ops::sum(grad * output, {dim}, /*keepdim=*/true);
    return output * (grad - dot);
}

// d log_softmax: g - exp(y) * sum(g)
Tensor log_softmax_backward_data(const Tensor& grad, const Tensor& output, int64_t dim) {
    if (output.ndim() == 0) {
        return zeros_like(output);
    }
    Tensor grad_sum = ops::sum(grad, {dim}, /*keepdim=*/true);
    return grad - ops::exp(output) * grad_sum;
}

} // namespace ops

DEFINE_DISPATCH(softmax_stub);
DEFINE_DISPATCH(log_softmax_stub);

} // namespace raxis
