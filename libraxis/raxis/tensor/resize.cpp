#include <raxis/tensor/resize.h>

#include <raxis/util/warning.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/detail/tensor_impl.h>


namespace raxis {

void resize_output(const Tensor& output, IntArrayRef shape, IntArrayRef strides) {
    if (output.shape() == shape) return;

    if (output.numel() != 0) {
        RAXIS_WARN(
            "An output with one or more elements was resized since it had shape ", output.shape(),
            ", which does not match the required output shape ", shape, ". This behavior is "
            "deprecated, and in a future PyTorch release outputs will not be resized unless "
            "they have zero elements. You can explicitly reuse an out tensor t by resizing it, "
            "inplace, to zero elements with t.resize_(0).");
    }

    TensorOptions options = output.options().requires_grad(false);
    Tensor fresh = strides.empty() ? empty(shape, options) : empty_strided(shape, strides, options);
    output.unsafe_get_tensor_impl()->shallow_copy_from(*fresh.unsafe_get_tensor_impl());
}

} // namespace raxis
