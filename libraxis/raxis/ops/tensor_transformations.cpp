#include <raxis/ops/tensor_transformations.h>

#include <bitset>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/tensor_shape.h>


namespace raxis {

namespace ops {

// From: aten/src/ATen/native/TensorTransformations.cpp (flip)
// No index arithmetic per element: the output is written backwards. The
// iterator is built as a plain copy, then the output pointer is moved to the
// last element of every flipped dim and its stride there is negated, the
// same as walking a cube from the opposite vertex. The dummy operand is self
// with stride 0 on the flipped dims: it keeps the iterator from coalescing a
// flipped dim with a non-flipped one (their strides no longer line up) and
// marks which iterator dims to flip after reordering and coalescing.
Tensor flip(const Tensor& self, IntArrayRef dims) {
    const int64_t ndim = self.ndim();
    std::bitset<64> flip_mask;
    for (int64_t d : dims) {
        d = wrap_dim(d, std::max<int64_t>(ndim, 1));
        RAXIS_CHECK(!flip_mask.test(d), "dim ", d, " appears multiple times in the list of dims");
        flip_mask.set(d);
    }

    Tensor result = empty_like(self);

    int64_t num_flipped = 0;
    DimVector dummy_strides(self.strides().begin(), self.strides().end());
    for (int64_t d = 0; d < ndim; ++d) {
        if (flip_mask.test(d) && self.shape()[d] > 1 && self.strides()[d] != 0) {
            num_flipped++;
            dummy_strides[d] = 0;
        }
    }
    if (num_flipped == 0 || self.numel() <= 1) {
        result.copy_(self);
        return result;
    }

    const Tensor dummy = make_view(
        self, self.storage_offset(), self.shape(), dummy_strides, /*is_differentiable_view=*/false);
    auto iter = TensorIteratorConfig()
        .check_mem_overlap(false)
        .add_output(result)
        .add_input(self)
        .add_input(dummy)
        .build();

    char* data = static_cast<char*>(iter.data_ptr(0));
    const IntArrayRef shape = iter.shape();
    DimVector result_strides(iter.strides(0).begin(), iter.strides(0).end());
    const IntArrayRef self_strides = iter.strides(1);
    const IntArrayRef dummy_strides_bytes = iter.strides(2);
    for (int d = 0; d < iter.ndim(); ++d) {
        // dummy stride 0 with a non-zero self stride: a flipped dim
        if (dummy_strides_bytes[d] == 0 && self_strides[d] != 0) {
            data += result_strides[d] * (shape[d] - 1);
            result_strides[d] = -result_strides[d];
        }
    }
    iter.unsafe_set_operand_strides(0, result_strides);
    iter.unsafe_replace_operand(0, data);

    flip_stub(iter.device_type(), iter);
    return result;
}

} // namespace ops

DEFINE_DISPATCH(flip_stub);

} // namespace raxis
