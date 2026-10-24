#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/array_ref.h>


namespace raxis {

struct TensorIterator;

// A copy from input 0 to the output; the iterator's output strides are
// already negated along the flipped dims (input 1 is a dummy operand)
using flip_fn = void(*)(TensorIterator& iter);
DECLARE_DISPATCH(flip_fn, flip_stub);

namespace ops {

RAXIS_API Tensor flip(const Tensor& self, IntArrayRef dims);

} // namespace ops

/// Reverses the order of the elements along dims (a copy, as in torch)
RAXIS_API Tensor flip(const Tensor& self, IntArrayRef dims);

} // namespace raxis
