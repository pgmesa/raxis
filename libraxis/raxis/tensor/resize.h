#pragma once

#include <raxis/export.h>
#include <raxis/core/types.h>


namespace raxis {

class Tensor;

// From: aten/src/ATen/native/Resize.cpp (resize_output)
// Gives `output` the shape an op with out= computed, and strides when they
// are given (contiguous otherwise). Warns if it had elements, as PyTorch: a
// non-empty out of the wrong shape is usually a mistake. Unlike PyTorch's
// resize_, which reuses the storage when it is large enough, the output
// always gets new memory, so views taken of it before do not see the result.
RAXIS_API void resize_output(const Tensor& output, IntArrayRef shape, IntArrayRef strides = {});

} // namespace raxis
