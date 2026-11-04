#pragma once

#include <raxis/export.h>
#include <raxis/core/types.h>


namespace raxis {

class Tensor;

// From: aten/src/ATen/native/Resize.cpp (resize_)
// Gives self a new shape, contiguous, over its storage and offset. The
// storage grows when it is too small, keeping its old bytes (every tensor
// over it sees the new memory), and never shrinks: resize_(0) keeps the
// memory for the next resize. Not for tensors that require grad, nor for
// memory raxis does not own (a NumPy array).
RAXIS_API const Tensor& resize_(const Tensor& self, IntArrayRef shape);

// From: aten/src/ATen/native/Resize.cpp (resize_output)
// Gives `output` the shape an op with out= computed, and strides when they
// are given (contiguous otherwise). Unlike PyTorch, which resizes any output
// and warns that it will stop doing so (pytorch#41027), an output with
// elements and another shape is an error here, as PyTorch plans. An output
// with zero elements is resized in place, reusing its storage when it is
// large enough (out=t after t.resize_(0) allocates nothing); if it is a
// view, it gets new memory instead, so it never writes into its base.
RAXIS_API void resize_output(const Tensor& output, IntArrayRef shape, IntArrayRef strides = {});

} // namespace raxis
