#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

struct TensorIterator;
// non_blocking: a copy between pinned CPU memory and the GPU returns without
// waiting for the transfer (the other copies ignore it)
using copy_fn = void(*)(TensorIterator&, bool non_blocking);
DECLARE_DISPATCH(copy_fn, copy_stub);

RAXIS_API Tensor& copy(const Tensor& a, Tensor& out, bool non_blocking = false);

// self.copy_(src) with autograd (CopyBackwards)
RAXIS_API Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking = false);

RAXIS_API Tensor clone(const Tensor& self, MemoryFormat memory_format = MemoryFormat::Preserve);
// self.to(dtype, device) with autograd (ToCopyBackward0)
RAXIS_API Tensor to_copy(const Tensor& self, Dtype dtype, Device device, bool non_blocking = false);
// A copy of a CPU tensor in pinned (page-locked) memory, with autograd
// (PinMemoryBackward0); self when it already is pinned
RAXIS_API Tensor pin_memory(const Tensor& self);
RAXIS_API bool is_pinned(const Tensor& self);

namespace ops {

RAXIS_API Tensor clone(const Tensor& self, MemoryFormat memory_format = MemoryFormat::Preserve);
RAXIS_API Tensor to_copy(const Tensor& self, Dtype dtype, Device device, bool non_blocking = false);
RAXIS_API Tensor pin_memory(const Tensor& self);

} // namespace ops

}