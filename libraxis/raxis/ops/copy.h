#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

struct TensorIterator;
using copy_fn = void(*)(TensorIterator&);
DECLARE_DISPATCH(copy_fn, copy_stub);

RAXIS_API Tensor& copy(const Tensor& a, Tensor& out);

// self.copy_(src) with autograd (CopyBackwards)
RAXIS_API Tensor& copy_(Tensor& self, const Tensor& src);

RAXIS_API Tensor clone(const Tensor& self, MemoryFormat memory_format = MemoryFormat::Preserve);
// self.to(dtype, device) with autograd (ToCopyBackward0)
RAXIS_API Tensor to_copy(const Tensor& self, Dtype dtype, Device device);

namespace ops {

RAXIS_API Tensor clone(const Tensor& self, MemoryFormat memory_format = MemoryFormat::Preserve);
RAXIS_API Tensor to_copy(const Tensor& self, Dtype dtype, Device device);

} // namespace ops

}