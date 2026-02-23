#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

class TensorIterator;
using copy_fn = void(*)(TensorIterator&);
DECLARE_DISPATCH(copy_fn, copy_stub);

RAXIS_API Tensor& copy(const Tensor& a, Tensor& out);
RAXIS_API Tensor clone(const Tensor& a);
RAXIS_API Tensor to(const Tensor& a, Dtype dtype);

}