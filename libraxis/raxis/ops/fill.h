#pragma once

#include <raxis/export.h>
#include <raxis/core/scalar.h>
#include <raxis/core/dtype.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

class TensorIterator;
using fill_fn = void(*)(TensorIterator&, Scalar);
DECLARE_DISPATCH(fill_fn, fill_stub);

RAXIS_API Tensor& fill_(Tensor& self, Scalar value);
RAXIS_API Tensor& zero_(Tensor& a);

}