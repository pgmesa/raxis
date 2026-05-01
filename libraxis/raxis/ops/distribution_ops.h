#pragma once

#include <raxis/export.h>

#include <raxis/ops/dispatch_stub.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_options.h>


namespace raxis {

class Generator;
struct TensorIterator;

using uniform_fn = void(*)(TensorIterator&, double, double, Generator&);
using normal_fn  = void(*)(TensorIterator&, double, double, Generator&);
using randint_fn = void(*)(TensorIterator&, int64_t, uint64_t, Generator&);


DECLARE_DISPATCH(uniform_fn, uniform_stub);
DECLARE_DISPATCH(normal_fn,  normal_stub);
DECLARE_DISPATCH(randint_fn, randint_stub);

/// Fills tensor with values drawn from U[low, high).
RAXIS_API Tensor& uniform_(Tensor& self, double low = 0.0, double high = 1.0);

/// Fills tensor with values drawn from N(mean, std^2).
RAXIS_API Tensor& normal_(Tensor& self, double mean = 0.0, double std = 1.0);

/// Fills tensor with integers drawn from U[low, high).
RAXIS_API Tensor& randint_(Tensor& self, int64_t low, int64_t high);

} // namespace raxis