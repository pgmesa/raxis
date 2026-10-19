#pragma once

#include <raxis/export.h>
#include <raxis/core/scalar.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

using scatter_gather_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
using scatter_fill_fn = void(*)(const Tensor& self, int64_t dim, const Tensor& index, Scalar value);

DECLARE_DISPATCH(scatter_gather_fn, gather_stub);
DECLARE_DISPATCH(scatter_gather_fn, scatter_stub);
DECLARE_DISPATCH(scatter_fill_fn, scatter_fill_stub);
DECLARE_DISPATCH(scatter_gather_fn, scatter_add_stub);

namespace ops {

RAXIS_API Tensor gather(const Tensor& self, int64_t dim, const Tensor& index);

RAXIS_API Tensor& scatter_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
RAXIS_API Tensor& scatter_(Tensor& self, int64_t dim, const Tensor& index, Scalar value);
RAXIS_API Tensor& scatter_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
RAXIS_API Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
RAXIS_API Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, Scalar value);
RAXIS_API Tensor scatter_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);

RAXIS_API Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index);
RAXIS_API Tensor& index_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source);
RAXIS_API Tensor index_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source);

} // namespace ops

RAXIS_API Tensor gather(const Tensor& self, int64_t dim, const Tensor& index);

RAXIS_API Tensor& scatter_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
RAXIS_API Tensor& scatter_(Tensor& self, int64_t dim, const Tensor& index, Scalar value);
RAXIS_API Tensor& scatter_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
RAXIS_API Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);
RAXIS_API Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, Scalar value);
RAXIS_API Tensor scatter_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src);

RAXIS_API Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index);
RAXIS_API Tensor& index_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source);
RAXIS_API Tensor index_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source);

} // namespace raxis
