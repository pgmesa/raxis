#pragma once

#include <raxis/export.h>
#include <raxis/core/scalar.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/array_ref.h>


namespace raxis {

struct TensorIterator;

using index_fn = void(*)(TensorIterator& iter, IntArrayRef indexed_sizes, IntArrayRef indexed_strides);
// unique_indices: no position is written twice, so accumulate needs no serial loop
using index_put_fn = void(*)(TensorIterator& iter, IntArrayRef indexed_sizes, IntArrayRef indexed_strides,
                             bool accumulate, bool unique_indices);
using masked_fill_fn = void(*)(TensorIterator& iter, Scalar value);

DECLARE_DISPATCH(index_fn, index_stub);
DECLARE_DISPATCH(index_put_fn, index_put_stub);
DECLARE_DISPATCH(masked_fill_fn, masked_fill_stub);

// result.select(dim, i) = self.select(dim, index[i]); index is contiguous
// int64, result has self's shape with index.numel() in dim
using index_select_fn = void(*)(const Tensor& result, const Tensor& self, int64_t dim, const Tensor& index);
DECLARE_DISPATCH(index_select_fn, index_select_stub);

// The (n, ndim) positions of the nonzero elements of self, column-major.
// The result's size depends on the data, so the kernel allocates it
using nonzero_fn = Tensor(*)(const Tensor& self);
DECLARE_DISPATCH(nonzero_fn, nonzero_stub);

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

// Advanced indexing: indices[d] indexes dim d of self, an undefined tensor
// leaves the dim whole (x[:, idx] is {Tensor(), idx}). Long and int tensors
// pick positions, bool (or uint8) masks select where they are true.
RAXIS_API Tensor index(const Tensor& self, ArrayRef<Tensor> indices);
RAXIS_API Tensor& index_put_(Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate = false);
RAXIS_API Tensor index_put(const Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate = false);

/// Indices of the nonzero elements, one row per element: shape (n, ndim)
RAXIS_API Tensor nonzero(const Tensor& self);

RAXIS_API Tensor& masked_fill_(Tensor& self, const Tensor& mask, Scalar value);
RAXIS_API Tensor masked_fill(const Tensor& self, const Tensor& mask, Scalar value);

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

RAXIS_API Tensor index(const Tensor& self, ArrayRef<Tensor> indices);
RAXIS_API Tensor& index_put_(Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate = false);
RAXIS_API Tensor index_put(const Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate = false);

RAXIS_API Tensor nonzero(const Tensor& self);

RAXIS_API Tensor& masked_fill_(Tensor& self, const Tensor& mask, Scalar value);
RAXIS_API Tensor masked_fill(const Tensor& self, const Tensor& mask, Scalar value);

} // namespace raxis
