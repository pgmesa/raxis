#pragma once

#include <vector>
#include <cstddef>
#include <optional>

#include <raxis/export.h>
#include <raxis/core/types.h>


namespace raxis {

RAXIS_API int64_t compute_numel(IntArrayRef shape) noexcept;
RAXIS_API int64_t compute_numel_from_strides(IntArrayRef shape, IntArrayRef strides);
RAXIS_API DimVector compute_strides(IntArrayRef shape);
RAXIS_API bool compute_contiguous(IntArrayRef shape, IntArrayRef strides) noexcept;
RAXIS_API bool compute_non_overlapping_and_dense(IntArrayRef shape, IntArrayRef strides);
RAXIS_API DimVector infer_dense_strides(IntArrayRef shape, IntArrayRef strides);
RAXIS_API DimVector broadcast_shapes(IntArrayRef shape1, IntArrayRef shape2);
RAXIS_API DimVector broadcast_shapes(ArrayRef<IntArrayRef> shapes);

RAXIS_API std::optional<DimVector> compute_view_strides(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    IntArrayRef newshape
);
RAXIS_API DimVector resolve_shape(IntArrayRef shape, int64_t numel);

RAXIS_API int64_t wrap_dim(int64_t dim, int64_t ndim);
RAXIS_API int64_t wrap_dim_allow_upper(int64_t dim, int64_t ndim);
RAXIS_API int64_t wrap_index(int64_t index, int64_t dim_size);
    
}