#pragma once

#include <vector>
#include <cstddef>
#include <optional>

#include <raxis/export.h>
#include <raxis/core/types.h>


namespace raxis {

RAXIS_API int64_t compute_numel(const IntArray& shape) noexcept;
RAXIS_API int64_t compute_numel_from_strides(const IntArray& shape, const IntArray& strides);
RAXIS_API IntArray compute_strides(const IntArray& shape);
RAXIS_API IntArray broadcast_shapes(const IntArray& shape1, const IntArray& shape2);
RAXIS_API IntArray broadcast_shapes(const std::vector<IntArray>& shapes);

RAXIS_API std::optional<IntArray> compute_view_strides(
    const IntArray& oldshape,
    const IntArray& oldstride,
    const IntArray& newshape
);
RAXIS_API IntArray resolve_shape(const IntArray& shape, int64_t numel);

RAXIS_API int64_t wrap_dim(int64_t dim, int64_t ndim);
RAXIS_API int64_t wrap_dim_allow_upper(int64_t dim, int64_t ndim);
RAXIS_API int64_t wrap_index(int64_t index, int64_t dim_size);
    
}