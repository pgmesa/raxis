#pragma once

#include <vector>
#include <cstddef>

#include <raxis/api.hpp>
#include <raxis/array_types.hpp>


namespace raxis {

    RAXIS_API size_t compute_numel(const IntArray& shape) noexcept;
    RAXIS_API IntArray compute_strides(const IntArray& shape);
    RAXIS_API IntArray broadcast_shapes(const IntArray& shape1, const IntArray& shape2);
    
    RAXIS_API IntArray broadcast_strides(
        const IntArray& shape, 
        const IntArray& strides, 
        const IntArray& target_shape
    );

    RAXIS_API void collapse_dim(IntArray& shape, IntArray& sa, IntArray& sb, IntArray& so);
}