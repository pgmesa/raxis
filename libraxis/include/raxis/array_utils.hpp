#pragma once

#include <vector>

#include <raxis/array_types.hpp>


namespace raxis{

    IntArray compute_strides(const IntArray& shape);

    IntArray broadcast_shapes(const IntArray& shape1, const IntArray& shape2);
    
    IntArray broadcast_strides(
        const IntArray& shape, 
        const IntArray& strides, 
        const IntArray& target_shape
    );

    void collapse_dim(IntArray& shape, IntArray& sa, IntArray& sb, IntArray& so);
}