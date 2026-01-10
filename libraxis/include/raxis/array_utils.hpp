#ifndef ARRAY_UTILS_HPP
#define ARRAY_UTILS_HPP

#include <vector>

#include <raxis/array.hpp>


namespace raxis{

    IntArray compute_strides(const IntArray& shape);
    // IntArray broadcast_shapes(const IntArray& shape1, const IntArray& shape2);
    // IntArray broadcast_strides(const IntArray& shape, const IntArray& strides, const IntArray& target_shape);

    // bool can_collapse_stride(int s_prev, int s_cur, size_t dim);
    // void collapse_dim(const IntArray& shape, IntArray& sa, IntArray& sb, IntArray& so);
}


#endif