
#include <raxis/array_utils.hpp>

#include <vector>


namespace raxis {

    /**
     * @brief Calculate strides (row-major)
     * 
     * @param shape 
     * @return IntArray 
     */
    IntArray compute_strides(const IntArray& shape) {
        IntArray strides(shape.size());
        
        int64_t s = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shape[i];
        }
        return strides;
    }

    // IntArray broadcast_shapes(const IntArray& shape1, const IntArray& shape2) {
        
    // }

    // IntArray broadcast_strides(const IntArray& shape, const IntArray& strides, const IntArray& target_shape) {

    // }

    // bool can_collapse_stride(int s_prev, int s_cur, size_t dim) {

    // }

    // void collapse_dim(const IntArray& shape, IntArray& sa, IntArray& sb, IntArray& so) {

    // }
}
