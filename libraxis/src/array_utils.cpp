
#include <raxis/array_utils.hpp>

#include <vector>
#include <string>
#include <format>
#include <exception>


namespace raxis {

    /**
     * @brief Calculate strides (row-major)
     * 
     * @param shape 
     * @return IntArray 
     */
    IntArray compute_strides(const IntArray& shape) {
        IntArray strides(shape.size());
        
        Dim s = 1;
        for (Dim i = shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shape[i];
        }
        return strides;
    }

    IntArray broadcast_shapes(const IntArray& shape1, const IntArray& shape2) {
        size_t ndim1 = shape1.size();
        size_t ndim2 = shape2.size();
        size_t target_ndim = std::max(ndim1, ndim2);
        
        IntArray result_shape(target_ndim);

        for (size_t i = 0; i < target_ndim; i++) {
            size_t ridx = 1 + i;
            Dim dim1 = i < ndim1? shape1[ndim1 - ridx] : 1;
            Dim dim2 = i < ndim2? shape2[ndim2 - ridx] : 1;
            
            if (dim1 != 1 && dim2 != 1 && dim1 != dim2) {
                std::string msg = std::format(
                    "Shapes {} and {} are not broadcastable at dimension {}",
                    shape1, shape2, i
                );
                throw std::runtime_error(msg);
            } 

            result_shape[target_ndim - ridx] = std::max(dim1, dim2);
        }

        return result_shape;
    }

    IntArray broadcast_strides(
        const IntArray& shape, 
        const IntArray& strides, 
        const IntArray& target_shape
    ) {
        if (shape.size() > target_shape.size()) {
            throw std::invalid_argument("shape has more dims than target_shape");
        }
        
        size_t ndim = target_shape.size();
        size_t offset = ndim - shape.size();
        IntArray b_strides(ndim);


        for (size_t i = 0; i < ndim; i++) {
            if (i < offset || shape[i - offset] == 1) {
                b_strides[i] = 0;
            } else {
                b_strides[i] = strides[i - offset];
            }
        }

        return b_strides;
    }

    // bool can_collapse_stride(int s_prev, int s_cur, size_t dim) {

    // }

    // void collapse_dim(const IntArray& shape, IntArray& sa, IntArray& sb, IntArray& so) {

    // }
}
