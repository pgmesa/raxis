
#include <raxis/tensor/shape_utils.h>

#include <vector>
#include <string>
#include <stdexcept>

#include <raxis/core/types.h>
#include <raxis/util/exception.h>


namespace raxis {

int64_t compute_numel(const IntArray& shape) noexcept {
    int64_t n = 1;
    for (int64_t d: shape) 
        n *= d;
    return n;
}

// Shape: (3, 4), Strides: (8, 2)
// Memory needed: max element offset + 1
// = (3-1)*8 + (4-1)*2 + 1 = 16 + 6 + 1 = 23 elements
int64_t compute_numel_from_strides(const IntArray& shape, const IntArray& strides) {
    if (shape.empty()) return 1;
    
    int64_t max_offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > 1) {  // size 1 dimensions don't contribute
            max_offset += (shape[i] - 1) * strides[i];
        } else if (shape[i] == 0) {
            return 0;
        }
    }
    return max_offset + 1;
}

IntArray compute_strides(const IntArray& shape) {
    IntArray strides(shape.size());
    
    int64_t s = 1;
    int ndim = static_cast<int>(shape.size());
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i];
    }
    return strides;
}

IntArray broadcast_shapes(const IntArray& shape1, const IntArray& shape2) {
    if (shape1 == shape2) return shape1;

    size_t ndim1 = shape1.size();
    size_t ndim2 = shape2.size();
    size_t target_ndim = std::max(ndim1, ndim2);
    
    IntArray result_shape(target_ndim);

    for (size_t i = 0; i < target_ndim; i++) {
        size_t ridx = 1 + i;
        int64_t size1 = i < ndim1? shape1[ndim1 - ridx] : 1;
        int64_t size2 = i < ndim2? shape2[ndim2 - ridx] : 1;
        
        bool broadcastable = size1 != 1 && size2 != 1 && size1 != size2;
        RAXIS_CHECK(!broadcastable, "Shapes ", shape1, " and ", shape2, 
            " are not broadcastable at dimension ", i);
        
        result_shape[target_ndim - ridx] = std::max(size1, size2);
    }

    return result_shape;
}

int64_t wrap_dim(int64_t dim, int64_t ndim) {
    RAXIS_CHECK(ndim > 0 || dim == 0,
        "wrap_dim: cannot use dim with a 0-d tensor");
    if (dim < 0) dim += ndim;
    RAXIS_CHECK(dim >= 0 && dim < ndim,
        "Dimension out of range (expected to be in range of [",
        -ndim, ", ", ndim - 1, "], but got ", dim, ")");
    return dim;
}

int64_t wrap_dim_allow_upper(int64_t dim, int64_t ndim) {
    int64_t upper = ndim + 1;
    if (dim < 0) dim += upper;
    RAXIS_CHECK(dim >= 0 && dim < upper,
        "Dimension out of range (expected to be in range of [",
        -upper, ", ", ndim, "], but got ", dim, ")");
    return dim;
}

int64_t wrap_index(int64_t index, int64_t dim_size) {
    if (index < 0) index += dim_size;
    RAXIS_CHECK(index >= 0 && index < dim_size,
        "Index ", index, " is out of bounds for dimension with size ", dim_size);
    return index;
}

}
