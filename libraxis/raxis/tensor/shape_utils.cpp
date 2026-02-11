
#include <raxis/tensor/shape_utils.h>

#include <vector>
#include <string>
#include <stdexcept>

#include <fmt/format.h>

#include <raxis/format.h>
#include <raxis/core/types.h>


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
    if (shape.empty()) return 0;
    
    int64_t max_offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] > 1) {  // size 1 dimensions don't contribute
            max_offset += (shape[i] - 1) * strides[i];
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
        
        if (size1 != 1 && size2 != 1 && size1 != size2) {
            std::string msg = fmt::format(
                "Shapes {} and {} are not broadcastable at dimension {}",
                shape1, shape2, i
            );
            throw std::runtime_error(msg);
        } 

        result_shape[target_ndim - ridx] = std::max(size1, size2);
    }

    return result_shape;
}

}
