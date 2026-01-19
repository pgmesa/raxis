
#include <raxis/core/shape.hpp>

#include <vector>
#include <string>
#include <format>
#include <exception>

#include <raxis/array_types.hpp>


namespace raxis {

    size_t compute_numel(const IntArray& shape) noexcept {
        size_t n = 1;
        for (Int d: shape) 
            n *=static_cast<size_t>(d);
        return n;
    }

    IntArray compute_strides(const IntArray& shape) {
        IntArray strides(shape.size());
        
        Int s = 1;
        Int ndim = static_cast<Int>(shape.size());
        for (Int i = ndim - 1; i >= 0; --i) {
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
            Int dim1 = i < ndim1? shape1[ndim1 - ridx] : 1;
            Int dim2 = i < ndim2? shape2[ndim2 - ridx] : 1;
            
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
        if (shape == target_shape) return strides;

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

    bool can_collapse_stride(Int s_prev, Int s_cur, Int size_cur) noexcept {
        return (s_prev == (s_cur * size_cur)) || (s_prev == 0 && s_cur == 0);
    }

    void collapse_dim(IntArray& shape, IntArray& sa, IntArray& sb, IntArray& so) {
        for (Int dim = shape.size() - 1; dim > 0; --dim) {
            Int prev_dim = shape[dim];
            Int dim_size = shape[dim];
            bool can_collapse =
                can_collapse_stride(sa[dim-1], sa[dim], dim_size) &&
                can_collapse_stride(sb[dim-1], sb[dim], dim_size) &&
                can_collapse_stride(so[dim-1], so[dim], dim_size);

            if (can_collapse) {
                shape[dim - 1] *= shape[dim];

                sa[dim - 1] = sa[dim];
                sb[dim - 1] = sb[dim];
                so[dim - 1] = so[dim];

                shape.erase(shape.begin() + dim);
                sa.erase(sa.begin() + dim);
                sb.erase(sb.begin() + dim);
                so.erase(so.begin() + dim);
            }
        }
    }
}
