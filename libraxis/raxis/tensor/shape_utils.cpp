
#include <raxis/tensor/shape_utils.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <optional>

#include <raxis/core/types.h>
#include <raxis/util/exception.h>


namespace raxis {

int64_t compute_numel(IntArrayRef shape) noexcept {
    int64_t n = 1;
    for (int64_t d: shape) 
        n *= d;
    return n;
}

// Shape: (3, 4), Strides: (8, 2)
// Memory needed: max element offset + 1
// = (3-1)*8 + (4-1)*2 + 1 = 16 + 6 + 1 = 23 elements
int64_t compute_numel_from_strides(IntArrayRef shape, IntArrayRef strides) {
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

DimVector compute_strides(IntArrayRef shape) {
    DimVector strides(shape.size());
    
    int64_t s = 1;
    int ndim = static_cast<int>(shape.size());
    for (int i = ndim - 1; i >= 0; --i) {
        strides[i] = s;
        s *= shape[i];
    }
    return strides;
}

DimVector broadcast_shapes(IntArrayRef shape1, IntArrayRef shape2) {
    if (shape1 == shape2) {
        return DimVector(shape1);
    }

    size_t ndim1 = shape1.size();
    size_t ndim2 = shape2.size();
    size_t target_ndim = std::max(ndim1, ndim2);
    
    DimVector result_shape(target_ndim);

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

DimVector broadcast_shapes(ArrayRef<IntArrayRef> shapes) {
    RAXIS_CHECK(!shapes.empty(), "broadcast_shapes(): no shapes provided");
    DimVector result(shapes[0]);
    for (size_t i = 1; i < shapes.size(); ++i) {
        result = broadcast_shapes(result, shapes[i]);
    }
    return result;
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


/// Resolve a proposed shape containing at most one -1 into concrete sizes.
/// E.g. infer_size({2, -1, 4}, 24) → {2, 3, 4}
DimVector resolve_shape(IntArrayRef shape, int64_t numel) {
    int64_t inferred_dim = -1;
    int64_t product = 1;

    for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
        if (shape[i] == -1) {
            RAXIS_CHECK(inferred_dim == -1,
                "only one dimension can be inferred (found -1 at dims ",
                inferred_dim, " and ", i, ")");
            inferred_dim = i;
        } else {
            RAXIS_CHECK(shape[i] >= 0,
                "invalid shape dimension: ", shape[i]);
            product *= shape[i];
        }
    }

    DimVector result(shape);

    if (inferred_dim != -1) {
        RAXIS_CHECK(product != 0 && numel % product == 0,
            "shape ", shape, " is invalid for input of size ", numel);
        result[inferred_dim] = numel / product;
    } else {
        RAXIS_CHECK(product == numel,
            "shape ", shape, " is invalid for input of size ", numel);
    }

    return result;
}

/// Try to compute strides for viewing `(oldshape, oldstride)` as `newshape`.
/// Returns nullopt if the view is impossible (requires a copy).
///
/// Algorithm: walk both shapes right-to-left, grouping old dimensions into
/// contiguous chunks. For each chunk, consume new dimensions until the
/// element counts match. If they ever don't align, the view is impossible.
std::optional<DimVector> compute_view_strides(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    IntArrayRef newshape
) {
    // Scalar tensor -> any new shape gets all-ones strides
    if (oldshape.empty()) {
        return DimVector(newshape.size(), 1);
    }

    // Zero-element tensor: same shape -> copy strides, else use contiguous
    int64_t numel = std::accumulate(
        oldshape.begin(), oldshape.end(), int64_t{1}, std::multiplies<>());

    if (numel == 0) {
        if (oldshape == newshape) {
            return DimVector(oldstride);
        }
        // Contiguous strides for the new shape
        return compute_strides(newshape);
    }

    DimVector newstride(newshape.size());

    int64_t view_d = static_cast<int64_t>(newshape.size()) - 1;

    // Stride of the rightmost element in the current contiguous chunk
    int64_t chunk_base_stride = oldstride.back();
    // Accumulated numel within old and new chunks
    int64_t tensor_numel = 1;
    int64_t view_numel = 1;

    for (int64_t tensor_d = static_cast<int64_t>(oldshape.size()) - 1;
         tensor_d >= 0; --tensor_d)
    {
        tensor_numel *= oldshape[tensor_d];

        // Detect chunk boundary:
        //  - tensor_d == 0 (leftmost dim, always a boundary)
        //  - next dim to the left is NOT contiguous with current chunk
        //    (skip size-1 dims since their stride doesn't matter)
        bool is_chunk_boundary = (tensor_d == 0) ||
            (oldshape[tensor_d - 1] != 1 &&
             oldstride[tensor_d - 1] != tensor_numel * chunk_base_stride);

        if (!is_chunk_boundary) {
            continue;
        }

        // Consume new dims right-to-left until view_numel matches tensor_numel
        while (view_d >= 0 && (view_numel < tensor_numel || newshape[view_d] == 1)) {
            newstride[view_d] = view_numel * chunk_base_stride;
            view_numel *= newshape[view_d];
            --view_d;
        }

        // If the chunks don't align, view is impossible
        if (view_numel != tensor_numel) {
            return std::nullopt;
        }

        // Start a new chunk
        if (tensor_d > 0) {
            chunk_base_stride = oldstride[tensor_d - 1];
            tensor_numel = 1;
            view_numel = 1;
        }
    }

    // If we didn't consume all new dims, the view is impossible
    if (view_d != -1) {
        return std::nullopt;
    }

    return newstride;
}

}
