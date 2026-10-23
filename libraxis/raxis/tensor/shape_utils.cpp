
#include <raxis/tensor/shape_utils.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>
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

// From: c10/core/Contiguity.h (_compute_contiguous)
bool compute_contiguous(IntArrayRef shape, IntArrayRef strides) noexcept {
    // A tensor without elements is contiguous whatever its strides
    for (int64_t size : shape) {
        if (size == 0) return true;
    }

    int64_t expected_stride = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        // Size 1 dims never advance, so their stride is irrelevant
        if (shape[i] == 1) continue;
        if (strides[i] != expected_stride) return false;
        expected_stride *= shape[i];
    }
    return true;
}

// From: c10/core/Contiguity.h (_compute_non_overlapping_and_dense)
// True when the elements fill a block of memory without gaps or repeats, in
// any dimension order: contiguous after some permutation of the dims (e.g. a
// transposed or permuted contiguous tensor). Sorting the dims by stride must
// give the strides of a contiguous tensor.
bool compute_non_overlapping_and_dense(IntArrayRef shape, IntArrayRef strides) {
    size_t ndim = shape.size();
    if (ndim == 1) {
        return shape[0] < 2 || strides[0] == 1;
    }

    DimVector perm(ndim);
    std::iota(perm.begin(), perm.end(), 0);
    // Sort by stride, leaving the size 0 and 1 dims at the end
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        if (shape[a] < 2) return false;
        if (shape[b] < 2) return true;
        return strides[a] < strides[b];
    });

    int64_t required_stride = 1;
    for (int64_t dim : perm) {
        if (shape[dim] < 2) return true;
        if (strides[dim] != required_stride) return false;
        required_stride *= shape[dim];
    }
    return true;
}

// From: aten/src/ATen/ExpandUtils.cpp
// Dense strides that keep the dimension order of the given strides: a tensor
// with gaps or repeats (a slice, a broadcast view) gets a dense layout that
// is still "transposed" if the input was. Dims are sorted exactly like
// TensorIterator::reorder_dimensions does, so that both propagate the same
// layout.
DimVector infer_dense_strides(IntArrayRef shape, IntArrayRef strides) {
    RAXIS_CHECK(shape.size() == strides.size(),
        "Input shape and strides should have the same size but got ",
        shape.size(), " and ", strides.size());

    size_t ndim = shape.size();
    if (ndim == 0) return {};
    if (ndim == 1) return {1};

    // initialize perm with n-1, n-2, ..., 1, 0
    DimVector perm(ndim);
    std::iota(perm.rbegin(), perm.rend(), 0);

    // Returns 1 if dim0 should come after dim1, -1 if before, and 0 if the
    // comparison is ambiguous (broadcast dims never move)
    auto should_swap = [&](int64_t dim0, int64_t dim1) {
        int64_t stride0 = strides[dim0];
        int64_t stride1 = strides[dim1];
        if (stride0 == 0 || stride1 == 0) return 0;
        if (stride0 < stride1) return -1;
        if (stride0 > stride1) return 1;
        // For equal strides, the dim with the smaller size goes first
        return shape[dim0] > shape[dim1] ? 1 : 0;
    };

    // Insertion sort with support for ambiguous comparisons
    for (size_t i = 1; i < ndim; i++) {
        size_t dim1 = i;
        for (size_t dim0 = i; dim0-- > 0;) {
            int comparison = should_swap(perm[dim0], perm[dim1]);
            if (comparison > 0) {
                std::swap(perm[dim0], perm[dim1]);
                dim1 = dim0;
            } else if (comparison < 0) {
                break;
            }
        }
    }

    // perm[0] is the innermost dim
    DimVector out_strides(ndim);
    int64_t stride = 1;
    for (int64_t dim : perm) {
        out_strides[dim] = stride;
        // A size 0 dim is treated as 1: the tensor has no elements anyway
        if (shape[dim] > 1) stride *= shape[dim];
    }
    return out_strides;
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
        
        // Not max(size1, size2): a size 1 broadcasts to 0 too
        // From: aten/src/ATen/ExpandUtils.cpp (infer_size_impl)
        result_shape[target_ndim - ridx] = size1 == 1 ? size2 : size1;
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
    RAXIS_CHECK_INDEX(dim >= 0 && dim < ndim,
        "Dimension out of range (expected to be in range of [",
        -ndim, ", ", ndim - 1, "], but got ", dim, ")");
    return dim;
}

int64_t wrap_dim_allow_upper(int64_t dim, int64_t ndim) {
    int64_t upper = ndim + 1;
    if (dim < 0) dim += upper;
    RAXIS_CHECK_INDEX(dim >= 0 && dim < upper,
        "Dimension out of range (expected to be in range of [",
        -upper, ", ", ndim, "], but got ", dim, ")");
    return dim;
}

int64_t wrap_index(int64_t index, int64_t dim_size) {
    if (index < 0) index += dim_size;
    RAXIS_CHECK_INDEX(index >= 0 && index < dim_size,
        "Index ", index, " is out of bounds for dimension with size ", dim_size);
    return index;
}


/// Resolve a proposed shape containing at most one -1 into concrete sizes.
/// E.g. infer_size({2, -1, 4}, 24) -> {2, 3, 4}
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
