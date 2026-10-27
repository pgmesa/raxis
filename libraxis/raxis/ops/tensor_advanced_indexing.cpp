#include <raxis/ops/tensor_advanced_indexing.h>

#include <algorithm>
#include <cstring>
#include <optional>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/tensor_shape.h>
#include <raxis/tensor/memory_overlap.h>

#include <raxis/ops/copy.h>
#include <raxis/ops/device_check.h>
#include <raxis/ops/binary_ops.h>


namespace raxis {

namespace {

// A 0-d tensor counts as 1-d with one element
int64_t ensure_nonempty_dim(int64_t ndim) {
    return std::max<int64_t>(ndim, 1);
}

int64_t ensure_nonempty_size(const Tensor& t, int64_t dim) {
    return t.ndim() == 0 ? 1 : t.shape()[dim];
}

// From: aten/src/ATen/native/ScatterGatherChecks.h
// index can be smaller than self (and src) in every dim but `dim`, where
// its values pick positions
void gather_shape_check(const Tensor& self, int64_t dim, const Tensor& index) {
    int64_t self_dims = ensure_nonempty_dim(self.ndim());
    RAXIS_CHECK(self_dims == ensure_nonempty_dim(index.ndim()),
        "Index tensor must have the same number of dimensions as input tensor");
    for (int64_t d = 0; d < self_dims; d++) {
        if (d != dim) {
            RAXIS_CHECK(ensure_nonempty_size(index, d) <= ensure_nonempty_size(self, d),
                "Size does not match at dimension ", d, " expected index ", index.shape(),
                " to be no larger than self ", self.shape(), " apart from dimension ", dim);
        }
    }
}

void scatter_shape_check(const Tensor& self, int64_t dim, const Tensor& index, const Tensor* src) {
    if (index.numel() == 0) return;
    int64_t self_dims = ensure_nonempty_dim(self.ndim());
    RAXIS_CHECK(self_dims == ensure_nonempty_dim(index.ndim()),
        "Index tensor must have the same number of dimensions as self tensor");

    bool is_wrong_shape = false;
    for (int64_t d = 0; d < self_dims && !is_wrong_shape; d++) {
        if (d != dim && ensure_nonempty_size(index, d) > ensure_nonempty_size(self, d)) {
            is_wrong_shape = true;
        }
    }
    if (src) {
        RAXIS_CHECK(ensure_nonempty_dim(src->ndim()) == ensure_nonempty_dim(index.ndim()),
            "Index tensor must have the same number of dimensions as src tensor");
        for (int64_t d = 0; d < self_dims && !is_wrong_shape; d++) {
            if (ensure_nonempty_size(index, d) > ensure_nonempty_size(*src, d)) {
                is_wrong_shape = true;
            }
        }
        RAXIS_CHECK(!is_wrong_shape, "Expected index ", index.shape(),
            " to be no larger than self ", self.shape(), " apart from dimension ", dim,
            " and to be no larger size than src ", src->shape());
    } else {
        RAXIS_CHECK(!is_wrong_shape, "Expected index ", index.shape(),
            " to be no larger than self ", self.shape(), " apart from dimension ", dim);
    }
}

void check_index_dtype(const char* name, const Tensor& index) {
    RAXIS_CHECK(index.dtype() == dtypes::Int64,
        name, "(): Expected dtype int64 for index, got ", index.dtype());
}

} // namespace

namespace ops {

// From: aten/src/ATen/native/TensorAdvancedIndexing.cpp
// out[i][j] = self[index[i][j]][j] (dim = 0): out has the shape of index
Tensor gather(const Tensor& self, int64_t dim, const Tensor& index) {
    check_index_dtype("gather", index);
    dim = wrap_dim(dim, ensure_nonempty_dim(self.ndim()));
    gather_shape_check(self, dim, index);

    Tensor result = empty(index.shape(), self.options());
    if (index.numel() == 0) return result;
    gather_stub(self.device().type(), result, dim, index, self);
    return result;
}

// self[index[i][j]][j] = src[i][j] (dim = 0). With repeated indices the
// last write wins (order not specified, as in torch)
Tensor& scatter_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    check_index_dtype("scatter", index);
    RAXIS_CHECK(self.dtype() == src.dtype(),
        "scatter(): Expected self.dtype to be equal to src.dtype");
    dim = wrap_dim(dim, ensure_nonempty_dim(self.ndim()));
    scatter_shape_check(self, dim, index, &src);
    assert_no_internal_overlap(self);
    assert_no_overlap(self, index);
    if (index.numel() == 0) return self;
    scatter_stub(self.device().type(), self, dim, index, src);
    return self;
}

Tensor& scatter_(Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    check_index_dtype("scatter", index);
    dim = wrap_dim(dim, ensure_nonempty_dim(self.ndim()));
    scatter_shape_check(self, dim, index, nullptr);
    assert_no_internal_overlap(self);
    assert_no_overlap(self, index);
    if (index.numel() == 0) return self;
    scatter_fill_stub(self.device().type(), self, dim, index, value);
    return self;
}

// self[index[i][j]][j] += src[i][j] (dim = 0): repeated indices accumulate
Tensor& scatter_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    check_index_dtype("scatter_add", index);
    RAXIS_CHECK(self.dtype() == src.dtype(),
        "scatter_add(): Expected self.dtype to be equal to src.dtype");
    dim = wrap_dim(dim, ensure_nonempty_dim(self.ndim()));
    scatter_shape_check(self, dim, index, &src);
    assert_no_internal_overlap(self);
    assert_no_overlap(self, index);
    if (index.numel() == 0) return self;
    scatter_add_stub(self.device().type(), self, dim, index, src);
    return self;
}

Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    Tensor result = self.clone();
    return ops::scatter_(result, dim, index, src);
}

Tensor scatter(const Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    Tensor result = self.clone();
    return ops::scatter_(result, dim, index, value);
}

Tensor scatter_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    Tensor result = self.clone();
    return ops::scatter_add_(result, dim, index, src);
}


// From: aten/src/ATen/native/TensorAdvancedIndexing.cpp (index_select_out_cpu_)
// out.select(dim, i) = self.select(dim, index[i]): whole slices are copied.
// One TensorIterator is built for the copy of one slice and pointed at each
// (source, destination) pair in turn, so its setup is paid once.
Tensor index_select(const Tensor& self, int64_t dim, const Tensor& index) {
    check_cpu_only("index_select", self, index);
    RAXIS_CHECK(index.ndim() <= 1, "index_select(): Index is supposed to be a vector");
    RAXIS_CHECK(index.dtype() == dtypes::Int64 || index.dtype() == dtypes::Int32,
        "index_select(): Expected dtype int32/int64 for index");
    dim = wrap_dim(dim, ensure_nonempty_dim(self.ndim()));
    Tensor index_contig = index.to(dtypes::Int64).contiguous();
    const int64_t* index_data = index_contig.data_ptr<int64_t>();
    int64_t numel = index_contig.numel();

    DimVector result_shape(self.shape());
    if (self.ndim() > 0) {
        result_shape[dim] = numel;
    }
    Tensor result = empty(result_shape, self.options());

    int64_t self_dim_size = ensure_nonempty_size(self, dim);
    auto check = [self_dim_size](int64_t i) {
        RAXIS_CHECK_INDEX(i >= 0 && i < self_dim_size, "index out of range in self");
    };

    if (self.ndim() == 0) {
        RAXIS_CHECK(numel == 1,
            "index_select(): Index to scalar can have only 1 value, got ", numel, " value(s)");
    }
    if (self.ndim() <= 1) {
        // Plain element copies
        int64_t self_stride = self.ndim() == 0 ? 1 : self.strides()[dim];
        int64_t result_stride = result.ndim() == 0 ? 1 : result.strides()[dim];
        self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            const scalar_t* self_ptr = self.data_ptr<scalar_t>();
            scalar_t* result_ptr = result.mutable_data_ptr<scalar_t>();
            for (int64_t i = 0; i < numel; i++) {
                check(index_data[i]);
                result_ptr[i * result_stride] = self_ptr[index_data[i] * self_stride];
            }
        });
        return result;
    }
    if (numel == 0 || self.numel() == 0) {
        return result;
    }

    Tensor self_slice = self.select(dim, 0);
    Tensor result_slice = result.select(dim, 0);
    const char* self_slice_data = static_cast<const char*>(self_slice.data_ptr());
    char* result_slice_data = static_cast<char*>(result_slice.mutable_data_ptr());
    int64_t itemsize = static_cast<int64_t>(self.dtype().itemsize());
    int64_t self_stride_bytes = self.strides()[dim] * itemsize;
    int64_t result_stride_bytes = result.strides()[dim] * itemsize;
    int64_t slice_size = self_slice.numel();

    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(result_slice)
        .add_input(self_slice)
        .build();

    auto copy_slices = [&](int64_t start, int64_t end) {
        TensorIterator sub_iter(iter);
        for (int64_t i = start; i < end; i++) {
            check(index_data[i]);
            sub_iter.unsafe_replace_operand(0, result_slice_data + i * result_stride_bytes);
            sub_iter.unsafe_replace_operand(1, const_cast<char*>(
                self_slice_data + index_data[i] * self_stride_bytes));
            copy_stub(sub_iter.device_type(), sub_iter);
        }
    };

    // Large slices: the copy kernel itself runs in parallel. Small slices:
    // split the indices between threads instead, with a memcpy per slice
    // when both slices are contiguous
    if (slice_size >= detail::GRAIN_SIZE) {
        copy_slices(0, numel);
    } else if (iter.is_contiguous()) {
        int64_t slice_bytes = slice_size * itemsize;
        parallel_for(0, numel, detail::GRAIN_SIZE / slice_size, [&](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; i++) {
                check(index_data[i]);
                std::memcpy(result_slice_data + i * result_stride_bytes,
                            self_slice_data + index_data[i] * self_stride_bytes, slice_bytes);
            }
        });
    } else {
        parallel_for(0, numel, detail::GRAIN_SIZE / slice_size, copy_slices);
    }
    return result;
}

// From: aten/src/ATen/native/TensorAdvancedIndexing.cpp (index_add_cpu_)
// self.select(dim, index[i]) += source.select(dim, i), one slice at a time
// (indices can repeat, so the slices are not split between threads)
Tensor& index_add_(Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    check_cpu_only("index_add_", self, index, source);
    RAXIS_CHECK(index.ndim() <= 1, "index_add_(): Index is supposed to be a vector");
    RAXIS_CHECK(index.dtype() == dtypes::Int64 || index.dtype() == dtypes::Int32,
        "index_add_(): Expected dtype int32/int64 for index");
    RAXIS_CHECK(self.dtype() == source.dtype(),
        "index_add_(): self (", self.dtype(), ") and source (", source.dtype(),
        ") must have the same scalar type");
    dim = wrap_dim(dim, ensure_nonempty_dim(self.ndim()));
    Tensor index_contig = index.to(dtypes::Int64).contiguous();
    const int64_t* index_data = index_contig.data_ptr<int64_t>();
    int64_t numel = index_contig.numel();
    RAXIS_CHECK(numel == ensure_nonempty_size(source, dim),
        "index_add_(): Number of indices (", numel,
        ") should be equal to source.size(dim) (", ensure_nonempty_size(source, dim), ")");
    assert_no_internal_overlap(self);
    assert_no_overlap(self, index);
    assert_no_overlap(self, source);

    int64_t self_dim_size = ensure_nonempty_size(self, dim);
    auto check = [self_dim_size](int64_t i) {
        RAXIS_CHECK(i >= 0 && i < self_dim_size, "index out of range in self");
    };

    if (self.ndim() <= 1) {
        int64_t self_stride = self.ndim() == 0 ? 1 : self.strides()[dim];
        int64_t source_stride = source.ndim() == 0 ? 1 : source.strides()[dim];
        self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            scalar_t* self_ptr = self.mutable_data_ptr<scalar_t>();
            const scalar_t* source_ptr = source.data_ptr<scalar_t>();
            for (int64_t i = 0; i < numel; i++) {
                check(index_data[i]);
                scalar_t& out = self_ptr[index_data[i] * self_stride];
                // A sum of bools is their OR (PyTorch writes += and lets it convert)
                if constexpr (std::is_same_v<scalar_t, bool>) {
                    out = out || source_ptr[i * source_stride];
                } else {
                    out += source_ptr[i * source_stride];
                }
            }
        });
        return self;
    }
    if (numel == 0) {
        return self;
    }

    Tensor self_slice = self.select(dim, 0);
    Tensor source_slice = source.select(dim, 0);
    char* self_slice_data = static_cast<char*>(self_slice.mutable_data_ptr());
    const char* source_slice_data = static_cast<const char*>(source_slice.data_ptr());
    int64_t itemsize = static_cast<int64_t>(self.dtype().itemsize());
    int64_t self_stride_bytes = self.strides()[dim] * itemsize;
    int64_t source_stride_bytes = source.strides()[dim] * itemsize;

    // self_slice = self_slice + source_slice, retargeted per index
    auto iter = TensorIteratorConfig()
        .add_output(self_slice)
        .add_input(self_slice)
        .add_input(source_slice)
        .build();

    for (int64_t i = 0; i < numel; i++) {
        check(index_data[i]);
        char* self_data = self_slice_data + index_data[i] * self_stride_bytes;
        iter.unsafe_replace_operand(0, self_data);
        iter.unsafe_replace_operand(1, self_data);
        iter.unsafe_replace_operand(2, const_cast<char*>(source_slice_data + i * source_stride_bytes));
        add_stub(iter.device_type(), iter);
    }
    return self;
}

Tensor index_add(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& source) {
    Tensor result = self.clone(MemoryFormat::Contiguous);
    return ops::index_add_(result, dim, index, source);
}


// ============================================================
// Advanced indexing: x[idx], x[mask], x[idx] = values
// ============================================================

namespace {

bool is_mask(const Tensor& index) {
    return index.dtype() == dtypes::Bool || index.dtype() == dtypes::Uint8;
}

// From: aten/src/ATen/native/IndexingUtils.h (checkIndexTensorTypes)
void check_index_tensor_types(ArrayRef<Tensor> indices) {
    for (const Tensor& index : indices) {
        if (!index.defined()) continue;
        RAXIS_CHECK_INDEX(is_mask(index) || index.dtype() == dtypes::Int64 || index.dtype() == dtypes::Int32,
            "tensors used as indices must be long, int, byte or bool tensors");
    }
}

// From: aten/src/ATen/native/IndexingUtils.h (expandTensors)
// A mask over k dims becomes k long index tensors, the columns of
// mask.nonzero(): x[mask] is x[i0, i1, ...] at the positions where the mask
// is true. From here on every index picks positions.
std::vector<Tensor> expand_masks(const Tensor& self, ArrayRef<Tensor> indices) {
    std::vector<Tensor> result;
    for (const Tensor& index : indices) {
        if (!index.defined() || !is_mask(index)) {
            result.push_back(index);
            continue;
        }
        for (int64_t j = 0; j < index.ndim(); j++) {
            int64_t src_idx = static_cast<int64_t>(result.size()) + j;
            RAXIS_CHECK_INDEX(src_idx < self.ndim(),
                "too many indices for tensor of dimension ", self.ndim());
            RAXIS_CHECK_INDEX(index.shape()[j] == self.shape()[src_idx],
                "The shape of the mask ", index.shape(), " at index ", j,
                " does not match the shape of the indexed tensor ", self.shape(),
                " at index ", src_idx);
        }
        Tensor nonzero = ops::nonzero(index);
        for (int64_t j = 0; j < index.ndim(); j++) {
            result.push_back(ops::select(nonzero, 1, j));
        }
    }
    return result;
}

// From: aten/src/ATen/native/IndexingUtils.h (hasContiguousSubspace)
// True if the defined indices are adjacent: x[:, i, j] but not x[i, :, j]
bool has_contiguous_subspace(ArrayRef<Tensor> indices) {
    int64_t first = -1;
    int64_t last = -1;
    for (int64_t i = 0; i < static_cast<int64_t>(indices.size()); i++) {
        if (indices[i].defined()) {
            if (first < 0) first = i;
            last = i;
        }
    }
    for (int64_t i = first + 1; i < last; i++) {
        if (!indices[i].defined()) return false;
    }
    return true;
}

// From: aten/src/ATen/native/IndexingUtils.h (transposeToFront)
// Moves the indexed dims to the front, the others keep their order:
// {-, a, -, b} -> self.permute({1, 3, 0, 2}) and {a, b, -, -}
std::pair<Tensor, std::vector<Tensor>> transpose_to_front(const Tensor& self, ArrayRef<Tensor> indices) {
    DimVector dims;
    std::vector<Tensor> transposed_indices;
    for (int64_t i = 0; i < self.ndim(); i++) {
        if (indices[i].defined()) {
            dims.push_back(i);
            transposed_indices.push_back(indices[i]);
        }
    }
    for (int64_t i = 0; i < self.ndim(); i++) {
        if (!indices[i].defined()) {
            dims.push_back(i);
            transposed_indices.emplace_back();
        }
    }
    return {ops::permute(self, dims), std::move(transposed_indices)};
}

// From: aten/src/ATen/native/TensorAdvancedIndexing.cpp (AdvancedIndex)
//
// The result of x[:, i0, i1] (x of shape (B, N, M, D), i0 and i1 broadcast to
// shape S) has shape (B, *S, D): the indexed dims are replaced by the shape
// of the indices. AdvancedIndex lays out the operands of a TensorIterator
// over that result shape:
//   src:     x with the indexed dims replaced by S with stride 0, so it only
//            moves along B and D; the kernel adds i0 * N_stride + i1 * M_stride
//   indices: each one viewed as (1, *S, 1), so it only moves along S
// The kernel also needs the sizes and byte strides of the indexed dims of x.
struct AdvancedIndex {
    Tensor src;
    std::vector<Tensor> indices;
    DimVector indexed_sizes;
    DimVector indexed_strides;
};

AdvancedIndex make_advanced_index(const Tensor& src, ArrayRef<Tensor> indices) {
    AdvancedIndex info;
    const int64_t element_size = static_cast<int64_t>(src.dtype().itemsize());
    int64_t dims_before = 0;
    int64_t dims_after = 0;
    int64_t dims_indexed = 0;
    IntArrayRef replacement_shape;
    for (int64_t dim = 0; dim < static_cast<int64_t>(indices.size()); dim++) {
        if (!indices[dim].defined()) {
            if (dims_indexed == 0) {
                dims_before++;
            } else {
                dims_after++;
            }
        } else {
            dims_indexed++;
            replacement_shape = indices[dim].shape();
            info.indexed_sizes.push_back(src.shape()[dim]);
            info.indexed_strides.push_back(src.strides()[dim] * element_size);
        }
    }

    // A size-0 indexed dim with a non-empty index: no index can be valid
    if (std::find(info.indexed_sizes.begin(), info.indexed_sizes.end(), 0) != info.indexed_sizes.end() &&
        std::find(replacement_shape.begin(), replacement_shape.end(), 0) == replacement_shape.end()) {
        RAXIS_CHECK_INDEX(false, "index is out of bounds for dimension with size 0");
    }

    // src: the indexed dims become replacement_shape with stride 0
    DimVector shape;
    DimVector strides;
    for (int64_t d = 0; d < dims_before; d++) {
        shape.push_back(src.shape()[d]);
        strides.push_back(src.strides()[d]);
    }
    for (int64_t size : replacement_shape) {
        shape.push_back(size);
        strides.push_back(0);
    }
    for (int64_t d = dims_before + dims_indexed; d < src.ndim(); d++) {
        shape.push_back(src.shape()[d]);
        strides.push_back(src.strides()[d]);
    }
    info.src = make_view(src, src.storage_offset(), shape, strides, /*is_differentiable_view=*/false);

    // Each index as (1, ..., 1, *index_shape, 1, ..., 1)
    for (const Tensor& index : indices) {
        if (!index.defined()) continue;
        DimVector index_shape;
        for (int64_t d = 0; d < dims_before; d++) index_shape.push_back(1);
        for (int64_t size : index.shape()) index_shape.push_back(size);
        for (int64_t d = 0; d < dims_after; d++) index_shape.push_back(1);
        info.indices.push_back(ops::reshape(index, index_shape));
    }
    return info;
}

// From: aten/src/ATen/native/TensorAdvancedIndexingUtils.h (make_info)
AdvancedIndex make_info(Tensor self, ArrayRef<Tensor> orig) {
    check_index_tensor_types(orig);
    std::vector<Tensor> indices = expand_masks(self, orig);

    // Broadcast the index tensors together
    SmallVector<IntArrayRef, 4> shapes;
    for (const Tensor& index : indices) {
        if (index.defined()) shapes.push_back(index.shape());
    }
    DimVector index_shape;
    try {
        index_shape = broadcast_shapes(shapes);
    } catch (const std::exception&) {
        std::ostringstream os;
        for (size_t i = 0; i < shapes.size(); i++) {
            os << (i ? ", " : "") << shapes[i];
        }
        RAXIS_CHECK_INDEX(false,
            "shape mismatch: indexing tensors could not be broadcast together with shapes ", os.str());
    }
    for (Tensor& index : indices) {
        if (index.defined() && index.shape() != IntArrayRef(index_shape)) {
            index = ops::expand(index, index_shape);
        }
    }
    indices.resize(self.ndim());

    // Indexed dims separated by a whole dim (x[i, :, j]) go to the front of
    // the result, as in NumPy: there is no natural place for them
    if (!has_contiguous_subspace(indices)) {
        std::tie(self, indices) = transpose_to_front(self, indices);
    }
    for (Tensor& index : indices) {
        if (index.defined() && index.dtype() == dtypes::Int32) {
            index = index.to(dtypes::Int64);
        }
    }
    return make_advanced_index(self, indices);
}

// A mask gives each selected position once (the rows of mask.nonzero()), and
// whole dims add distinct positions: no position can repeat
bool has_unique_positions(ArrayRef<Tensor> indices) {
    int64_t num_tensors = 0;
    bool all_masks = true;
    for (const Tensor& index : indices) {
        if (!index.defined()) continue;
        num_tensors++;
        all_masks &= is_mask(index);
    }
    return num_tensors == 1 && all_masks;
}

void check_too_many_indices(const Tensor& self, ArrayRef<Tensor> indices) {
    RAXIS_CHECK_INDEX(static_cast<int64_t>(indices.size()) <= self.ndim(),
        "too many indices for tensor of dimension ", self.ndim(),
        " (got ", indices.size(), ")");
}

bool is_expandable_to(IntArrayRef shape, IntArrayRef desired) {
    if (shape.size() > desired.size()) return false;
    const size_t offset = desired.size() - shape.size();
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] != desired[offset + i] && shape[i] != 1) return false;
    }
    return true;
}

// From: aten/src/ATen/native/TensorAdvancedIndexingUtils.h (canDispatchToMaskedFill)
// x[mask] = scalar is x.masked_fill_(mask, scalar): no nonzero, no index
// tensors. The mask gets trailing 1-dims to broadcast against self; whole
// dims before it broadcast on their own (x[:, mask] = 0).
std::optional<Tensor> mask_for_masked_fill(const Tensor& self, ArrayRef<Tensor> indices, const Tensor& value) {
    if (value.numel() != 1) {
        return std::nullopt;
    }
    int64_t num_ind = 0;
    Tensor mask;
    for (const Tensor& index : indices) {
        if (!index.defined()) {
            if (!mask.defined()) num_ind++;
            continue;
        }
        if (!is_mask(index) || mask.defined()) {
            return std::nullopt;
        }
        mask = index;
        for (int64_t j = 0; j < index.ndim(); j++) {
            int64_t src_idx = num_ind + j;
            RAXIS_CHECK_INDEX(src_idx < self.ndim() && index.shape()[j] == self.shape()[src_idx],
                "The shape of the mask ", index.shape(), " at index ", j,
                " does not match the shape of the indexed tensor ", self.shape(),
                " at index ", src_idx);
        }
        num_ind += mask.ndim();
    }
    if (!mask.defined()) {
        return std::nullopt;
    }
    if (mask.dtype() != dtypes::Bool) {
        mask = mask.to(dtypes::Bool);
    }
    for (int64_t i = num_ind; i < self.ndim(); i++) {
        mask = ops::unsqueeze(mask, -1);
    }
    return mask;
}

} // namespace

// From: aten/src/ATen/native/TensorAdvancedIndexing.cpp (index)
Tensor index(const Tensor& self, ArrayRef<Tensor> indices) {
    RAXIS_CHECK_INDEX(!indices.empty(), "at least one index must be provided");
    check_too_many_indices(self, indices);

    AdvancedIndex info = make_info(self, indices);
    Tensor result;
    TensorIteratorConfig config;
    config.check_mem_overlap(false)
        .check_all_same_dtype(false)
        .declare_static_dtype(self.dtype())
        .add_output(result)
        .add_input(info.src);
    for (const Tensor& index : info.indices) {
        config.add_input(index);
    }
    auto iter = config.build();
    index_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides);
    return iter.get_output(0);
}

// From: aten/src/ATen/native/TensorAdvancedIndexing.cpp (_index_put_impl_)
// The layout of index with the roles swapped: the output is the restrided
// self, and the values are read along the result shape
Tensor& index_put_(Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate) {
    check_too_many_indices(self, indices);
    if (!accumulate) {
        if (auto mask = mask_for_masked_fill(self, indices, values)) {
            return ops::masked_fill_(self, *mask, values.item());
        }
    }
    assert_no_overlap(self, values);
    for (const Tensor& index : indices) {
        if (index.defined()) assert_no_overlap(self, index);
    }

    AdvancedIndex info = make_info(self, indices);
    RAXIS_CHECK(is_expandable_to(values.shape(), info.src.shape()),
        "shape mismatch: value tensor of shape ", values.shape(),
        " cannot be broadcast to indexing result of shape ", info.src.shape());
    RAXIS_CHECK(values.dtype() == self.dtype(),
        "Index put requires the source and destination dtypes match, got ", self.dtype(),
        " for the destination and ", values.dtype(), " for the source.");

    TensorIteratorConfig config;
    config.check_mem_overlap(false)
        .check_all_same_dtype(false)
        .add_output(info.src)
        .add_input(values);
    for (const Tensor& index : info.indices) {
        config.add_input(index);
    }
    auto iter = config.build();
    index_put_stub(iter.device_type(), iter, info.indexed_sizes, info.indexed_strides,
                   accumulate, has_unique_positions(indices));
    return self;
}

Tensor index_put(const Tensor& self, ArrayRef<Tensor> indices, const Tensor& values, bool accumulate) {
    Tensor result = self.clone(MemoryFormat::Preserve);
    return ops::index_put_(result, indices, values, accumulate);
}

// From: aten/src/ATen/native/TensorAdvancedIndexing.cpp (nonzero_out_cpu)
// Two passes over the same chunks: count the nonzeros of each chunk, then
// each chunk writes its rows from the prefix sum of the counts. PyTorch
// matches the chunks of both passes through the thread id inside
// parallel_for; fixing the chunk boundaries here gives the same without
// relying on how parallel_for splits the range.
Tensor nonzero(const Tensor& self) {
    check_cpu_only("nonzero", self);
    const Tensor input = self.contiguous();
    const int64_t numel = input.numel();
    const int64_t ndim = input.ndim();
    const IntArrayRef sizes = input.shape();

    const int64_t num_chunks = std::max<int64_t>(1,
        std::min<int64_t>(get_num_threads(), divup(numel, detail::GRAIN_SIZE)));
    const int64_t chunk_size = std::max<int64_t>(1, divup(numel, num_chunks));
    std::vector<int64_t> chunk_offsets(num_chunks + 1, 0);

    Tensor result;
    input.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* data = input.data_ptr<scalar_t>();

        parallel_for(0, num_chunks, 1, [&](int64_t first, int64_t last) {
            for (int64_t c = first; c < last; c++) {
                const int64_t begin = c * chunk_size;
                const int64_t end = std::min(numel, begin + chunk_size);
                int64_t count = 0;
                for (int64_t i = begin; i < end; i++) {
                    count += data[i] != scalar_t(0);
                }
                chunk_offsets[c + 1] = count;
            }
        });
        for (int64_t c = 0; c < num_chunks; c++) {
            chunk_offsets[c + 1] += chunk_offsets[c];
        }
        const int64_t total = chunk_offsets[num_chunks];

        // Column-major (n, ndim): the indices along each dim are contiguous,
        // which is how x[mask] reads them (gh-46224 in PyTorch)
        result = ops::transpose(empty({ndim, total}, self.options().dtype(dtypes::Int64)), 0, 1);
        if (total == 0 || ndim == 0) return;
        int64_t* out = result.mutable_data_ptr<int64_t>();

        parallel_for(0, num_chunks, 1, [&](int64_t first, int64_t last) {
            DimVector idx(ndim, 0);
            for (int64_t c = first; c < last; c++) {
                const int64_t begin = c * chunk_size;
                const int64_t end = std::min(numel, begin + chunk_size);
                // Multi-index of the first element of the chunk
                int64_t linear = begin;
                for (int64_t k = ndim - 1; k >= 0; k--) {
                    idx[k] = linear % sizes[k];
                    linear /= sizes[k];
                }
                int64_t row = chunk_offsets[c];
                for (int64_t i = begin; i < end; i++) {
                    if (data[i] != scalar_t(0)) {
                        for (int64_t k = 0; k < ndim; k++) {
                            out[k * total + row] = idx[k];
                        }
                        row++;
                    }
                    for (int64_t k = ndim - 1; k >= 0 && ++idx[k] == sizes[k]; k--) {
                        idx[k] = 0;
                    }
                }
            }
        });
    });
    return result;
}

// From: aten/src/ATen/native/TensorAdvancedIndexing.cpp (masked_fill_impl_cpu)
Tensor& masked_fill_(Tensor& self, const Tensor& mask, Scalar value) {
    RAXIS_CHECK(mask.dtype() == dtypes::Bool,
        "masked_fill_ only supports boolean masks, but got mask with dtype ", mask.dtype());
    RAXIS_CHECK(is_expandable_to(mask.shape(), self.shape()),
        "masked_fill_: the mask of shape ", mask.shape(),
        " cannot be broadcast to the tensor of shape ", self.shape());
    assert_no_partial_overlap(self, mask);

    auto iter = TensorIteratorConfig()
        .check_mem_overlap(false)
        .check_all_same_dtype(false)
        .add_output(self)
        .add_input(mask)
        .build();
    masked_fill_stub(iter.device_type(), iter, value);
    return self;
}

Tensor masked_fill(const Tensor& self, const Tensor& mask, Scalar value) {
    DimVector shape = broadcast_shapes(self.shape(), mask.shape());
    Tensor result = ops::expand(self, shape).clone(MemoryFormat::Contiguous);
    return ops::masked_fill_(result, mask, value);
}

} // namespace ops

DEFINE_DISPATCH(gather_stub);
DEFINE_DISPATCH(scatter_stub);
DEFINE_DISPATCH(scatter_fill_stub);
DEFINE_DISPATCH(scatter_add_stub);
DEFINE_DISPATCH(index_stub);
DEFINE_DISPATCH(index_put_stub);
DEFINE_DISPATCH(masked_fill_stub);

} // namespace raxis
