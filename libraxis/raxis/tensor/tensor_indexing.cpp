
#include <raxis/tensor/tensor_indexing.h>

#include <ostream>
#include <utility>
#include <vector>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/core/scalar.h>
#include <raxis/util/exception.h>
#include <raxis/util/array_ref.h>
#include <raxis/ops/tensor_advanced_indexing.h>


namespace raxis::indexing {


ResolvedSlice Slice::resolve(int64_t dim_size) const {
    int64_t start = this->start.value_or(0);
    int64_t stop = this->stop.value_or(dim_size);
    int64_t step = this->step.value_or(1);

    RAXIS_CHECK(step != 0, "slice step cannot be zero");
    RAXIS_CHECK(step > 0,
        "Negative step not yet supported in indexing (got step=", step, ")");

    auto clamp = [](int64_t v, int64_t lo, int64_t hi) -> int64_t {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    };

    // Handle negative indexing
    if (start < 0) start += dim_size;
    if (stop  < 0) stop  += dim_size;

    // Clamp to valid range
    start = clamp(start, 0, dim_size);
    stop = clamp(stop,  0, dim_size);

    return {start, stop, step};
}


std::ostream& operator<<(std::ostream& os, const Slice& s) {
    auto fmt = [&](std::optional<int64_t> v) {
        if (v.has_value()) os << *v;
        else os << "None";
    };
    fmt(s.start); os << ":"; fmt(s.stop); os << ":"; fmt(s.step);
    return os;
}


std::ostream& operator<<(std::ostream& os, const TensorIndex& idx) {
    switch (idx.type()) {
        case TensorIndexType::Integer:   os << idx.integer(); break;
        case TensorIndexType::Slice:     os << idx.slice();   break;
        case TensorIndexType::None:      os << "None";        break;
        case TensorIndexType::Ellipsis:  os << "...";         break;
        case TensorIndexType::Boolean:   os << (idx.boolean() ? "true" : "false"); break;
        case TensorIndexType::Tensor:    os << "<tensor>";    break;
    }
    return os;
}


namespace {

bool is_mask(const Tensor& t) {
    return t.dtype() == dtypes::Bool || t.dtype() == dtypes::Uint8;
}

// From: aten/src/ATen/TensorIndexing.h (count_specified_dimensions)
// Dims of self the indices consume: all but None, Ellipsis and bools, and a
// mask consumes as many dims as it has
int64_t count_specified_dims(ArrayRef<TensorIndex> indices) {
    int64_t count = 0;
    for (const auto& idx : indices) {
        if (idx.is_tensor()) {
            count += is_mask(idx.tensor()) ? idx.tensor().ndim() : 1;
        } else if (!idx.is_none() && !idx.is_ellipsis() && !idx.is_boolean()) {
            count++;
        }
    }
    return count;
}

// From: aten/src/ATen/TensorIndexing.h (applySelect)
Tensor apply_select(const Tensor& self, int64_t dim, int64_t index, int64_t real_dim) {
    RAXIS_CHECK_INDEX(self.ndim() > 0,
        "invalid index of a 0-dim tensor. Use `tensor.item()` in Python or "
        "`tensor.item<T>()` in C++ to convert a 0-dim tensor to a number");
    const int64_t size = self.shape()[dim];
    RAXIS_CHECK_INDEX(index >= -size && index < size,
        "index ", index, " is out of bounds for dimension ", real_dim, " with size ", size);
    return self.select(dim, index);
}

// From: aten/src/ATen/TensorIndexing.h (applySlice)
// A slice over the whole dim returns self, unless the caller needs a new
// tensor (x[:] must not be x itself)
Tensor apply_slice(const Tensor& self, int64_t dim, const Slice& slice, bool disable_slice_optimization) {
    const int64_t size = self.shape()[dim];
    const auto [start, stop, step] = slice.resolve(size);
    if (!disable_slice_optimization && start == 0 && stop == size && step == 1) {
        return self;
    }
    return self.slice(dim, start, stop, step);
}

// From: aten/src/ATen/TensorIndexing.h (boolToIndexingTensor)
// A bool index adds a dim of size 1, then indexes it with [0] when true
// (keeps it) or with [] when false (empties it)
Tensor bool_to_indexing_tensor(const Tensor& self, bool value) {
    TensorOptions options = self.options().dtype(dtypes::Int64);
    return value ? zeros({1}, options) : empty({0}, options);
}

// From: aten/src/ATen/TensorIndexing.h (recordTensorIndex)
// out_indices[d] holds the index of dim d of the sliced tensor. Dims kept
// whole get an undefined tensor, but only once a tensor index was seen:
// trailing ones are dropped anyway.
void record_tensor_index(const Tensor& tensor, std::vector<Tensor>& out_indices, int64_t& dim) {
    if (out_indices.empty()) {
        out_indices.resize(dim + 1);
        out_indices[dim] = tensor;
    } else {
        out_indices.push_back(tensor);
    }
    dim += is_mask(tensor) ? tensor.ndim() : 1;
}

// From: aten/src/ATen/TensorIndexing.h (handleDimInMultiDimIndexing)
// Applies one index of x[i0, i1, ...]: basic indices narrow the view, tensor
// indices are recorded for the advanced indexing step. `dim` is the dim of
// the current view the next index applies to, real_dim its position in the
// index expression (for error messages).
Tensor handle_dim(
    const Tensor& prev, const Tensor& original, const TensorIndex& index,
    int64_t& dim, int64_t specified_dims, int64_t real_dim,
    std::vector<Tensor>& out_indices, bool disable_slice_optimization
) {
    if (index.is_integer()) {
        return apply_select(prev, dim, index.integer(), real_dim);
    }
    if (index.is_slice()) {
        Tensor result = apply_slice(prev, dim, index.slice(), disable_slice_optimization);
        dim++;
        if (!out_indices.empty()) out_indices.emplace_back();
        return result;
    }
    if (index.is_ellipsis()) {
        const int64_t ellipsis_ndims = original.ndim() - specified_dims;
        dim += ellipsis_ndims;
        if (!out_indices.empty()) out_indices.resize(out_indices.size() + ellipsis_ndims);
        return prev;
    }
    if (index.is_none()) {
        Tensor result = prev.unsqueeze(dim);
        dim++;
        if (!out_indices.empty()) out_indices.emplace_back();
        return result;
    }
    if (index.is_boolean()) {
        Tensor result = prev.unsqueeze(dim);
        record_tensor_index(bool_to_indexing_tensor(result, index.boolean()), out_indices, dim);
        return result;
    }
    RAXIS_INTERNAL_ASSERT(index.is_tensor(), "Invalid TensorIndex type");
    const Tensor& tensor = index.tensor();
    // A 0-d integer tensor is an integer, a 0-d bool tensor a bool
    if (tensor.ndim() == 0 && tensor.dtype().is_integral(/*include_bool=*/true)) {
        if (!is_mask(tensor)) {
            return apply_select(prev, dim, tensor.item().to<int64_t>(), real_dim);
        }
        Tensor result = prev.unsqueeze(dim);
        record_tensor_index(bool_to_indexing_tensor(result, tensor.item().to<bool>()), out_indices, dim);
        return result;
    }
    record_tensor_index(tensor, out_indices, dim);
    return prev;
}

// From: aten/src/ATen/TensorIndexing.h (applySlicing)
Tensor apply_slicing(
    const Tensor& self, ArrayRef<TensorIndex> indices,
    std::vector<Tensor>& out_indices, bool disable_slice_optimization
) {
    int64_t num_ellipsis = 0;
    for (const auto& idx : indices) {
        num_ellipsis += idx.is_ellipsis();
    }
    RAXIS_CHECK_INDEX(num_ellipsis <= 1, "an index can only have a single ellipsis ('...')");

    const int64_t specified_dims = count_specified_dims(indices);
    RAXIS_CHECK_INDEX(specified_dims <= self.ndim(),
        "too many indices for tensor of dimension ", self.ndim());

    Tensor result = self;
    int64_t dim = 0;
    for (size_t i = 0; i < indices.size(); i++) {
        result = handle_dim(result, self, indices[i], dim, specified_dims,
                            static_cast<int64_t>(i), out_indices, disable_slice_optimization);
    }
    return result;
}

std::vector<Tensor> drop_trailing_undefined(std::vector<Tensor> indices) {
    while (!indices.empty() && !indices.back().defined()) {
        indices.pop_back();
    }
    return indices;
}

// From: aten/src/ATen/TensorIndexing.h (slicePrefix1sSize)
// NumPy strips the leading size-1 dims of the value: x[0:2] = v with v of
// shape (1, 1, 2) works
DimVector strip_leading_ones(IntArrayRef shape) {
    size_t first = 0;
    while (first < shape.size() && shape[first] == 1) {
        first++;
    }
    return DimVector(shape.begin() + first, shape.end());
}

// From: aten/src/ATen/TensorIndexing.h (copy_to)
void copy_to(const Tensor& dst, const Tensor& src) {
    if (dst.shape() == src.shape()) {
        dst.copy_(src);
    } else if (src.ndim() == 0) {
        dst.fill_(src.item());
    } else {
        dst.copy_(src.view(strip_leading_ones(src.shape())).expand(dst.shape()));
    }
}

} // anonymous namespace


Tensor get_item(const Tensor& self, ArrayRef<TensorIndex> indices) {
    RAXIS_CHECK(self.defined(), "Cannot index an undefined tensor");

    // A single basic index skips the bookkeeping. x[:] and x[...] still
    // return a new tensor (an alias), as in PyTorch and NumPy.
    if (indices.size() == 1) {
        const TensorIndex& index = indices[0];
        if (index.is_integer()) {
            return apply_select(self, 0, index.integer(), 0);
        } else if (index.is_slice()) {
            return apply_slice(self, 0, index.slice(), /*disable_slice_optimization=*/true);
        } else if (index.is_none()) {
            return self.unsqueeze(0);
        } else if (index.is_ellipsis()) {
            return self.alias();
        } else if (index.is_boolean()) {
            Tensor result = self.unsqueeze(0);
            return ::raxis::index(result, {bool_to_indexing_tensor(result, index.boolean())});
        }
    }

    std::vector<Tensor> tensor_indices;
    Tensor sliced = apply_slicing(self, indices, tensor_indices, /*disable_slice_optimization=*/false);
    if (tensor_indices.empty()) {
        return sliced.is_same(self) ? self.alias() : sliced;
    }
    return ::raxis::index(sliced, drop_trailing_undefined(std::move(tensor_indices)));
}

void set_item(const Tensor& self, ArrayRef<TensorIndex> indices, const Tensor& value) {
    RAXIS_CHECK(self.defined(), "Cannot index an undefined tensor");

    if (indices.size() == 1) {
        const TensorIndex& index = indices[0];
        if (index.is_boolean() && !index.boolean()) {
            return;
        } else if (index.is_ellipsis()) {
            copy_to(self, value);
            return;
        } else if (index.is_none() || index.is_boolean()) {
            copy_to(self.unsqueeze(0), value);
            return;
        } else if (index.is_integer()) {
            copy_to(apply_select(self, 0, index.integer(), 0), value);
            return;
        } else if (index.is_slice()) {
            copy_to(apply_slice(self, 0, index.slice(), /*disable_slice_optimization=*/false), value);
            return;
        }
    }

    std::vector<Tensor> tensor_indices;
    Tensor sliced = apply_slicing(self, indices, tensor_indices, /*disable_slice_optimization=*/false);
    if (tensor_indices.empty()) {
        copy_to(sliced, value);
        return;
    }
    DimVector value_shape = strip_leading_ones(value.shape());
    Tensor values = value.shape() == IntArrayRef(value_shape) ? value : value.view(value_shape);
    ::raxis::index_put_(sliced, drop_trailing_undefined(std::move(tensor_indices)), values);
}

void set_item(const Tensor& self, ArrayRef<TensorIndex> indices, Scalar value) {
    set_item(self, indices, scalar_tensor(value, self.options()));
}


} // namespace raxis::indexing
