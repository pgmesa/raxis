
#include <raxis/tensor/tensor_shape.h>

#include <memory>
#include <numeric>
#include <algorithm>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_indexing.h>
#include <raxis/autograd/grad_mode.h>
#include <raxis/autograd/autograd_meta.h>


namespace raxis {

Tensor make_view(
    const Tensor& self,
    int64_t new_storage_offset,
    IntArrayRef new_shape,
    IntArrayRef new_strides,
    bool is_differentiable_view
) {
    auto impl_ = self.unsafe_get_tensor_impl();
    auto impl = std::make_shared<detail::TensorImpl>(
        impl_->storage(),
        new_storage_offset,
        new_shape,
        new_strides,
        impl_->dtype()
    );

    // From: torch/csrc/autograd/VariableTypeUtils.h (as_view)
    // A view of a view has the same base. A view made under no_grad cannot
    // be modified in place with grad mode on, and the views of a view keep
    // the restriction of their parent (propagate_creation_meta).
    if (is_differentiable_view) {
        using autograd::CreationMeta;
        CreationMeta creation_meta = autograd::GradMode::is_enabled()
            ? CreationMeta::Default : CreationMeta::NoGradMode;
        if (impl_->is_view()) {
            if (creation_meta == CreationMeta::Default) {
                creation_meta = impl_->creation_meta();
            }
            impl->set_view_base(impl_->view_base(), creation_meta);
        } else {
            impl->set_view_base(self.impl_ptr(), creation_meta);
        }
    }
    return Tensor(std::move(impl));
}

namespace ops {

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
    const int n = self.ndim();
    RAXIS_CHECK(n > 0, "select() cannot be applied to a 0-d tensor");

    dim = wrap_dim(dim, n);
    const int64_t size = self.shape()[dim];
    RAXIS_CHECK_INDEX(index >= -size && index < size,
        "select(): index ", index, " out of range for tensor of size ",
        self.shape(), " at dimension ", dim);
    if (index < 0) index += size;

    const auto& old_shape = self.shape();
    const auto& old_strides = self.strides();

    int64_t new_offset = self.storage_offset() + index * old_strides[dim];

    DimVector new_shape;
    DimVector new_strides;
    new_shape.reserve(n - 1);
    new_strides.reserve(n - 1);

    for (int i = 0; i < n; ++i) {
        if (i == dim) continue;
        new_shape.push_back(old_shape[i]);
        new_strides.push_back(old_strides[i]);
    }

    return make_view(self, new_offset, new_shape, new_strides);
}

Tensor slice(const Tensor& self, int64_t dim, int64_t start, int64_t stop, int64_t step) {
    const int n = self.ndim();
    RAXIS_CHECK(n > 0, "slice() cannot be applied to a 0-d tensor");
    RAXIS_CHECK(step > 0, "slice(): step must be > 0, got ", step);

    // Normalize dim
    dim = wrap_dim(dim, n);
    
    // Compute new size along this dim
    int64_t new_dim_size = 0;
    if (stop > start) {
        new_dim_size = (stop - start + step - 1) / step;   // ceil division
    }

    const auto& old_shape = self.shape();
    const auto& old_strides = self.strides();

    int64_t new_offset = self.storage_offset() + start * old_strides[dim];

    DimVector new_shape(old_shape);
    DimVector new_strides(old_strides);
    new_shape[dim] = new_dim_size;
    new_strides[dim] = old_strides[dim] * step;

    return make_view(self, new_offset, new_shape, new_strides);
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
    const int n = self.ndim();
    dim = wrap_dim_allow_upper(dim, n);

    const auto& old_shape = self.shape();
    const auto& old_strides = self.strides();

    DimVector new_shape;
    DimVector new_strides;
    new_shape.reserve(n + 1);
    new_strides.reserve(n + 1);

    for (int i = 0; i < n + 1; ++i) {
        if (i == dim) {
            new_shape.push_back(1);
            // Stride for a size-1 dim is arbitrary but we pick a sensible
            // value: the stride of the next inner dimension (or 1).
            int64_t s = (i < n) ? old_strides[i] : 1;
            new_strides.push_back(s);
        } else {
            int src = (i < dim) ? i : i - 1;
            new_shape.push_back(old_shape[src]);
            new_strides.push_back(old_strides[src]);
        }
    }

    return make_view(self, self.storage_offset(), new_shape, new_strides);
}

Tensor squeeze(const Tensor& self, std::optional<int64_t> dim_opt) {
    const int n = self.ndim();
    const auto& old_shape = self.shape();
    const auto& old_strides = self.strides();

    DimVector new_shape;
    DimVector new_strides;

    if (!dim_opt.has_value()) {
        // Squeeze all size-1 dims
        for (int i = 0; i < n; ++i) {
            if (old_shape[i] != 1) {
                new_shape.push_back(old_shape[i]);
                new_strides.push_back(old_strides[i]);
            }
        }
    } else {
        int64_t dim = wrap_dim(dim_opt.value(), n);

        for (int i = 0; i < n; ++i) {
            if (i == dim && old_shape[i] == 1) continue;
            new_shape.push_back(old_shape[i]);
            new_strides.push_back(old_strides[i]);
        }
    }

    return make_view(self, self.storage_offset(), new_shape, new_strides);
}

Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
    // As in torch's maybe_wrap_dim, a 0-d tensor accepts dims 0 and -1
    const int n = std::max(self.ndim(), 1);
    dim0 = wrap_dim(dim0, n);
    dim1 = wrap_dim(dim1, n);

    if (dim0 == dim1) return alias(self);

    DimVector new_shape(self.shape());
    DimVector new_strides(self.strides());
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    return make_view(self, self.storage_offset(), new_shape, new_strides);
}

Tensor permute(const Tensor& self, IntArrayRef dims) {
    const int n = self.ndim();
    RAXIS_CHECK(static_cast<int>(dims.size()) == n,
        "permute(): number of dims (", dims.size(),
        ") must match tensor ndim (", n, ")");

    const auto& old_shape = self.shape();
    const auto& old_strides = self.strides();

    DimVector new_shape(n);
    DimVector new_strides(n);
    SmallVector<bool, 5> seen(n, false);

    for (int i = 0; i < n; ++i) {
        int64_t d = wrap_dim(dims[i], n);
        RAXIS_CHECK(!seen[d], "permute(): duplicate dim ", dims[i]);
        seen[d] = true;
        new_shape[i]   = old_shape[d];
        new_strides[i] = old_strides[d];
    }

    return make_view(self, self.storage_offset(), new_shape, new_strides);
}

Tensor view(const Tensor& self, IntArrayRef new_shape) {
    DimVector resolved = resolve_shape(new_shape, self.numel());
    auto new_strides = compute_view_strides(self.shape(), self.strides(), resolved);

    RAXIS_CHECK(new_strides.has_value(),
        "view size ", new_shape, " is not compatible with input tensor's "
        "size and stride (at least one dimension spans across two "
        "contiguous subspaces). Use reshape() instead.");

    return make_view(self, self.storage_offset(), resolved, *new_strides);
}

Tensor reshape(const Tensor& self, IntArrayRef new_shape) {
    DimVector resolved = resolve_shape(new_shape, self.numel());
    auto new_strides = compute_view_strides(self.shape(), self.strides(), resolved);

    if (new_strides.has_value()) {
        return make_view(self, self.storage_offset(), resolved, *new_strides);
    }

    // View not possible, need a contiguous copy first
    return self.contiguous().view(resolved);
}

// From: aten/src/ATen/ExpandUtils.cpp (inferExpandGeometry)
// A view where every size 1 dim (or new leading dim) grows to the requested
// size by giving it stride 0, so all its indices read the same element. -1
// keeps the current size.
Tensor expand(const Tensor& self, IntArrayRef shape) {
    int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t self_ndim = self.ndim();
    RAXIS_CHECK(ndim >= self_ndim,
        "expand(): the number of sizes provided (", ndim,
        ") must be greater or equal to the number of dimensions in the tensor (",
        self_ndim, ")");

    IntArrayRef self_shape = self.shape();
    IntArrayRef self_strides = self.strides();
    DimVector new_shape(ndim);
    DimVector new_strides(ndim);

    for (int64_t i = ndim - 1; i >= 0; --i) {
        int64_t dim = i - (ndim - self_ndim);
        int64_t size = dim >= 0 ? self_shape[dim] : 1;
        // A new leading dim of size 1 gets the stride a contiguous tensor
        // would have, so expanding to a size 1 dim keeps the tensor contiguous
        int64_t stride = dim >= 0 ? self_strides[dim] : new_shape[i + 1] * new_strides[i + 1];
        int64_t target_size = shape[i];
        if (target_size == -1) {
            RAXIS_CHECK(dim >= 0,
                "The expanded size of the tensor (-1) isn't allowed in a leading, "
                "non-existing dimension ", i);
            target_size = size;
        }
        if (size != target_size) {
            RAXIS_CHECK(size == 1,
                "The expanded size of the tensor (", target_size,
                ") must match the existing size (", size,
                ") at non-singleton dimension ", i, ".  Target sizes: ", shape,
                ".  Tensor sizes: ", self_shape);
            size = target_size;
            stride = 0;
        }
        new_shape[i] = size;
        new_strides[i] = stride;
    }

    return make_view(self, self.storage_offset(), new_shape, new_strides);
}

Tensor broadcast_to(const Tensor& self, IntArrayRef shape) {
    return ops::expand(self, shape);
}

// From: aten/src/ATen/native/TensorShape.cpp
// The pieces are views (narrow) of self, so splitting copies nothing
std::vector<Tensor> split_with_sizes(const Tensor& self, IntArrayRef split_sizes, int64_t dim) {
    RAXIS_CHECK(self.ndim() != 0, "split expects at least a 1-dimensional tensor");
    dim = wrap_dim(dim, self.ndim());
    int64_t dim_size = self.shape()[dim];

    std::vector<Tensor> splits;
    splits.reserve(split_sizes.size());
    int64_t start = 0;
    for (int64_t length : split_sizes) {
        RAXIS_CHECK(length >= 0,
            "split_with_sizes expects split_sizes have only non-negative entries, "
            "but got split_sizes=", split_sizes);
        RAXIS_CHECK(start + length <= dim_size,
            "split_with_sizes expects split_sizes to sum exactly to ", dim_size,
            " (input tensor's size at dimension ", dim, "), but got split_sizes=", split_sizes);
        splits.push_back(ops::slice(self, dim, start, start + length));
        start += length;
    }
    RAXIS_CHECK(start == dim_size,
        "split_with_sizes expects split_sizes to sum exactly to ", dim_size,
        " (input tensor's size at dimension ", dim, "), but got split_sizes=", split_sizes);
    return splits;
}

// From: aten/src/ATen/native/TensorShape.cpp
DimVector split_sizes_for(int64_t dim_size, int64_t split_size) {
    RAXIS_CHECK(split_size >= 0,
        "split expects split_size be non-negative, but got split_size=", split_size);
    RAXIS_CHECK(split_size > 0 || dim_size == 0,
        "split_size can only be 0 if dimension size is 0, but got dimension size of ", dim_size);
    // At least one split: a split_size larger than the dim gives the whole dim
    int64_t num_splits = split_size == 0
        ? 1 : std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
    DimVector sizes(num_splits, split_size);
    sizes.back() = split_size - (split_size * num_splits - dim_size);
    return sizes;
}

std::vector<Tensor> split(const Tensor& self, int64_t split_size, int64_t dim) {
    RAXIS_CHECK(self.ndim() != 0, "split expects at least a 1-dimensional tensor");
    dim = wrap_dim(dim, self.ndim());
    return ops::split_with_sizes(self, split_sizes_for(self.shape()[dim], split_size), dim);
}

// From: aten/src/ATen/native/TensorShape.cpp
// Pieces of ceil(size / chunks), so there can be fewer than `chunks`
// (5 elements in 4 chunks give 2, 2, 1)
std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
    RAXIS_CHECK(self.ndim() > 0, "chunk expects at least a 1-dimensional tensor");
    RAXIS_CHECK(chunks > 0, "chunk expects `chunks` to be greater than 0, got: ", chunks);
    dim = wrap_dim(dim, self.ndim());
    int64_t dim_size = self.shape()[dim];
    int64_t split_size = (dim_size + chunks - 1) / chunks;
    // An empty dim still gives `chunks` empty pieces
    if (split_size == 0 && dim_size == 0) {
        DimVector sizes(chunks, 0);
        return ops::split_with_sizes(self, sizes, dim);
    }
    return ops::split(self, split_size, dim);
}
    
} // namespace ops

Tensor alias(const Tensor& self) {
    return make_view(self, self.storage_offset(), self.shape(), self.strides());
}

Tensor slice(const Tensor& self, int64_t dim, const indexing::Slice& slice) {
    dim = wrap_dim(dim, self.ndim());
    auto [start, stop, step] = slice.resolve(self.shape()[dim]);
    return ::raxis::slice(self, dim, start, stop, step);
}

Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length) {
    RAXIS_CHECK(self.ndim() > 0, "narrow() cannot be applied to a 0-d tensor");
    dim = wrap_dim(dim, self.ndim());
    RAXIS_CHECK(length >= 0, "narrow(): length must be non-negative, got ", length);
    RAXIS_CHECK(start >= 0 && start + length <= self.shape()[dim],
        "narrow(): start (", start, ") + length (", length,
        ") exceeds dimension size (", self.shape()[dim], ")");

    return slice(self, dim, start, start + length);
}

Tensor swapdims(const Tensor& self, int64_t dim0, int64_t dim1) {
    return transpose(self, dim0, dim1);
}

// Merges dims [start_dim, end_dim] into one. A view when possible (reshape).
// From: aten/src/ATen/native/TensorShape.cpp (flatten)
Tensor flatten(const Tensor& self, int64_t start_dim, int64_t end_dim) {
    if (self.ndim() == 0) {
        return reshape(self, {1});
    }
    start_dim = wrap_dim(start_dim, self.ndim());
    end_dim = wrap_dim(end_dim, self.ndim());
    RAXIS_CHECK(start_dim <= end_dim,
        "flatten() has invalid args: start_dim cannot come after end_dim");
    if (start_dim == end_dim) {
        return self;
    }

    DimVector shape;
    for (int64_t d = 0; d < start_dim; ++d) {
        shape.push_back(self.shape()[d]);
    }
    int64_t merged = 1;
    for (int64_t d = start_dim; d <= end_dim; ++d) {
        merged *= self.shape()[d];
    }
    shape.push_back(merged);
    for (int64_t d = end_dim + 1; d < self.ndim(); ++d) {
        shape.push_back(self.shape()[d]);
    }
    return reshape(self, shape);
}

Tensor movedim(const Tensor& self, IntArrayRef src, IntArrayRef dst) {
    const int n = self.ndim();
    RAXIS_CHECK(src.size() == dst.size(),
        "movedim(): source (", src.size(), " dims) and destination (",
        dst.size(), " dims) must have the same number of dims");

    if (n == 0) return alias(self);

    // Normalize and validate
    DimVector norm_src(src.size());
    DimVector norm_dst(src.size());

    for (size_t i = 0; i < src.size(); ++i) {
        norm_src[i] = wrap_dim(src[i], n);
        norm_dst[i] = wrap_dim(dst[i], n);
    }

    auto has_duplicates = [](DimVector v) {
        std::sort(v.begin(), v.end());
        return std::adjacent_find(v.begin(), v.end()) != v.end();
    };
    RAXIS_CHECK(!has_duplicates(norm_src), "movedim(): repeated dim in source");
    RAXIS_CHECK(!has_duplicates(norm_dst), "movedim(): repeated dim in destination");

    // Build permutation order
    // 1) Place known mappings: order[dst[i]] = src[i]
    DimVector order(n);
    DimVector src_remaining(n);
    DimVector dst_remaining(n);
    std::iota(src_remaining.begin(), src_remaining.end(), 0);
    std::iota(dst_remaining.begin(), dst_remaining.end(), 0);

    for (size_t i = 0; i < norm_src.size(); ++i) {
        order[norm_dst[i]] = norm_src[i];
        src_remaining[norm_src[i]] = -1;
        dst_remaining[norm_dst[i]] = -1;
    }

    // 2) Remove placed dims, leaving the "rest"
    auto src_end = std::remove(src_remaining.begin(), src_remaining.end(), -1);
    auto dst_end = std::remove(dst_remaining.begin(), dst_remaining.end(), -1);

    // 3) Fill remaining positions
    int64_t rest = std::distance(src_remaining.begin(), src_end);
    for (int64_t i = 0; i < rest; ++i) {
        order[dst_remaining[i]] = src_remaining[i];
    }

    return permute(self, order);
}

Tensor movedim(const Tensor& self, int64_t src, int64_t dst) {
    return movedim(self, IntArrayRef(src), IntArrayRef(dst));
}

std::vector<Tensor> broadcast_tensors(ArrayRef<Tensor> tensors) {
    RAXIS_CHECK(!tensors.empty(), "broadcast_tensors(): no tensors provided");

    // Compute common shape
    std::vector<IntArrayRef> shapes;
    shapes.reserve(tensors.size());
    for (const auto& t : tensors) {
        shapes.push_back(t.shape());
    }
    DimVector common = broadcast_shapes(shapes);

    // Expand each tensor to the common shape
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    for (const auto& t : tensors) {
        result.push_back(t.broadcast_to(common));
    }
    return result;
}

} // namespace raxis
