
#include <raxis/tensor/tensor_shape.h>

#include <memory>
#include <numeric>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_indexing.h>
#include <raxis/tensor/detail/tensor_impl.h>


namespace raxis {

Tensor make_view(
    const Tensor& self,
    int64_t new_storage_offset,
    const IntArray& new_shape,
    const IntArray& new_strides
) {
    auto impl_ = self.unsafe_get_tensor_impl();
    auto impl = std::make_shared<detail::TensorImpl>(
        impl_->storage(),
        new_storage_offset,
        new_shape,
        new_strides,
        impl_->dtype()
    );
    return Tensor(std::move(impl));
}

Tensor alias(const Tensor& self) {
    return make_view(self, self.storage_offset(), self.shape(), self.strides());
}

Tensor select(const Tensor& self, int64_t dim, int64_t index) {
    const int n = self.ndim();
    RAXIS_CHECK(n > 0, "select() cannot be applied to a 0-d tensor");

    dim = wrap_dim(dim, n);
    index = wrap_index(index, self.shape()[dim]);

    const auto& old_shape = self.shape();
    const auto& old_strides = self.strides();

    int64_t new_offset = self.storage_offset() + index * old_strides[dim];

    IntArray new_shape;
    IntArray new_strides;
    new_shape.reserve(n - 1);
    new_strides.reserve(n - 1);

    for (int i = 0; i < n; ++i) {
        if (i == dim) continue;
        new_shape.push_back(old_shape[i]);
        new_strides.push_back(old_strides[i]);
    }

    return make_view(self, new_offset, new_shape, new_strides);
}

Tensor slice(const Tensor& self, int64_t dim, const indexing::Slice& slice) {
    dim = wrap_dim(dim, self.ndim());
    auto [start, stop, step] = slice.resolve(self.shape()[dim]);
    return ::raxis::slice(self, dim, start, stop, step);
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

    IntArray new_shape(old_shape);
    IntArray new_strides(old_strides);
    new_shape[dim] = new_dim_size;
    new_strides[dim] = old_strides[dim] * step;

    return make_view(self, new_offset, new_shape, new_strides);
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
    const int n = self.ndim();
    dim = wrap_dim_allow_upper(dim, n);

    const auto& old_shape = self.shape();
    const auto& old_strides = self.strides();

    IntArray new_shape;
    IntArray new_strides;
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

    IntArray new_shape;
    IntArray new_strides;

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
    const int n = self.ndim();
    dim0 = wrap_dim(dim0, n);
    dim1 = wrap_dim(dim1, n);

    if (dim0 == dim1) return alias(self);

    IntArray new_shape(self.shape());
    IntArray new_strides(self.strides());
    std::swap(new_shape[dim0], new_shape[dim1]);
    std::swap(new_strides[dim0], new_strides[dim1]);

    return make_view(self, self.storage_offset(), new_shape, new_strides);
}

Tensor permute(const Tensor& self, const IntArray& dims) {
    const int n = self.ndim();
    RAXIS_CHECK(static_cast<int>(dims.size()) == n,
        "permute(): number of dims (", dims.size(),
        ") must match tensor ndim (", n, ")");

    const auto& old_shape = self.shape();
    const auto& old_strides = self.strides();

    IntArray new_shape(n);
    IntArray new_strides(n);
    std::vector<bool> seen(n, false);

    for (int i = 0; i < n; ++i) {
        int64_t d = wrap_dim(dims[i], n);
        RAXIS_CHECK(!seen[d], "permute(): duplicate dim ", dims[i]);
        seen[d] = true;
        new_shape[i]   = old_shape[d];
        new_strides[i] = old_strides[d];
    }

    return make_view(self, self.storage_offset(), new_shape, new_strides);
}

Tensor swapdims(const Tensor& self, int64_t dim0, int64_t dim1) {
    return transpose(self, dim0, dim1);
}

Tensor movedim(const Tensor& self, const IntArray& src, const IntArray& dst) {
    const int n = self.ndim();
    RAXIS_CHECK(src.size() == dst.size(),
        "movedim(): source (", src.size(), " dims) and destination (",
        dst.size(), " dims) must have the same number of dims");

    if (n == 0) return alias(self);

    // Normalize and validate
    IntArray norm_src(src.size());
    IntArray norm_dst(src.size());

    for (size_t i = 0; i < src.size(); ++i) {
        norm_src[i] = wrap_dim(src[i], n);
        norm_dst[i] = wrap_dim(dst[i], n);
    }

    auto has_duplicates = [](IntArray v) {
        std::sort(v.begin(), v.end());
        return std::adjacent_find(v.begin(), v.end()) != v.end();
    };
    RAXIS_CHECK(!has_duplicates(norm_src), "movedim(): repeated dim in source");
    RAXIS_CHECK(!has_duplicates(norm_dst), "movedim(): repeated dim in destination");

    // Build permutation order
    // 1) Place known mappings: order[dst[i]] = src[i]
    IntArray order(n);
    IntArray src_remaining(n);
    IntArray dst_remaining(n);
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
    return movedim(self, IntArray{src}, IntArray{dst});
}

Tensor view(const Tensor& self, const IntArray& new_shape) {
    IntArray resolved = resolve_shape(new_shape, self.numel());
    auto new_strides = compute_view_strides(self.shape(), self.strides(), resolved);

    RAXIS_CHECK(new_strides.has_value(),
        "view size ", new_shape, " is not compatible with input tensor's "
        "size and stride (at least one dimension spans across two "
        "contiguous subspaces). Use reshape() instead.");

    return make_view(self, self.storage_offset(), resolved, *new_strides);
}

Tensor reshape(const Tensor& self, const IntArray& new_shape) {
    IntArray resolved = resolve_shape(new_shape, self.numel());
    auto new_strides = compute_view_strides(self.shape(), self.strides(), resolved);

    if (new_strides.has_value()) {
        return make_view(self, self.storage_offset(), resolved, *new_strides);
    }

    // View not possible, need a contiguous copy first
    return self.contiguous().view(resolved);
}

Tensor broadcast_to(const Tensor& self, const IntArray& target_shape) {
    int64_t ndim_old = static_cast<int64_t>(self.shape().size());
    int64_t ndim_new = static_cast<int64_t>(target_shape.size());

    RAXIS_CHECK(ndim_new >= ndim_old,
        "broadcast_to(): the number of sizes provided (", ndim_new,
        ") must be >= the number of dimensions in the tensor (", ndim_old, ")");

    const auto& old_shape   = self.shape();
    const auto& old_strides = self.strides();

    IntArray new_shape(ndim_new);
    IntArray new_strides(ndim_new);

    for (int64_t i = ndim_new - 1; i >= 0; --i) {
        int64_t old_idx = i - (ndim_new - ndim_old);

        if (old_idx < 0) {
            // Prepended dimension — broadcast from implicit size 1
            new_shape[i]   = target_shape[i];
            new_strides[i] = 0;
        } else if (old_shape[old_idx] == target_shape[i]) {
            new_shape[i]   = old_shape[old_idx];
            new_strides[i] = old_strides[old_idx];
        } else if (old_shape[old_idx] == 1) {
            new_shape[i]   = target_shape[i];
            new_strides[i] = 0;
        } else {
            RAXIS_CHECK(false,
                "broadcast_to(): the expanded size of the tensor (",
                target_shape[i], ") must match the existing size (",
                old_shape[old_idx], ") at non-singleton dimension ", i);
        }
    }

    return make_view(self, self.storage_offset(), new_shape, new_strides);
}

std::vector<Tensor> broadcast_tensors(const std::vector<Tensor>& tensors) {
    RAXIS_CHECK(!tensors.empty(), "broadcast_tensors(): no tensors provided");

    // Compute common shape
    std::vector<IntArray> shapes;
    shapes.reserve(tensors.size());
    for (const auto& t : tensors) {
        shapes.push_back(t.shape());
    }
    IntArray common = broadcast_shapes(shapes);

    // Expand each tensor to the common shape
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    for (const auto& t : tensors) {
        result.push_back(t.broadcast_to(common));
    }
    return result;
}

} // namespace raxis
