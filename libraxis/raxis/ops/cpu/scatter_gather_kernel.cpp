
#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/loops.h>

#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>
#include <raxis/util/exception.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

inline namespace CPU_CAPABILITY {

namespace {

// From: aten/src/ATen/native/cpu/ScatterGatherKernel.cpp
//
// gather:  self[i][j] = src[index[i][j]][j]   (dim = 0)
// scatter: self[index[i][j]][j] = src[i][j]
//
// A TensorIterator walks every position of index except along `dim`
// (squashed to size 1), and the kernel loops along `dim` itself, where the
// index values say which element of self (scatter) or src (gather) to use.
// Different iterator positions touch different elements of self, so they
// can run in parallel even for scatter.

int64_t ensure_nonempty_size(const Tensor& t, int64_t dim) {
    return t.ndim() == 0 ? 1 : t.shape()[dim];
}

int64_t ensure_nonempty_stride(const Tensor& t, int64_t dim) {
    return t.ndim() == 0 ? 1 : t.strides()[dim];
}

struct TensorAssign {
    template <typename scalar_t>
    void operator()(scalar_t* self_data, const scalar_t* src_data) const {
        *self_data = *src_data;
    }
};

struct ReduceAdd {
    template <typename scalar_t>
    void operator()(scalar_t* self_data, const scalar_t* src_data) const {
        *self_data += *src_data;
    }
};

template <bool is_scatter_like, typename scalar_t, typename func_t>
void scatter_gather_dim_loop(
    scalar_t* self_data, int64_t self_dim_stride,
    const int64_t* index_data, int64_t index_dim_stride,
    const scalar_t* src_data, int64_t src_dim_stride,
    int64_t dim, int64_t index_dim_size, int64_t index_upper_bound,
    const func_t& f
) {
    for (int64_t i = 0; i < index_dim_size; i++) {
        int64_t idx_dim = index_data[i * index_dim_stride];
        RAXIS_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
            "index ", idx_dim, " is out of bounds for dimension ", dim,
            " with size ", index_upper_bound);
        f(self_data + (is_scatter_like ? idx_dim : i) * self_dim_stride,
          src_data + (is_scatter_like ? i : idx_dim) * src_dim_stride);
    }
}

// `src` is null for scatter with a scalar value: every write reads `value`
template <bool is_scatter_like, typename func_t>
void scatter_gather_base_kernel(
    const Tensor& self, int64_t dim, const Tensor& index, const Tensor* src,
    Scalar value, const func_t& f
) {
    const Tensor& src_or_index = src ? *src : index;
    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .check_mem_overlap(false)
        .declare_static_shape(index.shape(), {dim})
        .add_output(self)
        .add_input(src_or_index)
        .add_input(index)
        .build();

    int64_t self_dim_stride = ensure_nonempty_stride(self, dim);
    int64_t self_dim_size = ensure_nonempty_size(self, dim);
    int64_t index_dim_stride = ensure_nonempty_stride(index, dim);
    int64_t index_dim_size = ensure_nonempty_size(index, dim);
    int64_t src_dim_stride = src ? ensure_nonempty_stride(*src, dim) : 0;
    int64_t src_dim_size = src ? ensure_nonempty_size(*src, dim) : 0;
    int64_t index_upper_bound = is_scatter_like ? self_dim_size : src_dim_size;
    bool dim_is_last = dim == std::max<int64_t>(self.ndim(), 1) - 1;

    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t fill = value.to<scalar_t>();
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
            char* self_bytes = data[0];
            const char* src_bytes = src ? data[1] : reinterpret_cast<const char*>(&fill);
            int64_t src_iter_stride = src ? strides[1] : 0;
            const char* index_bytes = data[2];

            if (dim_is_last) {
                // The dim loop runs along the innermost (usually contiguous)
                // dim of each row
                for (int64_t k = 0; k < n; k++) {
                    scatter_gather_dim_loop<is_scatter_like>(
                        reinterpret_cast<scalar_t*>(self_bytes), self_dim_stride,
                        reinterpret_cast<const int64_t*>(index_bytes), index_dim_stride,
                        reinterpret_cast<const scalar_t*>(src_bytes), src_dim_stride,
                        dim, index_dim_size, index_upper_bound, f);
                    self_bytes += strides[0];
                    src_bytes += src_iter_stride;
                    index_bytes += strides[2];
                }
            } else {
                // Swap the loops so the inner one walks the iterator's
                // (innermost, contiguous) dim
                for (int64_t i = 0; i < index_dim_size; i++) {
                    char* self_ptr = self_bytes;
                    const char* src_ptr = src_bytes;
                    const char* index_ptr = index_bytes + i * index_dim_stride * sizeof(int64_t);
                    for (int64_t k = 0; k < n; k++) {
                        int64_t idx_dim = *reinterpret_cast<const int64_t*>(index_ptr);
                        RAXIS_CHECK(idx_dim >= 0 && idx_dim < index_upper_bound,
                            "index ", idx_dim, " is out of bounds for dimension ", dim,
                            " with size ", index_upper_bound);
                        f(reinterpret_cast<scalar_t*>(self_ptr) + (is_scatter_like ? idx_dim : i) * self_dim_stride,
                          reinterpret_cast<const scalar_t*>(src_ptr) + (is_scatter_like ? i : idx_dim) * src_dim_stride);
                        self_ptr += strides[0];
                        src_ptr += src_iter_stride;
                        index_ptr += strides[2];
                    }
                }
            }
        };
        iter.for_each(loop);
    });
}

} // namespace

void gather_kernel(const Tensor& result, int64_t dim, const Tensor& index, const Tensor& src) {
    scatter_gather_base_kernel</*is_scatter_like=*/false>(result, dim, index, &src, 0, TensorAssign{});
}

void scatter_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    scatter_gather_base_kernel</*is_scatter_like=*/true>(self, dim, index, &src, 0, TensorAssign{});
}

void scatter_fill_kernel(const Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    scatter_gather_base_kernel</*is_scatter_like=*/true>(self, dim, index, nullptr, value, TensorAssign{});
}

void scatter_add_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    scatter_gather_base_kernel</*is_scatter_like=*/true>(self, dim, index, &src, 0, ReduceAdd{});
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(gather_stub, &gather_kernel);
REGISTER_DISPATCH(scatter_stub, &scatter_kernel);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_kernel);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_kernel);

} // namespace raxis
