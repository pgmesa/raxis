#include <raxis/ops/tensor_advanced_indexing.h>

#include <cstring>
#include <type_traits>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/memory_overlap.h>

#include <raxis/ops/copy.h>
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

} // namespace ops

DEFINE_DISPATCH(gather_stub);
DEFINE_DISPATCH(scatter_stub);
DEFINE_DISPATCH(scatter_fill_stub);
DEFINE_DISPATCH(scatter_add_stub);

} // namespace raxis
