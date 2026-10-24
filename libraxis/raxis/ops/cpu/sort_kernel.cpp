#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <raxis/ops/sorting.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/numeric_utils.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/tensor_shape.h>


namespace raxis {

inline namespace CPU_CAPABILITY {

namespace {

template <typename scalar_t>
using Entry = std::pair<scalar_t, int64_t>;

// From: aten/src/ATen/native/cpu/SortingKernel.cpp (KeyValueCompAsc/Desc)
// NaN counts as larger than every number (as numpy), so it goes last in an
// ascending sort and first in a descending one. Unlike PyTorch, equal values
// are ordered by position: std::sort then gives the same result as a stable
// sort, the result does not depend on the sort algorithm, and one comparator
// serves both `stable` values.
template <typename scalar_t, bool descending>
struct EntryOrder {
    bool operator()(const Entry<scalar_t>& a, const Entry<scalar_t>& b) const {
        const bool a_nan = _isnan<scalar_t>(a.first);
        const bool b_nan = _isnan<scalar_t>(b.first);
        if (a_nan != b_nan) {
            return descending ? a_nan : b_nan;
        }
        if (!a_nan && a.first != b.first) {
            return descending ? a.first > b.first : a.first < b.first;
        }
        return a.second < b.second;
    }
};

template <typename Func>
void dispatch_order(bool descending, Func&& f) {
    if (descending) {
        f(std::true_type{});
    } else {
        f(std::false_type{});
    }
}

// From: aten/src/ATen/native/cpu/SortingKernel.cpp (_dim_apply)
// Runs f on every slice along dim: an iterator over the outputs with dim
// squashed to size 1, so each element of the loop is the start of a slice.
// self is viewed with the outputs' shape (topk's outputs are shorter along
// dim); only its strides matter, the kernel walks the whole slice. Each
// chunk of slices (one per thread) gets its own buffer for f.
template <typename scalar_t, typename Func>
void for_each_slice(const Tensor& values, const Tensor& indices, const Tensor& self, int64_t dim,
                    int64_t slice_size, const Func& f) {
    const Tensor self_view = make_view(
        self, self.storage_offset(), values.shape(), self.strides(), /*is_differentiable_view=*/false);
    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .check_mem_overlap(false)
        .declare_static_shape(values.shape(), {dim})
        .add_output(values)
        .add_output(indices)
        .add_input(self_view)
        .build();

    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        std::vector<Entry<scalar_t>> buffer(slice_size);
        for (int64_t i = 0; i < n; i++) {
            f(reinterpret_cast<scalar_t*>(data[0] + i * strides[0]),
              reinterpret_cast<int64_t*>(data[1] + i * strides[1]),
              reinterpret_cast<const scalar_t*>(data[2] + i * strides[2]),
              buffer);
        }
    };
    iter.for_each(loop, /*grain_size=*/std::max<int64_t>(1, detail::GRAIN_SIZE / slice_size));
}

// PyTorch sorts values and indices in place through a "composite" iterator
// that swaps both at once. Copying each slice into a buffer of (value,
// position) pairs instead is simpler, sorts contiguous memory whatever the
// strides, and is the same buffer topk uses.
void sort_kernel(const Tensor& values, const Tensor& indices, const Tensor& self,
                 int64_t dim, bool descending) {
    const int64_t size = self.shape()[dim];
    const int64_t self_stride = self.strides()[dim];
    const int64_t values_stride = values.strides()[dim];
    const int64_t indices_stride = indices.strides()[dim];

    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        dispatch_order(descending, [&](auto desc) {
            for_each_slice<scalar_t>(values, indices, self, dim, size,
                    [&](scalar_t* out_data, int64_t* idx, const scalar_t* in_data, std::vector<Entry<scalar_t>>& buffer) {
                for (int64_t j = 0; j < size; j++) {
                    buffer[j] = {in_data[j * self_stride], j};
                }
                std::sort(buffer.begin(), buffer.end(), EntryOrder<scalar_t, decltype(desc)::value>{});
                for (int64_t j = 0; j < size; j++) {
                    out_data[j * values_stride] = buffer[j].first;
                    idx[j * indices_stride] = buffer[j].second;
                }
            });
        });
    });
}

// From: aten/src/ATen/native/TopKImpl.h (topk_impl_loop)
// For a small k, partial_sort (a heap of k elements over the slice, about
// n log k). Otherwise nth_element puts the k-th entry in place with the
// k - 1 better ones before it in O(n), and only those get sorted.
void topk_kernel(const Tensor& values, const Tensor& indices, const Tensor& self,
                 int64_t k, int64_t dim, bool largest, bool sorted) {
    const int64_t size = self.shape()[dim];
    const int64_t self_stride = self.strides()[dim];
    const int64_t values_stride = values.strides()[dim];
    const int64_t indices_stride = indices.strides()[dim];
    const bool use_partial_sort = k * 64 <= size;

    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        dispatch_order(largest, [&](auto desc) {
            const EntryOrder<scalar_t, decltype(desc)::value> order;
            for_each_slice<scalar_t>(values, indices, self, dim, size,
                    [&](scalar_t* out_data, int64_t* idx, const scalar_t* in_data, std::vector<Entry<scalar_t>>& buffer) {
                for (int64_t j = 0; j < size; j++) {
                    buffer[j] = {in_data[j * self_stride], j};
                }
                if (use_partial_sort) {
                    std::partial_sort(buffer.begin(), buffer.begin() + k, buffer.end(), order);
                } else {
                    std::nth_element(buffer.begin(), buffer.begin() + k - 1, buffer.end(), order);
                    if (sorted) {
                        std::sort(buffer.begin(), buffer.begin() + k - 1, order);
                    }
                }
                for (int64_t j = 0; j < k; j++) {
                    out_data[j * values_stride] = buffer[j].first;
                    idx[j * indices_stride] = buffer[j].second;
                }
            });
        });
    });
}

} // namespace

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(sort_stub, &sort_kernel);
REGISTER_DISPATCH(topk_stub, &topk_kernel);

} // namespace raxis
