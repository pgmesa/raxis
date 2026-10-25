#include <raxis/ops/triangular_ops.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>

#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace {

// From: aten/src/ATen/native/TriangularOps.cpp (apply_triu_tril_single)
// Row i keeps the columns on the kept side of column i + k and zeroes the
// rest: two contiguous runs per row, no per-element test
template <typename scalar_t>
void apply_triu_tril_single(
    scalar_t* result, const scalar_t* self, int64_t k, int64_t n, int64_t m,
    int64_t self_row_stride, int64_t self_col_stride, bool upper
) {
    for (int64_t i = 0; i < n; ++i) {
        scalar_t* row = result + i * m;
        const scalar_t* self_row = self + i * self_row_stride;
        // Columns [0, split) belong to one triangle, [split, m) to the other
        const int64_t split = std::clamp(upper ? i + k : i + k + 1, int64_t{0}, m);
        const int64_t zero_begin = upper ? 0 : split;
        const int64_t zero_end = upper ? split : m;
        const int64_t copy_begin = upper ? split : 0;
        const int64_t copy_end = upper ? m : split;
        std::fill(row + zero_begin, row + zero_end, scalar_t(0));
        for (int64_t j = copy_begin; j < copy_end; ++j) {
            row[j] = self_row[j * self_col_stride];
        }
    }
}

// From: aten/src/ATen/native/TriangularOps.cpp (compute_triu_tril)
Tensor triu_tril(const Tensor& self, int64_t k, bool upper) {
    RAXIS_CHECK(self.ndim() >= 2, upper ? "triu" : "tril", ": input tensor must have at least 2 dimensions");
    Tensor result = empty(self.shape(), self.options());
    if (self.numel() == 0) {
        return result;
    }
    const int64_t n = self.shape()[self.ndim() - 2];
    const int64_t m = self.shape()[self.ndim() - 1];
    // Clamped so that i + k cannot overflow
    k = std::clamp(k, -n, m);
    // A matrix is read with its own strides (a transposed one is not
    // copied); a batch must step by whole matrices, so it is made contiguous
    const Tensor self_c = self.ndim() > 2 && !self.is_contiguous() ? self.contiguous() : self;
    const int64_t self_row_stride = self_c.strides()[self.ndim() - 2];
    const int64_t self_col_stride = self_c.strides()[self.ndim() - 1];
    const int64_t self_batch_stride = self.ndim() > 2 ? self_c.strides()[self.ndim() - 3] : 0;
    const int64_t batch_size = self.numel() / (n * m);

    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t* self_data = self_c.data_ptr<scalar_t>();
        scalar_t* result_data = result.mutable_data_ptr<scalar_t>();
        // A batch of matrices splits by matrix, a single matrix by rows
        if (batch_size > 1) {
            parallel_for(0, batch_size, 1, [&](int64_t begin, int64_t end) {
                for (int64_t b = begin; b < end; ++b) {
                    apply_triu_tril_single(result_data + b * n * m, self_data + b * self_batch_stride,
                                           k, n, m, self_row_stride, self_col_stride, upper);
                }
            });
        } else {
            parallel_for(0, n, std::max<int64_t>(1, detail::GRAIN_SIZE / m), [&](int64_t begin, int64_t end) {
                apply_triu_tril_single(result_data + begin * m, self_data + begin * self_row_stride,
                                       k + begin, end - begin, m, self_row_stride, self_col_stride, upper);
            });
        }
    });
    return result;
}

} // namespace


namespace ops {

Tensor tril(const Tensor& self, int64_t diagonal) {
    return triu_tril(self, diagonal, /*upper=*/false);
}

Tensor triu(const Tensor& self, int64_t diagonal) {
    return triu_tril(self, diagonal, /*upper=*/true);
}

} // namespace ops

} // namespace raxis
