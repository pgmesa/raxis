#include <raxis/ops/sorting.h>

#include <algorithm>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>


namespace raxis {

namespace ops {

// From: aten/src/ATen/native/Sorting.cpp (sort_out, sort_stable_out)
std::tuple<Tensor, Tensor> sort(const Tensor& self, int64_t dim, bool descending, bool /*stable*/) {
    const int64_t wrapped = wrap_dim(dim, std::max<int64_t>(self.ndim(), 1));
    Tensor values = empty_like(self);
    Tensor indices = empty_like(self, self.options().dtype(dtypes::Int64));
    if (self.ndim() == 0) {
        values.copy_(self);
        indices.fill_(0);
    } else if (self.numel() > 0) {
        sort_stub(self.device().type(), values, indices, self, wrapped, descending);
    }
    return {values, indices};
}

// From: aten/src/ATen/native/Sorting.cpp (topk_out_cpu)
std::tuple<Tensor, Tensor> topk(const Tensor& self, int64_t k, int64_t dim, bool largest, bool sorted) {
    const int64_t wrapped = wrap_dim(dim, std::max<int64_t>(self.ndim(), 1));
    const int64_t size = self.ndim() == 0 ? 1 : self.shape()[wrapped];
    RAXIS_CHECK(k >= 0 && k <= size, "selected index k out of range");

    DimVector shape(self.shape().begin(), self.shape().end());
    if (self.ndim() > 0) {
        shape[wrapped] = k;
    }
    Tensor values = empty(shape, self.options());
    Tensor indices = empty(shape, self.options().dtype(dtypes::Int64));
    if (self.ndim() == 0) {
        values.copy_(self);
        indices.fill_(0);
    } else if (k > 0 && self.numel() > 0) {
        topk_stub(self.device().type(), values, indices, self, k, wrapped, largest, sorted);
    }
    return {values, indices};
}

} // namespace ops

DEFINE_DISPATCH(sort_stub);
DEFINE_DISPATCH(topk_stub);

} // namespace raxis
