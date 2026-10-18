#pragma once

#include <raxis/export.h>
#include <raxis/tensor/tensor.h>

namespace raxis {

RAXIS_API Tensor make_view(
    const Tensor& self,
    int64_t new_storage_offset,
    IntArrayRef new_shape,
    IntArrayRef new_strides
);
RAXIS_API Tensor alias(const Tensor& self);
RAXIS_API Tensor unsqueeze(const Tensor& self, int64_t dim);
RAXIS_API Tensor squeeze(const Tensor& self, std::optional<int64_t> dim_opt = std::nullopt);
RAXIS_API Tensor select(const Tensor& self, int64_t dim, int64_t index);
RAXIS_API Tensor slice(const Tensor& self, int64_t dim, const indexing::Slice& slice);
RAXIS_API Tensor slice(const Tensor& self, int64_t dim, int64_t start, int64_t stop, int64_t step = 1);
RAXIS_API Tensor narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length);

RAXIS_API Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1);
RAXIS_API Tensor permute(const Tensor& self, IntArrayRef dims);
RAXIS_API Tensor swapdims(const Tensor& self, int64_t dim0, int64_t dim1);
RAXIS_API Tensor movedim(const Tensor& self, IntArrayRef src, IntArrayRef dst);
RAXIS_API Tensor movedim(const Tensor& self, int64_t src, int64_t dst);
RAXIS_API Tensor view(const Tensor& self, IntArrayRef shape);
RAXIS_API Tensor reshape(const Tensor& self, IntArrayRef shape);
RAXIS_API Tensor flatten(const Tensor& self, int64_t start_dim = 0, int64_t end_dim = -1);

RAXIS_API Tensor broadcast_to(const Tensor& self, IntArrayRef shape);
RAXIS_API std::vector<Tensor> broadcast_tensors(ArrayRef<Tensor> tensors);

namespace ops {

RAXIS_API Tensor unsqueeze(const Tensor& self, int64_t dim);
RAXIS_API Tensor squeeze(const Tensor& self, std::optional<int64_t> dim_opt = std::nullopt);
RAXIS_API Tensor select(const Tensor& self, int64_t dim, int64_t index);
RAXIS_API Tensor slice(const Tensor& self, int64_t dim, int64_t start, int64_t stop, int64_t step = 1);

RAXIS_API Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1);
RAXIS_API Tensor permute(const Tensor& self, IntArrayRef dims);
RAXIS_API Tensor view(const Tensor& self, IntArrayRef shape);
RAXIS_API Tensor reshape(const Tensor& self, IntArrayRef shape);

RAXIS_API Tensor broadcast_to(const Tensor& self, IntArrayRef shape);
    
} // namespace ops


} // namespace raxis