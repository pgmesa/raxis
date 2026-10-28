#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

using cat_serial_fn = void(*)(const Tensor&, ArrayRef<Tensor>, int64_t);
using stack_serial_fn = void(*)(const Tensor&, ArrayRef<Tensor>, int64_t);
// GPU only: many contiguous inputs per launch into a contiguous result
using cat_batched_fn = void(*)(const Tensor&, ArrayRef<Tensor>, int64_t);

DECLARE_DISPATCH(cat_serial_fn, cat_serial_stub);
DECLARE_DISPATCH(stack_serial_fn, stack_serial_stub);
DECLARE_DISPATCH(cat_batched_fn, cat_batched_stub);

RAXIS_API Tensor cat(ArrayRef<Tensor> tensors, int64_t dim = 0);
RAXIS_API Tensor stack(ArrayRef<Tensor> tensors, int64_t dim = 0);
RAXIS_API std::vector<Tensor> unbind(const Tensor& self, int64_t dim = 0);

namespace ops {

RAXIS_API Tensor cat(ArrayRef<Tensor> tensors, int64_t dim = 0);
RAXIS_API Tensor stack(ArrayRef<Tensor> tensors, int64_t dim = 0);
RAXIS_API std::vector<Tensor> unbind(const Tensor& self, int64_t dim = 0);
    
} // namespace ops

} // namespace raxis