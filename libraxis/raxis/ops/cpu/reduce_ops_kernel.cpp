
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/reduce_ops.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>

#include <limits>
#include <cmath>
#include <utility>


namespace raxis {

namespace {

} // namespace

REGISTER_CPU_DISPATCH(mean_stub, nullptr);

} // namespace raxis