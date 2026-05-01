
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/fill.h>

#include <raxis/core/scalar.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace {
    
void fill_kernel(TensorIterator& iter, Scalar value) {
    iter.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        scalar_t val = value.to<scalar_t>();
        cpu_kernel_vec(iter,
            [val]() -> scalar_t { return val; },
            [val]() { return Vectorized<scalar_t>(val); });
    });
}

} // namespace

REGISTER_DISPATCH(fill_stub, &fill_kernel);

} // namespace raxis
