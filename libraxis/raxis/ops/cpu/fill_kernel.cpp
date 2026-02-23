
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
        cpu_kernel(iter, [=]() -> scalar_t {
            return val;
        });
    });
}

} // namespace

REGISTER_CPU_DISPATCH(fill_stub, &fill_kernel);

} // namespace raxis
