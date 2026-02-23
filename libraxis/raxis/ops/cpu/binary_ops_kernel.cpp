
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/binary_ops.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace {

void add_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return a + b;
        });
    });
}

void sub_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return a - b;
        });
    });
}

void mul_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if constexpr (std::is_same_v<scalar_t, bool>) {
            cpu_kernel(iter, [](bool a, bool b) -> bool {
                return a && b;
            });
        } else {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
                return a * b;
            });
        }
    });
}

void div_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return a / b;
        });
    });
}

} // namespace

REGISTER_CPU_DISPATCH(add_stub, &add_kernel);
REGISTER_CPU_DISPATCH(sub_stub, &sub_kernel);
REGISTER_CPU_DISPATCH(mul_stub, &mul_kernel);
REGISTER_CPU_DISPATCH(div_stub, &div_kernel);

} // namesapce raxis