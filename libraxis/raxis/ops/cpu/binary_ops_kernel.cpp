
#include <raxis/ops/cpu/binary_ops_kernel.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/ops/cpu/loops.h>


namespace raxis::kernels::cpu {

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


} // namesapce raxis::kernels::cpu