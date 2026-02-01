
#include <raxis/kernels/cpu/binary_ops.hpp>

#include <raxis/dtype.hpp>
#include <raxis/core/array_iterator.hpp>
#include <raxis/kernels/cpu/loops.hpp>


namespace raxis::kernels::cpu {

void add_kernel(const ArrayIterator& iter) {
    iter.common_dtype().dispatch([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel<scalar_t>(iter, [](scalar_t a, scalar_t b) -> scalar_t {
            return a + b;
        });
    });
}

void mul_kernel(const ArrayIterator& iter) {
    iter.common_dtype().dispatch([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if constexpr (std::is_same_v<scalar_t, bool>) {
            cpu_kernel<bool>(iter, [](bool a, bool b) -> bool {
                return a && b;
            });
        } else {
            cpu_kernel<scalar_t>(iter, [](scalar_t a, scalar_t b) -> scalar_t {
                return a * b;
            });
        }
    });
}

}