#include <raxis/ops/activation.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>

#include <algorithm>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

void relu_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                // a != a keeps NaN, as PyTorch's clamp_min(self, 0) does
                if constexpr (std::is_floating_point_v<scalar_t>) {
                    if (a != a) return a;
                }
                return std::max(a, static_cast<scalar_t>(0));
            },
            [](Vectorized<scalar_t> a) {
                return vec::maximum(a, Vectorized<scalar_t>(static_cast<scalar_t>(0)));
            }
        );
    });
}

// From: aten/src/ATen/native/cpu/Activation.cpp (threshold_kernel)
// Inputs are (self, grad): the output keeps grad where self > threshold.
// The mask select is branchless in the vectorized path (blendv).
void threshold_backward_kernel(TensorIterator& iter, Scalar threshold_scalar) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using Vec = Vectorized<scalar_t>;
        scalar_t threshold = threshold_scalar.to<scalar_t>();
        Vec threshold_vec(threshold);
        Vec zero_vec(static_cast<scalar_t>(0));
        cpu_kernel_vec(iter,
            [threshold](scalar_t self, scalar_t grad) -> scalar_t {
                return self <= threshold ? static_cast<scalar_t>(0) : grad;
            },
            [threshold_vec, zero_vec](Vec self, Vec grad) {
                return Vec::blendv(grad, zero_vec, self <= threshold_vec);
            }
        );
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(relu_stub, &relu_kernel);
REGISTER_DISPATCH(threshold_backward_stub, &threshold_backward_kernel);

} // namespace raxis
