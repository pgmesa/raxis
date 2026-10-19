
#include <raxis/ops/tensor_compare.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/vec/vec.h>

#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>
#include <raxis/tensor/tensor_iterator.h>

#include <algorithm>


namespace raxis {

inline namespace CPU_CAPABILITY {

// From: aten/src/ATen/native/cpu/TensorCompareKernel.cpp
void where_kernel(TensorIterator& iter) {
    iter.dtype(0).dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](bool cond, scalar_t self, scalar_t other) -> scalar_t {
            return cond ? self : other;
        });
    });
}

// From: aten/src/ATen/native/cpu/TensorCompareKernel.cpp
// min(max(a, lo), hi): with lo > hi every element becomes hi, as in torch.
// A NaN element stays NaN without extra work: std::max/min return their first
// argument when the comparison is false, and vec::clamp puts `a` second in
// maxps/minps, which return the second operand when either one is NaN
void clamp_scalar_kernel(TensorIterator& iter, Scalar min_, Scalar max_) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const auto min = min_.to<scalar_t>();
        const auto max = max_.to<scalar_t>();
        const Vectorized<scalar_t> min_vec(min);
        const Vectorized<scalar_t> max_vec(max);
        cpu_kernel_vec(iter,
            [=](scalar_t a) -> scalar_t {
                return std::min(std::max(a, min), max);
            },
            [=](Vectorized<scalar_t> a) {
                return vec::clamp(a, min_vec, max_vec);
            }
        );
    });
}

void clamp_min_scalar_kernel(TensorIterator& iter, Scalar min_) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const auto min = min_.to<scalar_t>();
        const Vectorized<scalar_t> min_vec(min);
        cpu_kernel_vec(iter,
            [=](scalar_t a) -> scalar_t {
                return std::max(a, min);
            },
            [=](Vectorized<scalar_t> a) {
                return vec::clamp_min(a, min_vec);
            }
        );
    });
}

void clamp_max_scalar_kernel(TensorIterator& iter, Scalar max_) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const auto max = max_.to<scalar_t>();
        const Vectorized<scalar_t> max_vec(max);
        cpu_kernel_vec(iter,
            [=](scalar_t a) -> scalar_t {
                return std::min(a, max);
            },
            [=](Vectorized<scalar_t> a) {
                return vec::clamp_max(a, max_vec);
            }
        );
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(where_stub, &where_kernel);
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel);
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel);
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel);

} // namespace raxis
