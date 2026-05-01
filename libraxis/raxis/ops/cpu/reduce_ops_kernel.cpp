
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/numeric_utils.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/reduce.h>
#include <raxis/ops/cpu/vec/vec.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>

namespace raxis {

namespace {

void prod_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) {
            if constexpr (std::is_same_v<scalar_t, bool>) {
                cpu_reduce_kernel_vec<scalar_t>(
                    iter,
                    [](scalar_t a, scalar_t b) -> scalar_t { return a && b; },
                    [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a && b; },
                    /*ident=*/1);
            } else {
                cpu_reduce_kernel_vec<scalar_t>(
                    iter,
                    [](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
                    [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a * b; },
                    /*ident=*/1);
            }
        }
    );
}

void max_values_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_reduce_kernel_vec<scalar_t>(
            iter,
            [](scalar_t a, scalar_t b) -> scalar_t { return max_impl(a, b); },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return vec::maximum(a, b); },
            /*ident=*/lower_bound<scalar_t>());
    });
}

void min_values_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_reduce_kernel_vec<scalar_t>(
            iter,
            [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return vec::minimum(a, b); },
            /*ident=*/upper_bound<scalar_t>());
    });
}

} // namespace

REGISTER_DISPATCH(mean_stub, nullptr);
REGISTER_DISPATCH(prod_stub, &prod_kernel);
REGISTER_DISPATCH(max_values_stub, &max_values_kernel);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel);
// REGISTER_DISPATCH(argmax_stub, &argmax_kernel);
// REGISTER_DISPATCH(argmin_stub, &argmin_kernel);

} // namespace raxis