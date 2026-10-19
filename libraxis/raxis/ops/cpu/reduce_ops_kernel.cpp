
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/numeric_utils.h>
#include <raxis/ops/shared_reduce_ops.h>
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

// From: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp
// When the reduced dim is the contiguous innermost one, each output element
// is a plain loop over one row. Otherwise binary_kernel_reduce walks the
// strided elements through a sub-iterator per output element.
template <template <typename> class Ops>
void arg_reduce_kernel(TensorIterator& iter, bool find_max) {
    iter.dtype(1).dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using arg_t = std::pair<scalar_t, int64_t>;
        const arg_t init(find_max ? lower_bound<scalar_t>() : upper_bound<scalar_t>(), 0);
        if (is_reduce_lastdim(iter)) {
            binary_kernel_reduce_lastdim(iter, [&](char* out, char* in, int64_t size) {
                const auto* data = reinterpret_cast<const scalar_t*>(in);
                arg_t acc = init;
                for (int64_t i = 0; i < size; i++) {
                    acc = Ops<scalar_t>::reduce(acc, data[i], i);
                }
                *reinterpret_cast<int64_t*>(out) = acc.second;
            });
            return;
        }
        binary_kernel_reduce<scalar_t>(iter, Ops<scalar_t>{}, init);
    });
}

// From: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp
// Welford per output element, accumulating in double
void std_var_kernel(TensorIterator& iter, double correction, bool take_sqrt) {
    iter.dtype(0).dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        binary_kernel_reduce<scalar_t>(
            iter,
            WelfordOps<scalar_t, double, int64_t, scalar_t>{correction, take_sqrt},
            WelfordData<double, int64_t>());
    });
}

void argmax_kernel(TensorIterator& iter) {
    arg_reduce_kernel<ArgMaxOps>(iter, /*find_max=*/true);
}

void argmin_kernel(TensorIterator& iter) {
    arg_reduce_kernel<ArgMinOps>(iter, /*find_max=*/false);
}

} // namespace

REGISTER_DISPATCH(mean_stub, nullptr);
REGISTER_DISPATCH(prod_stub, &prod_kernel);
REGISTER_DISPATCH(max_values_stub, &max_values_kernel);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel);
REGISTER_DISPATCH(std_var_stub, &std_var_kernel);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel);

} // namespace raxis