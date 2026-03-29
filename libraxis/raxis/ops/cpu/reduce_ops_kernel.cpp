
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/cpu/reduce.h>
#include <raxis/ops/numeric_utils.h>
#include <raxis/util/parallel.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>

#include <limits>
#include <cmath>
#include <utility>


namespace raxis {

namespace {

void prod_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) {
            if constexpr (std::is_same_v<scalar_t, bool>) {
                cpu_reduce_kernel<scalar_t>(
                    iter,
                    [](scalar_t a, scalar_t b) -> scalar_t { return a && b; },
                    /*ident=*/1);
            } else {
                cpu_reduce_kernel<scalar_t>(
                    iter,
                    [](scalar_t a, scalar_t b) -> scalar_t { return a * b; },
                    /*ident=*/1);
            }
        }
    );
}

template <typename scalar_t, typename func_t>
inline void reduce_all_impl(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op)
{
    const int64_t input_numel = input.numel();
    const auto* input_data = input.data_ptr<scalar_t>();

    scalar_t result = parallel_reduce(0, input_numel, detail::GRAIN_SIZE, ident_v,
        [&](int64_t start, int64_t end, scalar_t ident) -> scalar_t {
            scalar_t partial = ident;
            for (int64_t i = start; i < end; ++i) {
                partial = op(partial, input_data[i]);
            }
            return partial;
        }, op);

    output.fill_(result);
}

void max_all_kernel(Tensor& result, const Tensor& input) {
    input.dtype().dispatch_all_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) {
            reduce_all_impl<scalar_t>(
                result, input, lower_bound<scalar_t>(),
                [](scalar_t a, scalar_t b) -> scalar_t { return max_impl(a, b); });
        }
    );
}

void min_all_kernel(Tensor& result, const Tensor& input) {
    input.dtype().dispatch_all_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) {
            reduce_all_impl<scalar_t>(
                result, input, upper_bound<scalar_t>(),
                [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); });
        }
    );
}

void max_values_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_reduce_kernel<scalar_t>(iter, [](scalar_t a, scalar_t b) -> scalar_t { 
            return max_impl(a, b);
        },
        /*ident=*/lower_bound<scalar_t>());
    });
}

void min_values_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_reduce_kernel<scalar_t>(iter, [](scalar_t a, scalar_t b) -> scalar_t { 
            return min_impl(a, b);
        },
        /*ident=*/upper_bound<scalar_t>());
    });
}

} // namespace

REGISTER_CPU_DISPATCH(mean_stub, nullptr);
REGISTER_CPU_DISPATCH(prod_stub, &prod_kernel);
REGISTER_CPU_DISPATCH(max_all_stub, &max_all_kernel);
REGISTER_CPU_DISPATCH(min_all_stub, &min_all_kernel);
REGISTER_CPU_DISPATCH(max_values_stub, &max_values_kernel);
REGISTER_CPU_DISPATCH(min_values_stub, &min_values_kernel);
// REGISTER_CPU_DISPATCH(argmax_stub, &argmax_kernel);
// REGISTER_CPU_DISPATCH(argmin_stub, &argmin_kernel);

} // namespace raxis