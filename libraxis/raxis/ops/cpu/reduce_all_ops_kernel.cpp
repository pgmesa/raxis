
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/numeric_utils.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/reduce.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/vec/functional_base.h>


#include <raxis/util/parallel.h>
#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace {

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

template <typename scalar_t, typename func_t, typename vec_func_t>
inline void reduce_all_impl_vec(
    Tensor& output,
    const Tensor& input,
    const scalar_t ident_v,
    func_t op,
    vec_func_t vop)
{
    using Vec = Vectorized<scalar_t>;
    const int64_t input_numel = input.numel();
    const auto* input_data = input.data_ptr<scalar_t>();

    scalar_t result = parallel_reduce(0, input_numel, detail::GRAIN_SIZE, ident_v,
        [&](int64_t start, int64_t end, const scalar_t /*ident*/) -> scalar_t {
            return vec::reduce_all<scalar_t>(
                [=](Vec x, Vec y) { return vop(x, y); },
                input_data + start,
                end - start);
        }, op);

    output.fill_(result);
}

void max_all_kernel(Tensor& result, const Tensor& input) {
    input.dtype().dispatch_all_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) {
            using Vec = Vectorized<scalar_t>;
            reduce_all_impl_vec<scalar_t>(
                result, input, lower_bound<scalar_t>(),
                [](scalar_t a, scalar_t b) -> scalar_t { return max_impl(a, b); },
                [](Vec a, Vec b) -> Vec { return vec::maximum(a, b); });
        }
    );
}

void min_all_kernel(Tensor& result, const Tensor& input) {
    input.dtype().dispatch_all_types(
        [&]<typename scalar_t>(std::type_identity<scalar_t>) {
            using Vec = Vectorized<scalar_t>;
            reduce_all_impl_vec<scalar_t>(
                result, input, upper_bound<scalar_t>(),
                [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },
                [](Vec a, Vec b) -> Vec { return vec::minimum(a, b); });
        }
    );
}

} // namespace

REGISTER_DISPATCH(max_all_stub, &max_all_kernel);
REGISTER_DISPATCH(min_all_stub, &min_all_kernel);

} // namespace raxis