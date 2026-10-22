
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/numeric_utils.h>
#include <raxis/ops/shared_reduce_ops.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/reduce.h>
#include <raxis/ops/cpu/vec/vec.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/util/parallel.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

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

// From: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp (and_kernel_impl,
// or_kernel_impl). The input was cast to bool, whose lanes hold 0 or 1, so
// the bitwise & and | are the logical ones and run on whole words.
void and_kernel(TensorIterator& iter) {
    cpu_reduce_kernel_vec<bool>(
        iter,
        [](bool a, bool b) -> bool { return a && b; },
        [](Vectorized<bool> a, Vectorized<bool> b) { return a & b; },
        /*ident=*/true);
}

void or_kernel(TensorIterator& iter) {
    cpu_reduce_kernel_vec<bool>(
        iter,
        [](bool a, bool b) -> bool { return a || b; },
        [](Vectorized<bool> a, Vectorized<bool> b) { return a | b; },
        /*ident=*/false);
}

// From: aten/src/ATen/native/cpu/ReduceOpsKernel.cpp (cpu_cum_base_kernel,
// cumsum_cpu_kernel). Elementwise over the positions of self with dim
// squashed to 1, each one scanning its whole slice along dim. The running
// sum is kept in double for float (int64 for integers), as PyTorch's
// acc_type on CPU, so a long float cumsum does not drift. reverse scans
// from the end with the same additions as flip + cumsum + flip.
void cumsum_kernel(const Tensor& result, const Tensor& self, int64_t dim, bool reverse) {
    const int64_t size = self.shape()[dim];
    const int64_t self_stride = self.strides()[dim];
    const int64_t result_stride = result.strides()[dim];

    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .declare_static_shape(self.shape(), {dim})
        .add_output(result)
        .add_input(self)
        .build();

    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using acc_t = std::conditional_t<std::is_floating_point_v<scalar_t>, double, int64_t>;
        // Walk from the last element with negative strides when reversed
        const int64_t first = reverse ? size - 1 : 0;
        const int64_t step_self = reverse ? -self_stride : self_stride;
        const int64_t step_result = reverse ? -result_stride : result_stride;
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
            char* result_bytes = data[0];
            const char* self_bytes = data[1];
            for (int64_t i = 0; i < n; i++) {
                scalar_t* out = reinterpret_cast<scalar_t*>(result_bytes) + first * result_stride;
                const scalar_t* in = reinterpret_cast<const scalar_t*>(self_bytes) + first * self_stride;
                acc_t acc = 0;
                for (int64_t j = 0; j < size; j++) {
                    acc += in[j * step_self];
                    out[j * step_result] = static_cast<scalar_t>(acc);
                }
                result_bytes += strides[0];
                self_bytes += strides[1];
            }
        };
        iter.for_each(loop, /*grain_size=*/std::max<int64_t>(1, detail::GRAIN_SIZE / size));
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
REGISTER_DISPATCH(cumsum_stub, &cumsum_kernel);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel);
REGISTER_DISPATCH(and_stub, &and_kernel);
REGISTER_DISPATCH(or_stub, &or_kernel);

} // namespace raxis