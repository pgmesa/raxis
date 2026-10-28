#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <raxis/ops/cuda/reduce.cuh>
#include <raxis/ops/numeric_utils.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/util/small_vector.h>


// From: aten/src/ATen/native/cuda/ReduceSumProdKernel.cu,
// ReduceMomentKernel.cu, ReduceMaxValuesKernel.cu, ReduceArgMaxKernel.cu,
// ReduceLogicKernel.cu and SharedReduceOps.h
// The ops of the reductions in reduce.cuh. make_reduction already cast the
// input to the output dtype (except for argmax / argmin), so value
// reductions accumulate in that dtype, as PyTorch on CUDA does for float
// (its CPU sum uses a cascade instead: the two give different roundings).

namespace raxis {
namespace {

// __shfl_down_sync has overloads for 32 and 64-bit types; narrower ones
// travel as int
template <typename T>
__device__ T shfl_down(T value, int delta) {
    if constexpr (sizeof(T) < sizeof(int)) {
        return static_cast<T>(__shfl_down_sync(0xffffffff, static_cast<int>(value), delta));
    } else if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
        return static_cast<T>(__shfl_down_sync(0xffffffff, static_cast<long long>(value), delta));
    } else {
        return __shfl_down_sync(0xffffffff, value, delta);
    }
}

template <typename acc_t>
struct SumOps {
    __device__ acc_t reduce(acc_t acc, acc_t value, int) const { return acc + value; }
    __device__ acc_t combine(acc_t a, acc_t b) const { return a + b; }
    __device__ acc_t project(acc_t a) const { return a; }
    __device__ acc_t warp_shfl_down(acc_t a, int offset) const { return shfl_down(a, offset); }
};

// NaN counts as 0
template <typename acc_t>
struct NanSumOps {
    __device__ acc_t reduce(acc_t acc, acc_t value, int) const { return acc + (value != value ? acc_t(0) : value); }
    __device__ acc_t combine(acc_t a, acc_t b) const { return a + b; }
    __device__ acc_t project(acc_t a) const { return a; }
    __device__ acc_t warp_shfl_down(acc_t a, int offset) const { return shfl_down(a, offset); }
};

// The sum times num_outputs / numel, applied once to the result
template <typename acc_t>
struct MeanOps {
    acc_t factor;
    __device__ acc_t reduce(acc_t acc, acc_t value, int) const { return acc + value; }
    __device__ acc_t combine(acc_t a, acc_t b) const { return a + b; }
    __device__ acc_t project(acc_t a) const { return a * factor; }
    __device__ acc_t warp_shfl_down(acc_t a, int offset) const { return shfl_down(a, offset); }
};

template <typename acc_t>
struct ProdOps {
    __device__ acc_t reduce(acc_t acc, acc_t value, int) const { return acc * value; }
    __device__ acc_t combine(acc_t a, acc_t b) const { return a * b; }
    __device__ acc_t project(acc_t a) const { return a; }
    __device__ acc_t warp_shfl_down(acc_t a, int offset) const { return shfl_down(a, offset); }
};

// From: SharedReduceOps.h (MaxNanFunctor, MinNanFunctor): a NaN on either
// side wins, so one NaN makes the result NaN
template <typename T>
__device__ bool is_nan(T a) {
    if constexpr (std::is_floating_point_v<T>) {
        return a != a;
    } else {
        return false;
    }
}

template <typename acc_t>
struct MaxOps {
    __device__ acc_t combine(acc_t a, acc_t b) const { return (is_nan(a) || a > b) ? a : b; }
    __device__ acc_t reduce(acc_t acc, acc_t value, int) const { return combine(acc, value); }
    __device__ acc_t project(acc_t a) const { return a; }
    __device__ acc_t warp_shfl_down(acc_t a, int offset) const { return shfl_down(a, offset); }
};

template <typename acc_t>
struct MinOps {
    __device__ acc_t combine(acc_t a, acc_t b) const { return (is_nan(a) || a < b) ? a : b; }
    __device__ acc_t reduce(acc_t acc, acc_t value, int) const { return combine(acc, value); }
    __device__ acc_t project(acc_t a) const { return a; }
    __device__ acc_t warp_shfl_down(acc_t a, int offset) const { return shfl_down(a, offset); }
};

struct AndOps {
    __device__ bool reduce(bool acc, bool value, int) const { return acc && value; }
    __device__ bool combine(bool a, bool b) const { return a && b; }
    __device__ bool project(bool a) const { return a; }
    __device__ bool warp_shfl_down(bool a, int offset) const { return shfl_down(a, offset); }
};

struct OrOps {
    __device__ bool reduce(bool acc, bool value, int) const { return acc || value; }
    __device__ bool combine(bool a, bool b) const { return a || b; }
    __device__ bool project(bool a) const { return a; }
    __device__ bool warp_shfl_down(bool a, int offset) const { return shfl_down(a, offset); }
};

// (value, index) pairs. From: SharedReduceOps.h (ArgMaxOps, greater_or_nan):
// NaN is the largest value, and among equal values the first index wins,
// so the result does not depend on how the threads split the reduction
template <typename scalar_t>
struct ValueIndex {
    scalar_t value;
    int64_t index;
};

template <typename scalar_t, bool find_max>
struct ArgOps {
    using arg_t = ValueIndex<scalar_t>;

    __device__ arg_t combine(arg_t a, arg_t b) const {
        const bool a_nan = is_nan(a.value);
        const bool b_nan = is_nan(b.value);
        if (a_nan || b_nan) {
            if (a_nan && b_nan) return a.index < b.index ? a : b;
            return a_nan ? a : b;
        }
        if (a.value == b.value) return a.index < b.index ? a : b;
        if constexpr (find_max) {
            return a.value > b.value ? a : b;
        } else {
            return a.value < b.value ? a : b;
        }
    }
    __device__ arg_t reduce(arg_t acc, scalar_t value, int idx) const { return combine(acc, arg_t{value, idx}); }
    __device__ int64_t project(arg_t a) const { return a.index; }
    __device__ arg_t warp_shfl_down(arg_t a, int offset) const {
        return {shfl_down(a.value, offset), shfl_down(a.index, offset)};
    }
};

// Running mean and M2 = sum((x - mean)^2), as the CPU's WelfordOps
// (ops/shared_reduce_ops.h): n counts in int32 without rounding, nf is n as
// a float for combine, where the sum of two counts could overflow int32.
// From: SharedReduceOps.h (WelfordData, WelfordOps)
template <typename acc_scalar_t>
struct WelfordData {
    acc_scalar_t mean;
    acc_scalar_t m2;
    int32_t n;
    acc_scalar_t nf;
};

template <typename scalar_t, typename acc_scalar_t>
struct WelfordOps {
    using acc_t = WelfordData<acc_scalar_t>;

    acc_scalar_t correction;
    bool take_sqrt;

    __device__ acc_t reduce(acc_t acc, scalar_t data, int) const {
        const int32_t new_n = acc.n + 1;
        const auto new_nf = static_cast<acc_scalar_t>(new_n);
        const acc_scalar_t delta = data - acc.mean;
        const acc_scalar_t new_mean = acc.mean + delta / new_nf;
        const acc_scalar_t new_delta = data - new_mean;
        return {new_mean, acc.m2 + delta * new_delta, new_n, new_nf};
    }
    // Chan et al.: two partial results combine exactly
    __device__ acc_t combine(acc_t a, acc_t b) const {
        if (a.nf == 0) return b;
        if (b.nf == 0) return a;
        const acc_scalar_t delta = b.mean - a.mean;
        const acc_scalar_t new_count = a.nf + b.nf;
        const acc_scalar_t nb_over_n = b.nf / new_count;
        return {a.mean + delta * nb_over_n, a.m2 + b.m2 + delta * delta * a.nf * nb_over_n, -1, new_count};
    }
    // var = M2 / (n - correction); with n <= correction the divisor is 0
    __device__ scalar_t project(acc_t acc) const {
        const acc_scalar_t divisor = acc.nf > correction ? acc.nf - correction : 0;
        const acc_scalar_t var = acc.m2 / divisor;
        return static_cast<scalar_t>(take_sqrt ? ::sqrt(var) : var);
    }
    __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
        return {shfl_down(acc.mean, offset), shfl_down(acc.m2, offset), shfl_down(acc.n, offset),
                shfl_down(acc.nf, offset)};
    }
};

void sum_kernel_cuda(TensorIterator& iter) {
    iter.dtype(0).dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_reduce_kernel<scalar_t, scalar_t>(iter, SumOps<scalar_t>{}, scalar_t(0));
    });
}

void nansum_kernel_cuda(TensorIterator& iter) {
    iter.dtype(0).dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_reduce_kernel<scalar_t, scalar_t>(iter, NanSumOps<scalar_t>{}, scalar_t(0));
    });
}

void mean_kernel_cuda(TensorIterator& iter) {
    iter.dtype(0).dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        // num_outputs / numel = 1 / inputs_per_output
        const auto factor = static_cast<scalar_t>(cuda::num_output_elements(iter)) / static_cast<scalar_t>(iter.numel());
        cuda::gpu_reduce_kernel<scalar_t, scalar_t>(iter, MeanOps<scalar_t>{factor}, scalar_t(0));
    });
}

void prod_kernel_cuda(TensorIterator& iter) {
    iter.dtype(0).dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_reduce_kernel<scalar_t, scalar_t>(iter, ProdOps<scalar_t>{}, scalar_t(1));
    });
}

void max_values_kernel_cuda(TensorIterator& iter) {
    iter.dtype(0).dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_reduce_kernel<scalar_t, scalar_t>(iter, MaxOps<scalar_t>{}, lower_bound<scalar_t>());
    });
}

void min_values_kernel_cuda(TensorIterator& iter) {
    iter.dtype(0).dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_reduce_kernel<scalar_t, scalar_t>(iter, MinOps<scalar_t>{}, upper_bound<scalar_t>());
    });
}

// max() / min() of the whole tensor: a reduction into a result viewed with
// every dim of size 1
template <void (*kernel)(TensorIterator&)>
void reduce_all_kernel_cuda(Tensor& result, const Tensor& input) {
    Tensor out = result.view(DimVector(input.ndim(), 1));
    auto iter = TensorIterator::reduce_op(out, input);
    kernel(iter);
}

// From: aten/src/ATen/native/cuda/ReduceMomentKernel.cu (std_var_kernel_impl)
void std_var_kernel_cuda(TensorIterator& iter, double correction, bool take_sqrt) {
    iter.dtype(0).dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        // Two accumulators per thread instead of four: a Welford state is
        // four values, and PyTorch cuts the unrolling to avoid register spills
        using Ops = WelfordOps<scalar_t, scalar_t>;
        cuda::gpu_reduce_kernel<scalar_t, scalar_t, /*vt0=*/2>(
            iter, Ops{static_cast<scalar_t>(correction), take_sqrt}, typename Ops::acc_t{0, 0, 0, 0});
    });
}

void argmax_kernel_cuda(TensorIterator& iter) {
    iter.dtype(1).dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_reduce_kernel<scalar_t, int64_t>(
            iter, ArgOps<scalar_t, true>{}, ValueIndex<scalar_t>{lower_bound<scalar_t>(), 0});
    });
}

void argmin_kernel_cuda(TensorIterator& iter) {
    iter.dtype(1).dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_reduce_kernel<scalar_t, int64_t>(
            iter, ArgOps<scalar_t, false>{}, ValueIndex<scalar_t>{upper_bound<scalar_t>(), 0});
    });
}

void and_kernel_cuda(TensorIterator& iter) {
    cuda::gpu_reduce_kernel<bool, bool>(iter, AndOps{}, true);
}

void or_kernel_cuda(TensorIterator& iter) {
    cuda::gpu_reduce_kernel<bool, bool>(iter, OrOps{}, false);
}

} // anonymous namespace

REGISTER_DISPATCH(sum_stub, &sum_kernel_cuda);
REGISTER_DISPATCH(nansum_stub, &nansum_kernel_cuda);
REGISTER_DISPATCH(mean_stub, &mean_kernel_cuda);
REGISTER_DISPATCH(prod_stub, &prod_kernel_cuda);
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_cuda);
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_cuda);
REGISTER_DISPATCH(max_all_stub, &reduce_all_kernel_cuda<&max_values_kernel_cuda>);
REGISTER_DISPATCH(min_all_stub, &reduce_all_kernel_cuda<&min_values_kernel_cuda>);
REGISTER_DISPATCH(std_var_stub, &std_var_kernel_cuda);
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_cuda);
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_cuda);
REGISTER_DISPATCH(and_stub, &and_kernel_cuda);
REGISTER_DISPATCH(or_stub, &or_kernel_cuda);

} // namespace raxis
