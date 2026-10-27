#include <limits>
#include <type_traits>

#include <raxis/ops/binary_ops.h>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/tensor/tensor_iterator.h>


// From: aten/src/ATen/native/cuda/BinaryAddSubKernel.cu, BinaryMulKernel.cu,
// BinaryDivTrueKernel.cu
// Same semantics as the CPU kernels (ops/cpu/binary_ops_kernel.cpp).

namespace raxis {
namespace {

template <typename T>
struct AddFunctor {
    __device__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct SubFunctor {
    __device__ T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct MulFunctor {
    __device__ T operator()(T a, T b) const { return a * b; }
};

// a && b for bool, as the CPU kernel (a * b would promote to int)
template <>
struct MulFunctor<bool> {
    __device__ bool operator()(bool a, bool b) const { return a && b; }
};

template <typename T>
struct DivFunctor {
    __device__ T operator()(T a, T b) const { return a / b; }
};

#define RAXIS_COMPARISON_FUNCTOR(name, op)                                         template <typename T>                                                         struct name {                                                                     __device__ bool operator()(T a, T b) const { return a op b; }            };

RAXIS_COMPARISON_FUNCTOR(GtFunctor, >)
RAXIS_COMPARISON_FUNCTOR(GeFunctor, >=)
RAXIS_COMPARISON_FUNCTOR(LtFunctor, <)
RAXIS_COMPARISON_FUNCTOR(LeFunctor, <=)
RAXIS_COMPARISON_FUNCTOR(EqFunctor, ==)
RAXIS_COMPARISON_FUNCTOR(NeFunctor, !=)

#undef RAXIS_COMPARISON_FUNCTOR

// NaN wins in both, as on the CPU (std::max alone would drop it when it
// is the second argument); for bool max is || and min is &&
template <typename T>
struct MaximumFunctor {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_same_v<T, bool>) {
            return a || b;
        } else {
            if constexpr (std::is_floating_point_v<T>) {
                if (a != a || b != b) return std::numeric_limits<T>::quiet_NaN();
            }
            return a > b ? a : b;
        }
    }
};

template <typename T>
struct MinimumFunctor {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_same_v<T, bool>) {
            return a && b;
        } else {
            if constexpr (std::is_floating_point_v<T>) {
                if (a != a || b != b) return std::numeric_limits<T>::quiet_NaN();
            }
            return a < b ? a : b;
        }
    }
};

template <typename T>
struct LogicalAndFunctor {
    __device__ bool operator()(T a, T b) const { return static_cast<bool>(a) && static_cast<bool>(b); }
};

template <typename T>
struct LogicalOrFunctor {
    __device__ bool operator()(T a, T b) const { return static_cast<bool>(a) || static_cast<bool>(b); }
};

// Every dtype, one functor template per op: comparisons, maximum/minimum
// and the logical ops dispatch alike
template <template <typename> class Functor>
void all_types_binary_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, Functor<scalar_t>());
    });
}

void add_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, AddFunctor<scalar_t>());
    });
}

void sub_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, SubFunctor<scalar_t>());
    });
}

void mul_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, MulFunctor<scalar_t>());
    });
}

void div_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, DivFunctor<scalar_t>());
    });
}

} // anonymous namespace

REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
REGISTER_DISPATCH(sub_stub, &sub_kernel_cuda);
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);
REGISTER_DISPATCH(div_stub, &div_kernel_cuda);
REGISTER_DISPATCH(gt_stub, &all_types_binary_kernel<GtFunctor>);
REGISTER_DISPATCH(ge_stub, &all_types_binary_kernel<GeFunctor>);
REGISTER_DISPATCH(lt_stub, &all_types_binary_kernel<LtFunctor>);
REGISTER_DISPATCH(le_stub, &all_types_binary_kernel<LeFunctor>);
REGISTER_DISPATCH(eq_stub, &all_types_binary_kernel<EqFunctor>);
REGISTER_DISPATCH(ne_stub, &all_types_binary_kernel<NeFunctor>);
REGISTER_DISPATCH(maximum_stub, &all_types_binary_kernel<MaximumFunctor>);
REGISTER_DISPATCH(minimum_stub, &all_types_binary_kernel<MinimumFunctor>);
REGISTER_DISPATCH(logical_and_stub, &all_types_binary_kernel<LogicalAndFunctor>);
REGISTER_DISPATCH(logical_or_stub, &all_types_binary_kernel<LogicalOrFunctor>);

} // namespace raxis
