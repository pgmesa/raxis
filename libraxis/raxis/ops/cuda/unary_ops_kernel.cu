#include <cmath>
#include <type_traits>

#include <raxis/ops/cuda/loops.cuh>
#include <raxis/ops/unary_ops.h>
#include <raxis/tensor/tensor_iterator.h>


// From: aten/src/ATen/native/cuda/UnaryOpsKernel.cu, UnaryGeometric*.cu,
// UnarySignKernels.cu, UnaryLogKernels.cu, ...
// Same dtypes and semantics as the CPU kernels (ops/cpu/unary_ops_kernel.cpp).
// An integer input of a floating op (exp of an int tensor) has a floating
// common dtype: gpu_kernel reads it as int and converts it in registers.
//
// The math functions are CUDA's device overloads (float and double);
// PyTorch calls the same ones (::exp, ::log1p, ...).

namespace raxis {
namespace {

// Device lambdas cannot live inside the templated lambdas of the dtype
// dispatch, so every op is a functor template on its scalar type
template <template <typename> class Functor>
void floating_unary_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, Functor<scalar_t>());
    });
}

#define RAXIS_FLOATING_UNARY_OP(name, expr)                                    \
    template <typename T>                                                     \
    struct name##_functor {                                                   \
        __device__ T operator()(T a) const { return expr; }                   \
    };

RAXIS_FLOATING_UNARY_OP(ceil, ::ceil(a))
RAXIS_FLOATING_UNARY_OP(floor, ::floor(a))
RAXIS_FLOATING_UNARY_OP(round, ::nearbyint(a))
RAXIS_FLOATING_UNARY_OP(trunc, ::trunc(a))
RAXIS_FLOATING_UNARY_OP(frac, a - ::trunc(a))
RAXIS_FLOATING_UNARY_OP(exp, ::exp(a))
RAXIS_FLOATING_UNARY_OP(expm1, ::expm1(a))
RAXIS_FLOATING_UNARY_OP(log, ::log(a))
RAXIS_FLOATING_UNARY_OP(log2, ::log2(a))
RAXIS_FLOATING_UNARY_OP(log10, ::log10(a))
RAXIS_FLOATING_UNARY_OP(log1p, ::log1p(a))
RAXIS_FLOATING_UNARY_OP(sqrt, ::sqrt(a))
RAXIS_FLOATING_UNARY_OP(rsqrt, ::rsqrt(a))
RAXIS_FLOATING_UNARY_OP(reciprocal, T(1) / a)
RAXIS_FLOATING_UNARY_OP(sigmoid, T(1) / (T(1) + ::exp(-a)))
RAXIS_FLOATING_UNARY_OP(sin, ::sin(a))
RAXIS_FLOATING_UNARY_OP(cos, ::cos(a))
RAXIS_FLOATING_UNARY_OP(tan, ::tan(a))
RAXIS_FLOATING_UNARY_OP(asin, ::asin(a))
RAXIS_FLOATING_UNARY_OP(acos, ::acos(a))
RAXIS_FLOATING_UNARY_OP(atan, ::atan(a))
RAXIS_FLOATING_UNARY_OP(sinh, ::sinh(a))
RAXIS_FLOATING_UNARY_OP(cosh, ::cosh(a))
RAXIS_FLOATING_UNARY_OP(tanh, ::tanh(a))
RAXIS_FLOATING_UNARY_OP(erf, ::erf(a))
RAXIS_FLOATING_UNARY_OP(erfc, ::erfc(a))

#undef RAXIS_FLOATING_UNARY_OP

template <typename T>
struct NegFunctor {
    __device__ T operator()(T a) const { return -a; }
};

template <typename T>
struct AbsFunctor {
    __device__ T operator()(T a) const {
        if constexpr (std::is_floating_point_v<T>) {
            return ::fabs(a);  // clears the sign of -0 and -NaN too
        } else if constexpr (std::is_unsigned_v<T>) {
            return a;
        } else {
            return a < T(0) ? -a : a;
        }
    }
};

// (0 < a) - (a < 0): 0 for 0, -0 and NaN
template <typename T>
struct SignFunctor {
    __device__ T operator()(T a) const {
        if constexpr (std::is_unsigned_v<T>) {
            return static_cast<T>(a > 0);
        } else {
            return static_cast<T>((T(0) < a) - (a < T(0)));
        }
    }
};

template <typename T>
struct LogicalNotFunctor {
    __device__ bool operator()(T a) const { return !static_cast<bool>(a); }
};

void neg_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, NegFunctor<scalar_t>());
    });
}

// bool counts as unsigned (std::is_unsigned_v<bool>): abs and sign of a
// bool are the bool itself, as on the CPU
void abs_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, AbsFunctor<scalar_t>());
    });
}

void sign_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, SignFunctor<scalar_t>());
    });
}

// ~ of a bool promotes it to int (~true is -2): bool is !a
template <typename T>
struct BitwiseNotFunctor {
    __device__ T operator()(T a) const {
        if constexpr (std::is_same_v<T, bool>) {
            return !a;
        } else {
            return static_cast<T>(~a);
        }
    }
};

void bitwise_not_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_integral_and_bool_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, BitwiseNotFunctor<scalar_t>());
    });
}

void logical_not_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, LogicalNotFunctor<scalar_t>());
    });
}

} // anonymous namespace

REGISTER_DISPATCH(neg_stub, &neg_kernel_cuda);
REGISTER_DISPATCH(abs_stub, &abs_kernel_cuda);
REGISTER_DISPATCH(sign_stub, &sign_kernel_cuda);
REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel_cuda);
REGISTER_DISPATCH(ceil_stub, &floating_unary_kernel<ceil_functor>);
REGISTER_DISPATCH(floor_stub, &floating_unary_kernel<floor_functor>);
REGISTER_DISPATCH(round_stub, &floating_unary_kernel<round_functor>);
REGISTER_DISPATCH(trunc_stub, &floating_unary_kernel<trunc_functor>);
REGISTER_DISPATCH(frac_stub, &floating_unary_kernel<frac_functor>);
REGISTER_DISPATCH(bitwise_not_stub, &bitwise_not_kernel_cuda);
REGISTER_DISPATCH(exp_stub, &floating_unary_kernel<exp_functor>);
REGISTER_DISPATCH(expm1_stub, &floating_unary_kernel<expm1_functor>);
REGISTER_DISPATCH(log_stub, &floating_unary_kernel<log_functor>);
REGISTER_DISPATCH(log2_stub, &floating_unary_kernel<log2_functor>);
REGISTER_DISPATCH(log10_stub, &floating_unary_kernel<log10_functor>);
REGISTER_DISPATCH(log1p_stub, &floating_unary_kernel<log1p_functor>);
REGISTER_DISPATCH(sqrt_stub, &floating_unary_kernel<sqrt_functor>);
REGISTER_DISPATCH(rsqrt_stub, &floating_unary_kernel<rsqrt_functor>);
REGISTER_DISPATCH(reciprocal_stub, &floating_unary_kernel<reciprocal_functor>);
REGISTER_DISPATCH(sigmoid_stub, &floating_unary_kernel<sigmoid_functor>);
REGISTER_DISPATCH(sin_stub, &floating_unary_kernel<sin_functor>);
REGISTER_DISPATCH(cos_stub, &floating_unary_kernel<cos_functor>);
REGISTER_DISPATCH(tan_stub, &floating_unary_kernel<tan_functor>);
REGISTER_DISPATCH(asin_stub, &floating_unary_kernel<asin_functor>);
REGISTER_DISPATCH(acos_stub, &floating_unary_kernel<acos_functor>);
REGISTER_DISPATCH(atan_stub, &floating_unary_kernel<atan_functor>);
REGISTER_DISPATCH(sinh_stub, &floating_unary_kernel<sinh_functor>);
REGISTER_DISPATCH(cosh_stub, &floating_unary_kernel<cosh_functor>);
REGISTER_DISPATCH(tanh_stub, &floating_unary_kernel<tanh_functor>);
REGISTER_DISPATCH(erf_stub, &floating_unary_kernel<erf_functor>);
REGISTER_DISPATCH(erfc_stub, &floating_unary_kernel<erfc_functor>);

} // namespace raxis
