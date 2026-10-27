#include <type_traits>

#include <raxis/core/scalar.h>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/ops/pow.h>
#include <raxis/tensor/tensor_iterator.h>


// From: aten/src/ATen/native/cuda/PowKernel.cu
// The common scalar exponents become multiplications or a rsqrt instead of a
// pow, as in ops/cpu/pow_kernel.cpp; integers use repeated squaring (powi).

namespace raxis {
namespace {

template <typename T>
struct PowFunctor {
    __device__ T operator()(T base, T exp) const {
        if constexpr (std::is_floating_point_v<T>) {
            return ::pow(base, exp);
        } else {
            return powi(base, exp);
        }
    }
};

template <typename T>
struct PowScalarFunctor {
    T exp;
    __device__ T operator()(T base) const {
        if constexpr (std::is_floating_point_v<T>) {
            return ::pow(base, exp);
        } else {
            return powi(base, exp);
        }
    }
};

template <typename T>
struct SquareFunctor {
    __device__ T operator()(T base) const { return base * base; }
};

template <typename T>
struct CubeFunctor {
    __device__ T operator()(T base) const { return base * base * base; }
};

template <typename T>
struct InverseSquareFunctor {
    __device__ T operator()(T base) const { return T(1) / (base * base); }
};

template <typename T>
struct SqrtFunctor {
    __device__ T operator()(T base) const { return ::sqrt(base); }
};

template <typename T>
struct RsqrtFunctor {
    __device__ T operator()(T base) const { return ::rsqrt(base); }
};

template <typename T>
struct ReciprocalFunctor {
    __device__ T operator()(T base) const { return T(1) / base; }
};

void pow_tensor_tensor_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, PowFunctor<scalar_t>());
    });
}

void pow_tensor_scalar_kernel_cuda(TensorIterator& iter, Scalar exp_scalar) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if constexpr (std::is_floating_point_v<scalar_t>) {
            const double exp = exp_scalar.to<double>();
            if (exp == 2.0) {
                cuda::gpu_kernel(iter, SquareFunctor<scalar_t>());
            } else if (exp == 3.0) {
                cuda::gpu_kernel(iter, CubeFunctor<scalar_t>());
            } else if (exp == -2.0) {
                cuda::gpu_kernel(iter, InverseSquareFunctor<scalar_t>());
            } else if (exp == 0.5) {
                cuda::gpu_kernel(iter, SqrtFunctor<scalar_t>());
            } else if (exp == -0.5) {
                cuda::gpu_kernel(iter, RsqrtFunctor<scalar_t>());
            } else if (exp == -1.0) {
                cuda::gpu_kernel(iter, ReciprocalFunctor<scalar_t>());
            } else {
                cuda::gpu_kernel(iter, PowScalarFunctor<scalar_t>{static_cast<scalar_t>(exp)});
            }
        } else {
            cuda::gpu_kernel(iter, PowScalarFunctor<scalar_t>{exp_scalar.to<scalar_t>()});
        }
    });
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel_cuda);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel_cuda);

} // namespace raxis
