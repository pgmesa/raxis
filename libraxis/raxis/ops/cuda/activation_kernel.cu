#include <type_traits>

#include <raxis/core/scalar.h>
#include <raxis/ops/activation.h>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/tensor/tensor_iterator.h>


// From: aten/src/ATen/native/cuda/Activation*.cu, GeluKernel.cu
// Same semantics as ops/cpu/activation_kernel.cpp. PyTorch computes the
// float GELU in float on the GPU (its opmath type is float), as here.

namespace raxis {
namespace {

// max(a, 0), keeping NaN (PyTorch's relu on CUDA is clamp_min(a, 0))
template <typename T>
struct ReluFunctor {
    __device__ T operator()(T a) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (a != a) return a;
        }
        return a > T(0) ? a : T(0);
    }
};

// Inputs (self, grad): the gradient passes where the forward input was
// above the threshold
template <typename T>
struct ThresholdBackwardFunctor {
    T threshold;
    __device__ T operator()(T self, T grad) const { return self <= threshold ? T(0) : grad; }
};

template <typename T>
struct GeluTanhFunctor {
    __device__ T operator()(T x) const {
        const T inner = T(kGeluBeta) * (x + T(kGeluKappa) * x * x * x);
        return T(0.5) * x * (T(1) + ::tanh(inner));
    }
};

template <typename T>
struct GeluErfFunctor {
    __device__ T operator()(T x) const { return x * T(0.5) * (T(1) + ::erf(x * T(kGeluAlpha))); }
};

// Inputs (dy, x). With tanh, gelu = left * right with left = 0.5 x and
// right = 1 + tanh(inner): gelu' = 0.5 * right + left * (1 - tanh^2) * inner'
template <typename T>
struct GeluTanhBackwardFunctor {
    __device__ T operator()(T dy, T x) const {
        const T x_sq = x * x;
        const T x_cube = x_sq * x;
        const T inner = T(kGeluBeta) * (x + T(kGeluKappa) * x_cube);
        const T tanh_inner = ::tanh(inner);
        const T left = T(0.5) * x;
        const T right = T(1) + tanh_inner;
        const T left_derivative = T(0.5) * right;
        const T tanh_derivative = T(1) - tanh_inner * tanh_inner;
        const T inner_derivative = T(kGeluBeta) * (T(1) + T(3) * T(kGeluKappa) * x_sq);
        const T right_derivative = left * tanh_derivative * inner_derivative;
        return dy * (left_derivative + right_derivative);
    }
};

// With erf: gelu'(x) = Phi(x) + x * pdf(x)
template <typename T>
struct GeluErfBackwardFunctor {
    __device__ T operator()(T dy, T x) const {
        const T cdf = T(0.5) * (T(1) + ::erf(x * T(kGeluAlpha)));
        const T pdf = T(kGeluPdfScale) * ::exp(x * x * T(-0.5));
        return dy * (cdf + x * pdf);
    }
};

void relu_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, ReluFunctor<scalar_t>());
    });
}

void threshold_backward_kernel_cuda(TensorIterator& iter, Scalar threshold) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, ThresholdBackwardFunctor<scalar_t>{threshold.to<scalar_t>()});
    });
}

void gelu_kernel_cuda(TensorIterator& iter, GeluType approximate) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if (approximate == GeluType::Tanh) {
            cuda::gpu_kernel(iter, GeluTanhFunctor<scalar_t>());
        } else {
            cuda::gpu_kernel(iter, GeluErfFunctor<scalar_t>());
        }
    });
}

void gelu_backward_kernel_cuda(TensorIterator& iter, GeluType approximate) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if (approximate == GeluType::Tanh) {
            cuda::gpu_kernel(iter, GeluTanhBackwardFunctor<scalar_t>());
        } else {
            cuda::gpu_kernel(iter, GeluErfBackwardFunctor<scalar_t>());
        }
    });
}

} // anonymous namespace

REGISTER_DISPATCH(relu_stub, &relu_kernel_cuda);
REGISTER_DISPATCH(threshold_backward_stub, &threshold_backward_kernel_cuda);
REGISTER_DISPATCH(gelu_stub, &gelu_kernel_cuda);
REGISTER_DISPATCH(gelu_backward_stub, &gelu_backward_kernel_cuda);

} // namespace raxis
