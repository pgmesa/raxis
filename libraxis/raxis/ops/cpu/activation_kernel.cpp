#include <raxis/ops/activation.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/util/parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

void relu_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                // a != a keeps NaN, as PyTorch's clamp_min(self, 0) does
                if constexpr (std::is_floating_point_v<scalar_t>) {
                    if (a != a) return a;
                }
                return std::max(a, static_cast<scalar_t>(0));
            },
            [](Vectorized<scalar_t> a) {
                return vec::maximum(a, Vectorized<scalar_t>(static_cast<scalar_t>(0)));
            }
        );
    });
}

// From: aten/src/ATen/native/cpu/Activation.cpp (threshold_kernel)
// Inputs are (self, grad): the output keeps grad where self > threshold.
// The mask select is branchless in the vectorized path (blendv).
void threshold_backward_kernel(TensorIterator& iter, Scalar threshold_scalar) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using Vec = Vectorized<scalar_t>;
        scalar_t threshold = threshold_scalar.to<scalar_t>();
        Vec threshold_vec(threshold);
        Vec zero_vec(static_cast<scalar_t>(0));
        cpu_kernel_vec(iter,
            [threshold](scalar_t self, scalar_t grad) -> scalar_t {
                return self <= threshold ? static_cast<scalar_t>(0) : grad;
            },
            [threshold_vec, zero_vec](Vec self, Vec grad) {
                return Vec::blendv(grad, zero_vec, self <= threshold_vec);
            }
        );
    });
}

// From: aten/src/ATen/native/cpu/Activation.cpp (GeluKernelImpl) and
// cpu/Gelu.h. Each element costs an erf or a tanh, so the op is compute
// bound and splits between threads from a lower size than the default grain
void gelu_kernel(TensorIterator& iter, GeluType approximate) {
    constexpr int64_t GELU_MIN_ELEMENTS_FOR_MULTI_THREADING = 16384;
    int64_t grain_size = detail::GRAIN_SIZE;
    if (iter.numel() > GELU_MIN_ELEMENTS_FOR_MULTI_THREADING) {
        grain_size = iter.numel() / get_num_threads();
    }
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using Vec = Vectorized<scalar_t>;
        const Vec point_five(scalar_t(0.5));
        const Vec one(scalar_t(1));
        if (approximate == GeluType::Tanh) {
            // 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
            const Vec beta(static_cast<scalar_t>(kGeluBeta));
            const Vec kappa(static_cast<scalar_t>(kGeluKappa));
            cpu_kernel_vec(iter,
                [](scalar_t x) -> scalar_t {
                    const scalar_t x_cube = x * x * x;
                    const scalar_t inner = scalar_t(kGeluBeta) * (x + scalar_t(kGeluKappa) * x_cube);
                    return scalar_t(0.5) * x * (scalar_t(1) + std::tanh(inner));
                },
                [=](Vec x) {
                    const Vec x_cube = x * x * x;
                    const Vec inner = beta * (x + kappa * x_cube);
                    return point_five * x * (one + inner.tanh());
                },
                grain_size);
        } else {
            // x * 0.5 * (1 + erf(x / sqrt(2)))
            const Vec alpha(static_cast<scalar_t>(kGeluAlpha));
            cpu_kernel_vec(iter,
                [](scalar_t x) -> scalar_t {
                    return x * scalar_t(0.5) * (scalar_t(1) + std::erf(x * scalar_t(kGeluAlpha)));
                },
                [=](Vec x) {
                    return x * point_five * (one + (x * alpha).erf());
                },
                grain_size);
        }
    });
}

// From: aten/src/ATen/native/cpu/Activation.cpp (GeluBackwardKernelImpl)
// Inputs are (dy, x). With erf: gelu'(x) = Phi(x) + x * pdf(x). With tanh,
// writing gelu = left * right with left = 0.5 x and right = 1 + tanh(inner):
// gelu' = 0.5 * right + left * (1 - tanh^2) * inner'
void gelu_backward_kernel(TensorIterator& iter, GeluType approximate) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using Vec = Vectorized<scalar_t>;
        const Vec point_five(scalar_t(0.5));
        const Vec one(scalar_t(1));
        if (approximate == GeluType::Tanh) {
            const Vec beta(static_cast<scalar_t>(kGeluBeta));
            const Vec kappa(static_cast<scalar_t>(kGeluKappa));
            const Vec three(scalar_t(3));
            cpu_kernel_vec(iter,
                [](scalar_t dy, scalar_t x) -> scalar_t {
                    const scalar_t beta = kGeluBeta;
                    const scalar_t kappa = kGeluKappa;
                    const scalar_t x_sq = x * x;
                    const scalar_t x_cube = x_sq * x;
                    const scalar_t tanh_inner = std::tanh(beta * (x + kappa * x_cube));
                    const scalar_t left = scalar_t(0.5) * x;
                    const scalar_t right = scalar_t(1) + tanh_inner;
                    const scalar_t left_derivative = scalar_t(0.5) * right;
                    const scalar_t tanh_derivative = scalar_t(1) - tanh_inner * tanh_inner;
                    const scalar_t inner_derivative = beta * (scalar_t(1) + scalar_t(3) * kappa * x_sq);
                    const scalar_t right_derivative = left * tanh_derivative * inner_derivative;
                    return dy * (left_derivative + right_derivative);
                },
                [=](Vec dy, Vec x) {
                    const Vec x_sq = x * x;
                    const Vec x_cube = x * x * x;
                    const Vec tanh_inner = (beta * (x + kappa * x_cube)).tanh();
                    const Vec left = point_five * x;
                    const Vec right = one + tanh_inner;
                    const Vec left_derivative = point_five * right;
                    const Vec tanh_derivative = one - tanh_inner * tanh_inner;
                    const Vec inner_derivative = beta * (one + three * kappa * x_sq);
                    const Vec right_derivative = left * tanh_derivative * inner_derivative;
                    return dy * (left_derivative + right_derivative);
                });
        } else {
            const Vec alpha(static_cast<scalar_t>(kGeluAlpha));
            const Vec pdf_scale(static_cast<scalar_t>(kGeluPdfScale));
            const Vec minus_point_five(scalar_t(-0.5));
            cpu_kernel_vec(iter,
                [](scalar_t dy, scalar_t x) -> scalar_t {
                    const scalar_t cdf = scalar_t(0.5) * (scalar_t(1) + std::erf(x * scalar_t(kGeluAlpha)));
                    const scalar_t pdf = scalar_t(kGeluPdfScale) * std::exp(x * x * scalar_t(-0.5));
                    return dy * (cdf + x * pdf);
                },
                [=](Vec dy, Vec x) {
                    const Vec cdf = point_five * (one + (x * alpha).erf());
                    const Vec pdf = pdf_scale * (x * x * minus_point_five).exp();
                    return dy * (cdf + x * pdf);
                });
        }
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(relu_stub, &relu_kernel);
REGISTER_DISPATCH(threshold_backward_stub, &threshold_backward_kernel);
ALSO_REGISTER_AVX512_DISPATCH(gelu_stub, &gelu_kernel);
ALSO_REGISTER_AVX512_DISPATCH(gelu_backward_stub, &gelu_backward_kernel);

} // namespace raxis
