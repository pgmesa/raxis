#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/vml.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/unary_ops.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

// ---------------------------------------------------------------------------
// VML kernel infrastructure
//
// For transcendental ops (exp, log, sin, ...) we bypass cpu_kernel_vec and
// use iter.for_each with a lambda that calls into VML. When MKL is available,
// VML dispatches to hand-tuned Intel routines (vsExp, vsLn, vsSin, ...) that
// are significantly faster than Sleef.
//
// For non-contiguous strides, data is staged through a stack buffer so the
// vectorized compute always operates on contiguous memory.
// ---------------------------------------------------------------------------

#define RAXIS_VML_KERNEL_BODY(op)                                               \
    [&](char** data_, const int64_t* strides, int64_t n) {                      \
        scalar_t* out_data = reinterpret_cast<scalar_t*>(data_[0]);             \
        scalar_t* in_data = reinterpret_cast<scalar_t*>(data_[1]);              \
        int64_t out_stride = strides[0] / static_cast<int64_t>(sizeof(scalar_t)); \
        int64_t in_stride = strides[1] / static_cast<int64_t>(sizeof(scalar_t)); \
        if (out_stride == 1 && in_stride == 1) {                                \
            vml::v##op(out_data, in_data, n);                                   \
            return;                                                             \
        }                                                                       \
        static constexpr int64_t WIDTH = (8 * 1024) / sizeof(scalar_t);         \
        for (int64_t i = 0; i < n; i += WIDTH) {                                \
            scalar_t buffer[WIDTH];                                             \
            const int64_t width = std::min(WIDTH, n - i);                       \
            scalar_t* in_buffer = in_stride == 1 ? &in_data[i] : &buffer[0];   \
            scalar_t* out_buffer = out_stride == 1 ? &out_data[i] : &buffer[0]; \
            if (in_stride != 1) {                                               \
                for (int64_t j = 0; j < width; j++) {                           \
                    in_buffer[j] = in_data[in_stride * (i + j)];                \
                }                                                               \
            }                                                                   \
            vml::v##op(out_buffer, in_buffer, width);                           \
            if (out_stride != 1) {                                              \
                for (int64_t j = 0; j < width; j++) {                           \
                    out_data[out_stride * (i + j)] = out_buffer[j];             \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }

#define RAXIS_IMPLEMENT_VML_KERNEL(name, op)                                    \
    void name##_kernel(TensorIterator& iter) {                                  \
        RAXIS_INTERNAL_ASSERT(iter.num_operands() == 2);                        \
        iter.common_dtype().dispatch_floating_types(                             \
            [&]<typename scalar_t>(std::type_identity<scalar_t>) {              \
                constexpr int64_t grain_size = 2048;                            \
                iter.for_each(RAXIS_VML_KERNEL_BODY(op), grain_size);           \
            }                                                                   \
        );                                                                      \
        iter.cast_outputs();                                                    \
    }

RAXIS_IMPLEMENT_VML_KERNEL(exp,   exp)
RAXIS_IMPLEMENT_VML_KERNEL(expm1, expm1)
RAXIS_IMPLEMENT_VML_KERNEL(log,   log)
RAXIS_IMPLEMENT_VML_KERNEL(log2,  log2)
RAXIS_IMPLEMENT_VML_KERNEL(log10, log10)
RAXIS_IMPLEMENT_VML_KERNEL(log1p, log1p)
RAXIS_IMPLEMENT_VML_KERNEL(sqrt,  sqrt)
RAXIS_IMPLEMENT_VML_KERNEL(sin,   sin)
RAXIS_IMPLEMENT_VML_KERNEL(cos,   cos)
RAXIS_IMPLEMENT_VML_KERNEL(tan,   tan)
RAXIS_IMPLEMENT_VML_KERNEL(asin,  asin)
RAXIS_IMPLEMENT_VML_KERNEL(acos,  acos)
RAXIS_IMPLEMENT_VML_KERNEL(atan,  atan)
RAXIS_IMPLEMENT_VML_KERNEL(sinh,  sinh)
RAXIS_IMPLEMENT_VML_KERNEL(cosh,  cosh)
RAXIS_IMPLEMENT_VML_KERNEL(tanh,  tanh)
RAXIS_IMPLEMENT_VML_KERNEL(erf,   erf)
RAXIS_IMPLEMENT_VML_KERNEL(erfc,  erfc)

#undef RAXIS_IMPLEMENT_VML_KERNEL
#undef RAXIS_VML_KERNEL_BODY

void neg_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                return -a;
            },
            [](Vectorized<scalar_t> a) {
                return a.neg();
            }
        );
    });
}

void abs_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                return std::abs(a);
            },
            [](Vectorized<scalar_t> a) {
                return a.abs();
            }
        );
    });
}

// From: aten/src/ATen/native/cpu/UnaryOpsKernel.cpp
// (0 < a) - (a < 0) is branchless and gives 0 for 0, -0 and NaN. The vector
// version turns each comparison mask into 0 or 1 with a blend.
void sign_kernel(TensorIterator& iter) {
    if (iter.common_dtype().is_bool()) {
        cpu_kernel(iter, [](bool a) -> bool { return a; });
        return;
    }
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const Vectorized<scalar_t> zero_vec(static_cast<scalar_t>(0));
        const Vectorized<scalar_t> one_vec(static_cast<scalar_t>(1));
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                if constexpr (std::is_unsigned_v<scalar_t>) {
                    return static_cast<scalar_t>(a > 0);
                } else {
                    return static_cast<scalar_t>((scalar_t(0) < a) - (a < scalar_t(0)));
                }
            },
            [=](Vectorized<scalar_t> a) {
                auto left = Vectorized<scalar_t>::blendv(zero_vec, one_vec, zero_vec < a);
                auto right = Vectorized<scalar_t>::blendv(zero_vec, one_vec, a < zero_vec);
                return left - right;
            }
        );
    });
}

void ceil_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t { return std::ceil(a); },
            [](Vectorized<scalar_t> a) { return a.ceil(); }
        );
    });
}

void floor_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t { return std::floor(a); },
            [](Vectorized<scalar_t> a) { return a.floor(); }
        );
    });
}

void round_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t { return std::nearbyint(a); },
            [](Vectorized<scalar_t> a) { return a.round(); }
        );
    });
}

void rsqrt_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                return static_cast<scalar_t>(1.0) / std::sqrt(a);
            },
            [](Vectorized<scalar_t> a) {
                return a.rsqrt();
            }
        );
    });
}

void reciprocal_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                return static_cast<scalar_t>(1.0) / a;
            },
            [](Vectorized<scalar_t> a) {
                return a.reciprocal();
            }
        );
    });
}

void sigmoid_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                return static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp(-a));
            },
            [](Vectorized<scalar_t> a) {
                auto one = Vectorized<scalar_t>(static_cast<scalar_t>(1));
                return (one + a.neg().exp()).reciprocal();
            }
        );
    });
}

void logical_not_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> bool {
            return !static_cast<bool>(a);
        });
    });
}

} // namespace CPU_CAPABILITY

// AVX512 only where PyTorch registers it (UnaryOpsKernel.cpp): the
// transcendentals; neg, abs, rounding, sqrt and sin/cos/tan stay on AVX2
REGISTER_DISPATCH(neg_stub, &neg_kernel);
REGISTER_DISPATCH(abs_stub, &abs_kernel);
ALSO_REGISTER_AVX512_DISPATCH(sign_stub, &sign_kernel);
REGISTER_DISPATCH(ceil_stub, &ceil_kernel);
REGISTER_DISPATCH(floor_stub, &floor_kernel);
REGISTER_DISPATCH(round_stub, &round_kernel);
ALSO_REGISTER_AVX512_DISPATCH(exp_stub, &exp_kernel);
ALSO_REGISTER_AVX512_DISPATCH(expm1_stub, &expm1_kernel);
ALSO_REGISTER_AVX512_DISPATCH(log_stub, &log_kernel);
ALSO_REGISTER_AVX512_DISPATCH(log2_stub, &log2_kernel);
ALSO_REGISTER_AVX512_DISPATCH(log10_stub, &log10_kernel);
ALSO_REGISTER_AVX512_DISPATCH(log1p_stub, &log1p_kernel);
REGISTER_DISPATCH(sqrt_stub, &sqrt_kernel);
REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel);
ALSO_REGISTER_AVX512_DISPATCH(reciprocal_stub, &reciprocal_kernel);
ALSO_REGISTER_AVX512_DISPATCH(sigmoid_stub, &sigmoid_kernel);
REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel);
REGISTER_DISPATCH(sin_stub, &sin_kernel);
REGISTER_DISPATCH(cos_stub, &cos_kernel);
REGISTER_DISPATCH(tan_stub, &tan_kernel);
ALSO_REGISTER_AVX512_DISPATCH(asin_stub, &asin_kernel);
ALSO_REGISTER_AVX512_DISPATCH(acos_stub, &acos_kernel);
ALSO_REGISTER_AVX512_DISPATCH(atan_stub, &atan_kernel);
ALSO_REGISTER_AVX512_DISPATCH(sinh_stub, &sinh_kernel);
ALSO_REGISTER_AVX512_DISPATCH(cosh_stub, &cosh_kernel);
ALSO_REGISTER_AVX512_DISPATCH(tanh_stub, &tanh_kernel);
ALSO_REGISTER_AVX512_DISPATCH(erf_stub, &erf_kernel);
ALSO_REGISTER_AVX512_DISPATCH(erfc_stub, &erfc_kernel);

} // namespace raxis