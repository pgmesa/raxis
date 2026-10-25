
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/pow.h>
#include <raxis/ops/unary_ops.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>

#include <cmath>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

void pow_tensor_tensor_kernel(TensorIterator& iter) {
    auto dtype = iter.common_dtype();
    if (dtype.is_floating()) {
        dtype.dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            cpu_kernel_vec(iter,
                [](scalar_t base, scalar_t exp) -> scalar_t {
                    return std::pow(base, exp);
                },
                [](Vectorized<scalar_t> base, Vectorized<scalar_t> exp) {
                    return base.pow(exp);
                }
            );
        });
    } else {
        dtype.dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            cpu_kernel(iter,
                [](scalar_t base, scalar_t exp) -> scalar_t {
                    return powi(base, exp);
                }
            );
        });
    }
}

template <typename scalar_t>
void pow_tensor_scalar_optimized_kernel(TensorIterator& iter, double exp) {
    using Vec = Vectorized<scalar_t>;

    if (exp == 2.0) {
        cpu_kernel_vec(iter,
            [](scalar_t base) -> scalar_t {
                return base * base;
            },
            [](Vec base) -> Vec { return base * base; }
        );
    } else if (exp == 3.0) {
        cpu_kernel_vec(iter,
            [](scalar_t base) -> scalar_t {
                return base * base * base;
            },
            [](Vec base) -> Vec { return base * base * base; }
        );
    } else if (exp == -2.0) {
        cpu_kernel_vec(iter,
            [](scalar_t base) -> scalar_t {
                return static_cast<scalar_t>(1.0) / (base * base);
            },
            [](Vec base) -> Vec { return (base * base).reciprocal(); }
        );
    } else {
        cpu_kernel_vec(iter,
            [=](scalar_t base) -> scalar_t {
                return std::pow(base, static_cast<scalar_t>(exp));
            },
            [=](Vec base) -> Vec {
                return base.pow(static_cast<scalar_t>(exp));
            }
        );
    }
}

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar) {
    auto dtype = iter.common_dtype();

    if (dtype.is_floating()) {
        double exp = exp_scalar.to<double>();

        if (exp == 0.5) {
            return sqrt_kernel(iter);
        } else if (exp == -0.5) {
            return rsqrt_kernel(iter);
        } else if (exp == -1.0) {
            return reciprocal_kernel(iter);
        }

        dtype.dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            pow_tensor_scalar_optimized_kernel<scalar_t>(iter, exp);
        });
    } else {
        dtype.dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            scalar_t exp = exp_scalar.to<scalar_t>();
            cpu_kernel(iter,
                [=](scalar_t base) -> scalar_t {
                    return powi(base, exp);
                }
            );
        });
    }
}

} // namespace CPU_CAPABILITY

ALSO_REGISTER_AVX512_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
ALSO_REGISTER_AVX512_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);

} // namespace raxis