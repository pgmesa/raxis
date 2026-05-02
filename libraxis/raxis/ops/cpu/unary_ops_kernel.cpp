
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/unary_ops.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>

#include <cmath>
#include <limits>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;

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

void exp_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                return std::exp(a);
            },
            [](Vectorized<scalar_t> a) {
                return a.exp();
            }
        );
    });
}

void log_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                return std::log(a);
            },
            [](Vectorized<scalar_t> a) {
                return a.log();
            }
        );
    });
}

void sqrt_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a) -> scalar_t {
                return std::sqrt(a);
            },
            [](Vectorized<scalar_t> a) {
                return a.sqrt();
            }
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

REGISTER_DISPATCH(neg_stub, &neg_kernel);
REGISTER_DISPATCH(abs_stub, &abs_kernel);
REGISTER_DISPATCH(ceil_stub, &ceil_kernel);
REGISTER_DISPATCH(floor_stub, &floor_kernel);
REGISTER_DISPATCH(round_stub, &round_kernel);
REGISTER_DISPATCH(exp_stub, &exp_kernel);
REGISTER_DISPATCH(log_stub, &log_kernel);
REGISTER_DISPATCH(sqrt_stub, &sqrt_kernel);
REGISTER_DISPATCH(rsqrt_stub, &rsqrt_kernel);
REGISTER_DISPATCH(reciprocal_stub, &reciprocal_kernel);
REGISTER_DISPATCH(sigmoid_stub, &sigmoid_kernel);
REGISTER_DISPATCH(logical_not_stub, &logical_not_kernel);

} // namespace raxis