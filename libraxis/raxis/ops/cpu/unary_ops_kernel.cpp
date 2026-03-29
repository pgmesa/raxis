
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/unary_ops.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>

#include <cmath>
#include <limits>
#include <type_traits>


namespace raxis {

namespace {

void neg_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> scalar_t {
            return -a;
        });
    });
}

void abs_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> scalar_t {
            return std::abs(a);
        });
    });
}

void ceil_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> scalar_t {
            return std::ceil(a);
        });
    });
}

void floor_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> scalar_t {
            return std::floor(a);
        });
    });
}

void round_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> scalar_t {
            return std::nearbyint(a);
        });
    });
}

void exp_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> scalar_t {
            return std::exp(a);
        });
    });
}

void log_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> scalar_t {
            return std::log(a);
        });
    });
}

void sqrt_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> scalar_t {
            return std::sqrt(a);
        });
    });
}

void sigmoid_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> scalar_t {
            return static_cast<scalar_t>(1) / (static_cast<scalar_t>(1) + std::exp(-a));
        });
    });
}

void logical_not_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a) -> bool {
            return !static_cast<bool>(a);
        });
    });
}

} // namespace

REGISTER_CPU_DISPATCH(neg_stub, &neg_kernel);
REGISTER_CPU_DISPATCH(abs_stub, &abs_kernel);
REGISTER_CPU_DISPATCH(ceil_stub, &ceil_kernel);
REGISTER_CPU_DISPATCH(floor_stub, &floor_kernel);
REGISTER_CPU_DISPATCH(round_stub, &round_kernel);
REGISTER_CPU_DISPATCH(exp_stub, &exp_kernel);
REGISTER_CPU_DISPATCH(log_stub, &log_kernel);
REGISTER_CPU_DISPATCH(sqrt_stub, &sqrt_kernel);
REGISTER_CPU_DISPATCH(sigmoid_stub, &sigmoid_kernel);
REGISTER_CPU_DISPATCH(logical_not_stub, &logical_not_kernel);

} // namespace raxis