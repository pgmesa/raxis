#pragma once

#include <type_traits>

#include <raxis/export.h>

#include <raxis/ops/dispatch_stub.h>

#include <raxis/tensor/tensor.h>


namespace raxis {

struct TensorIterator;

// Integer power via repeated squaring.
// Handles negative exponents: only 1 and -1 survive, everything else truncates to 0.
template <
    typename T, std::enable_if_t<std::is_integral_v<T> 
        && !std::is_signed_v<T> 
            && !std::is_same_v<T, bool>, int> = 0>
inline T powi(T a, T b) {
    T result = 1;
    while (b) {
        if (b & 1) {
            result *= a;
        }
        b /= 2;
        a *= a;
    }
    return result;
}

template <
    typename T, std::enable_if_t<std::is_integral_v<T> 
        && std::is_signed_v<T> 
            && !std::is_same_v<T, bool>, int> = 0>
inline T powi(T a, T b) {
    if (b < 0) {
        if (a == 1) {
            return 1;
        } else if (a == -1) {
            auto negative = (-b) % static_cast<T>(2);
            return negative ? -1 : 1;
        } else {
            return 0;
        }
    }
    T result = 1;
    while (b) {
        if (b & 1) {
            result *= a;
        }
        b /= 2;
        a *= a;
    }
    return result;
}

using pow_tensor_tensor_fn = void (*)(TensorIterator&);
using pow_tensor_scalar_fn = void (*)(TensorIterator&, Scalar);

DECLARE_DISPATCH(pow_tensor_tensor_fn, pow_tensor_tensor_stub);
DECLARE_DISPATCH(pow_tensor_scalar_fn, pow_tensor_scalar_stub);

RAXIS_API Tensor pow(const Tensor& base, const Tensor& exp);
RAXIS_API Tensor pow(const Tensor& base, Scalar exp);
RAXIS_API Tensor pow(Scalar base, const Tensor& exp);

namespace ops {

RAXIS_API Tensor pow(const Tensor& base, const Tensor& exp);
RAXIS_API Tensor pow(const Tensor& base, Scalar exp);
RAXIS_API Tensor pow(Scalar base, const Tensor& exp);

}

} // namespace raxis