#include <climits>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <raxis/ops/binary_ops.h>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/tensor/tensor_iterator.h>


// Kept apart from binary_ops_kernel.cu: one file with every binary functor
// for every arch would exceed the sections an MSVC object can hold (C1128).
// The CPU kernels are in ops/cpu/binary_ops_kernel.cpp.

namespace raxis {
namespace {

// From: aten/src/ATen/native/cuda/BinaryBitwiseOpsKernels.cu, BinaryShiftOpsKernels.cu
// bool has no ^ (it promotes to int): its cases are logical ops
template <typename T>
struct BitwiseAndFunctor {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_same_v<T, bool>) {
            return a && b;
        } else {
            return a & b;
        }
    }
};

template <typename T>
struct BitwiseOrFunctor {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_same_v<T, bool>) {
            return a || b;
        } else {
            return a | b;
        }
    }
};

template <typename T>
struct BitwiseXorFunctor {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_same_v<T, bool>) {
            return a != b;
        } else {
            return a ^ b;
        }
    }
};

// A negative shift or one of the bit width or more: << gives 0, >> the sign
template <typename T>
struct LshiftFunctor {
    __device__ T operator()(T a, T b) const {
        constexpr T max_shift = sizeof(T) * CHAR_BIT;
        if (static_cast<std::make_signed_t<T>>(b) < 0 || b >= max_shift) {
            return 0;
        }
        return static_cast<T>(static_cast<std::make_unsigned_t<T>>(a) << b);
    }
};

template <typename T>
struct RshiftFunctor {
    __device__ T operator()(T a, T b) const {
        constexpr T max_shift = sizeof(T) * CHAR_BIT - std::is_signed_v<T>;
        if (static_cast<std::make_signed_t<T>>(b) < 0 || b >= max_shift) {
            if constexpr (std::is_signed_v<T>) {
                return a < 0 ? T(-1) : T(0);
            } else {
                return 0;
            }
        }
        return static_cast<T>(a >> b);
    }
};

template <template <typename> class Functor>
void bitwise_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_integral_and_bool_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, Functor<scalar_t>());
    });
}

template <template <typename> class Functor>
void shift_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, Functor<scalar_t>());
    });
}

} // anonymous namespace

REGISTER_DISPATCH(bitwise_and_stub, &bitwise_kernel_cuda<BitwiseAndFunctor>);
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_kernel_cuda<BitwiseOrFunctor>);
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_kernel_cuda<BitwiseXorFunctor>);
REGISTER_DISPATCH(lshift_stub, &shift_kernel_cuda<LshiftFunctor>);
REGISTER_DISPATCH(rshift_stub, &shift_kernel_cuda<RshiftFunctor>);

} // namespace raxis
