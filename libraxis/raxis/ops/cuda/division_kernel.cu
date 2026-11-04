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

// From: aten/src/ATen/native/cuda/BinaryDivTruncKernel.cu, BinaryDivFloorKernel.cu,
// BinaryRemainderKernel.cu, c10/util/generic_math.h
// The steps of the CPU kernels. A kernel cannot raise, so an integer
// division by zero gives what the hardware does, as in torch (all bits set
// for / and %, which torch keeps; uint8 % 0 is pinned to 255 there too).
// torch divides a float by a CPU scalar as a * (1 / b) in the trunc and
// floor modes; raxis divides (as the CPU), so when a / b is within a
// rounding of an integer the two can land on different sides of it.

template <typename T>
__device__ bool signs_differ(T a, T b) {
    if constexpr (std::is_unsigned_v<T>) {
        return false;
    } else {
        return (a < 0) != (b < 0);
    }
}

template <typename T>
struct DivTruncFunctor {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return ::trunc(a / b);
        } else {
            return a / b;
        }
    }
};

template <typename T>
struct DivFloorFunctor {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            if (b == 0) {
                return a / b;
            }
            const T mod = ::fmod(a, b);
            T div = (a - mod) / b;
            if (mod != 0 && (b < 0) != (mod < 0)) {
                div -= T(1);
            }
            if (div == 0) {
                return ::copysign(T(0), a / b);
            }
            T floordiv = ::floor(div);
            if (div - floordiv > T(0.5)) {
                floordiv += T(1);
            }
            return floordiv;
        } else {
            if (b == 0) {
                return 0;
            }
            if constexpr (std::is_signed_v<T>) {
                if (a == std::numeric_limits<T>::min() && b == T(-1)) {
                    return a;
                }
            }
            const T quot = a / b;
            return signs_differ(a, b) && a % b != 0 ? quot - 1 : quot;
        }
    }
};

template <typename T>
struct RemainderFunctor {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            T mod = ::fmod(a, b);
            if (mod != 0 && (b < 0) != (mod < 0)) {
                mod += b;
            }
            return mod;
        } else {
            if constexpr (std::is_same_v<T, uint8_t>) {
                if (b == 0) {
                    return std::numeric_limits<uint8_t>::max();
                }
            }
            T r = a % b;
            if (r != 0 && signs_differ(r, b)) {
                r += b;
            }
            return r;
        }
    }
};

template <typename T>
struct FmodFunctor {
    __device__ T operator()(T a, T b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return ::fmod(a, b);
        } else {
            return a % b;
        }
    }
};

// Integers and floats; the front-end has refused bool
template <template <typename> class Functor>
void division_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, Functor<scalar_t>());
    });
}

void div_floor_kernel_cuda(TensorIterator& iter) {
    if (iter.common_dtype() == dtypes::Uint8) {
        // Both signs are positive: floor and truncation are the same
        division_kernel_cuda<DivTruncFunctor>(iter);
        return;
    }
    division_kernel_cuda<DivFloorFunctor>(iter);
}

} // anonymous namespace

REGISTER_DISPATCH(div_trunc_stub, &division_kernel_cuda<DivTruncFunctor>);
REGISTER_DISPATCH(div_floor_stub, &div_floor_kernel_cuda);
REGISTER_DISPATCH(remainder_stub, &division_kernel_cuda<RemainderFunctor>);
REGISTER_DISPATCH(fmod_stub, &division_kernel_cuda<FmodFunctor>);

} // namespace raxis
