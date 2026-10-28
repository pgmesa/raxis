#include <limits>
#include <tuple>

#include <raxis/ops/copy.h>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/ops/reduce_ops.h>
#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/ops/tensor_compare.h>
#include <raxis/tensor/tensor_iterator.h>


// where and clamp on the GPU, with the CPU kernels' NaN semantics
// (ops/cpu/tensor_compare_kernel.cpp).
// From: aten/src/ATen/native/cuda/TensorCompare.cu (where_kernel_impl),
// UnaryOpsKernel.cu and TensorCompare.cu (clamp kernels)
//
// std::max(a, b) is (a < b) ? b : a and std::min(a, b) is (b < a) ? b : a;
// spelled out here so a NaN `a` falls through both comparisons and stays NaN,
// as on the CPU.

namespace raxis {
namespace {

template <typename T>
__device__ inline T max_keep_nan(T a, T b) { return a < b ? b : a; }

template <typename T>
__device__ inline T min_keep_nan(T a, T b) { return b < a ? b : a; }

template <typename T>
struct WhereFunctor {
    __device__ T operator()(bool cond, T self, T other) const { return cond ? self : other; }
};

template <typename T>
struct ClampScalarFunctor {
    T min, max;
    __device__ T operator()(T a) const { return min_keep_nan(max_keep_nan(a, min), max); }
};

template <typename T>
struct ClampMinFunctor {
    T min;
    __device__ T operator()(T a) const { return max_keep_nan(a, min); }
};

template <typename T>
struct ClampMaxFunctor {
    T max;
    __device__ T operator()(T a) const { return min_keep_nan(a, max); }
};

// A NaN bound gives NaN, as on the CPU
template <typename T>
struct ClampTensorFunctor {
    __device__ T operator()(T a, T min, T max) const {
        if constexpr (std::numeric_limits<T>::has_quiet_NaN) {
            if (min != min || max != max) return std::numeric_limits<T>::quiet_NaN();
        }
        return min_keep_nan(max_keep_nan(a, min), max);
    }
};

// The iterator is (out, condition, self, other), self and other already in
// the result dtype
void where_kernel_cuda(TensorIterator& iter) {
    iter.dtype(0).dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, WhereFunctor<scalar_t>());
    });
}

void clamp_scalar_kernel_cuda(TensorIterator& iter, Scalar min, Scalar max) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, ClampScalarFunctor<scalar_t>{min.to<scalar_t>(), max.to<scalar_t>()});
    });
}

void clamp_min_scalar_kernel_cuda(TensorIterator& iter, Scalar min) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, ClampMinFunctor<scalar_t>{min.to<scalar_t>()});
    });
}

void clamp_max_scalar_kernel_cuda(TensorIterator& iter, Scalar max) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, ClampMaxFunctor<scalar_t>{max.to<scalar_t>()});
    });
}

void clamp_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, ClampTensorFunctor<scalar_t>());
    });
}

} // anonymous namespace

// max(dim) / min(dim): the index from argmax / argmin, then the value it
// points to. PyTorch reduces (value, index) pairs in one kernel with two
// outputs; reduce.cuh has one output, so this takes a reduction and a
// gather over the (much smaller) result. argmax already gives what max
// needs: NaN wins, and among equal values the first index.
template <bool is_max>
void minmax_dim_kernel_cuda(const Tensor& values, const Tensor& indices, const Tensor& self, int64_t dim) {
    // argmax takes no bool; as uint8 the order is the same
    const Tensor source = self.dtype() == dtypes::Bool ? self.to(dtypes::Uint8) : self;
    const Tensor index = is_max ? ops::argmax(source, dim, /*keepdim=*/true) : ops::argmin(source, dim, /*keepdim=*/true);
    Tensor values_out = values;
    Tensor indices_out = indices;
    copy(ops::gather(self, dim, index), values_out);
    copy(index, indices_out);
}

REGISTER_DISPATCH(max_dim_stub, &minmax_dim_kernel_cuda<true>);
REGISTER_DISPATCH(min_dim_stub, &minmax_dim_kernel_cuda<false>);
REGISTER_DISPATCH(where_stub, &where_kernel_cuda);
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel_cuda);
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel_cuda);
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel_cuda);
REGISTER_DISPATCH(clamp_stub, &clamp_kernel_cuda);

} // namespace raxis
