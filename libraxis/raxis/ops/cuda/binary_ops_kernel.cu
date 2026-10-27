#include <raxis/ops/binary_ops.h>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/tensor/tensor_iterator.h>


// From: aten/src/ATen/native/cuda/BinaryAddSubKernel.cu, BinaryMulKernel.cu,
// BinaryDivTrueKernel.cu
// Same semantics as the CPU kernels (ops/cpu/binary_ops_kernel.cpp).

namespace raxis {
namespace {

template <typename T>
struct AddFunctor {
    __device__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct SubFunctor {
    __device__ T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct MulFunctor {
    __device__ T operator()(T a, T b) const { return a * b; }
};

// a && b for bool, as the CPU kernel (a * b would promote to int)
template <>
struct MulFunctor<bool> {
    __device__ bool operator()(bool a, bool b) const { return a && b; }
};

template <typename T>
struct DivFunctor {
    __device__ T operator()(T a, T b) const { return a / b; }
};

void add_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, AddFunctor<scalar_t>());
    });
}

void sub_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, SubFunctor<scalar_t>());
    });
}

void mul_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, MulFunctor<scalar_t>());
    });
}

void div_kernel_cuda(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel_with_scalars(iter, DivFunctor<scalar_t>());
    });
}

} // anonymous namespace

REGISTER_DISPATCH(add_stub, &add_kernel_cuda);
REGISTER_DISPATCH(sub_stub, &sub_kernel_cuda);
REGISTER_DISPATCH(mul_stub, &mul_kernel_cuda);
REGISTER_DISPATCH(div_stub, &div_kernel_cuda);

} // namespace raxis
