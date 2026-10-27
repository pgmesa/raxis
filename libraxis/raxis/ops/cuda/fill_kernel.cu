#include <raxis/core/scalar.h>
#include <raxis/ops/fill.h>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/tensor/tensor_iterator.h>


// From: aten/src/ATen/native/cuda/FillKernel.cu

namespace raxis {
namespace {

// A nullary functor: the output is the only operand
template <typename T>
struct FillFunctor {
    explicit FillFunctor(T value) : value(value) {}
    __device__ T operator()() const { return value; }
    T value;
};

void fill_kernel_cuda(TensorIterator& iter, Scalar value) {
    iter.dtype(0).dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
    });
}

} // anonymous namespace

REGISTER_DISPATCH(fill_stub, &fill_kernel_cuda);

} // namespace raxis
