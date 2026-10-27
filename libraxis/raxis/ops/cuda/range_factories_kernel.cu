#include <cstdint>
#include <type_traits>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/range_factories.h>
#include <raxis/tensor/tensor.h>


// arange on the GPU: element i is start + step * i, computed per element so
// no rounding error builds up along the range. In float for float, as
// PyTorch's CUDA kernel (its CPU one, and raxis's, accumulate in double), so
// the values match torch's on each device.
// From: aten/src/ATen/native/cuda/RangeFactories.cu (arange_cuda_out)

namespace raxis {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t, typename acc_t>
__global__ void arange_kernel(scalar_t* out, int64_t n, acc_t start, acc_t step) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = static_cast<scalar_t>(start + step * static_cast<acc_t>(i));
    }
}

void arange_kernel_cuda(const Tensor& result, Scalar start, Scalar step) {
    DeviceGuard guard(result.device());
    RAXIS_INTERNAL_ASSERT(result.is_contiguous());
    const int64_t n = result.numel();
    const auto blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    result.dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using acc_t = std::conditional_t<std::is_floating_point_v<scalar_t>, scalar_t, int64_t>;
        arange_kernel<scalar_t, acc_t><<<blocks, kThreads, 0, cuda::current_stream()>>>(
            result.mutable_data_ptr<scalar_t>(), n, start.to<acc_t>(), step.to<acc_t>());
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // anonymous namespace

REGISTER_DISPATCH(arange_stub, &arange_kernel_cuda);

} // namespace raxis
