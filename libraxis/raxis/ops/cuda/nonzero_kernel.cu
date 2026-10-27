#include <cstdint>
#include <limits>

#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/offset_calculator.cuh>
#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_shape.h>


// nonzero on the GPU.
// From: aten/src/ATen/native/cuda/Nonzero.cu
//
// A stream compaction: cub's DeviceSelect::Flagged keeps the linear index i
// of every element whose value is nonzero (a prefix sum over the flags gives
// each kept index its output slot). The number of rows decides the result's
// shape, so the host waits for it, as in PyTorch: nonzero (and x[mask]) is a
// synchronization point. A second kernel turns the linear indices into one
// column per dim.

namespace raxis {
namespace {

constexpr int kThreads = 256;

struct Sizes {
    int64_t sizes[cuda::kMaxDims];
};

__global__ void unravel_kernel(int64_t* out, const int64_t* linear, int64_t total, int ndim, Sizes sizes) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= total) {
        return;
    }
    int64_t index = linear[row];
    for (int k = ndim - 1; k >= 0; k--) {
        // Column-major output: the positions along dim k are contiguous
        out[k * total + row] = index % sizes.sizes[k];
        index /= sizes.sizes[k];
    }
}

Tensor nonzero_kernel_cuda(const Tensor& self_) {
    DeviceGuard guard(self_.device());
    const Tensor self = self_.contiguous();
    const int64_t numel = self.numel();
    const int ndim = static_cast<int>(self.ndim());
    RAXIS_CHECK(numel <= std::numeric_limits<int32_t>::max(),
        "nonzero: tensors over 2^31 elements are not supported on CUDA yet");
    RAXIS_CHECK(ndim <= cuda::kMaxDims, "nonzero: tensor has too many (>", cuda::kMaxDims, ") dims");
    cudaStream_t stream = cuda::current_stream();
    const TensorOptions int64_options = self.options().dtype(dtypes::Int64);

    Tensor linear = empty({numel}, int64_options);
    Tensor count = empty({1}, int64_options);
    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        // The values themselves are the flags: nonzero converts to true
        auto run = [&](void* temp, size_t& temp_bytes) {
            return cub::DeviceSelect::Flagged(
                temp, temp_bytes, thrust::counting_iterator<int64_t>(0), self.data_ptr<scalar_t>(),
                linear.mutable_data_ptr<int64_t>(), count.mutable_data_ptr<int64_t>(),
                static_cast<int>(numel), stream);
        };
        size_t temp_bytes = 0;
        RAXIS_CUDA_CHECK(run(nullptr, temp_bytes));
        Tensor temp = empty({static_cast<int64_t>(temp_bytes)}, self.options().dtype(dtypes::Uint8));
        RAXIS_CUDA_CHECK(run(temp.mutable_data_ptr<uint8_t>(), temp_bytes));
    });
    const int64_t total = count.item().to<int64_t>();

    Tensor result = ops::transpose(empty({ndim, total}, int64_options), 0, 1);
    if (total == 0 || ndim == 0) {
        return result;
    }
    Sizes sizes;
    for (int k = 0; k < ndim; k++) {
        sizes.sizes[k] = self.shape()[k];
    }
    unravel_kernel<<<static_cast<unsigned>((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
        result.mutable_data_ptr<int64_t>(), linear.data_ptr<int64_t>(), total, ndim, sizes);
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

} // anonymous namespace

REGISTER_DISPATCH(nonzero_stub, &nonzero_kernel_cuda);

} // namespace raxis
