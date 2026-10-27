#include <cstdint>
#include <limits>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/integer_divider.cuh>
#include <raxis/ops/triangular_ops.h>
#include <raxis/tensor/tensor.h>


// tril / triu on the GPU: one thread per element of the contiguous result,
// which keeps self's value on the kept side of diagonal k and writes 0 on
// the other (the CPU kernel's two runs per row become a test per element).
// From: aten/src/ATen/native/cuda/TriangularOps.cu (triu_tril_kernel)

namespace raxis {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void triu_tril_kernel(
    scalar_t* result, const scalar_t* self, uint32_t numel, int64_t k, bool upper,
    cuda::IntDivider cols_div, cuda::IntDivider rows_div,
    int64_t batch_stride, int64_t row_stride, int64_t col_stride
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }
    const auto j = cols_div.divmod(index);
    const auto i = rows_div.divmod(j.div);
    const int64_t diff = static_cast<int64_t>(j.mod) - static_cast<int64_t>(i.mod);
    const bool keep = upper ? diff >= k : diff <= k;
    result[index] = keep ? self[i.div * batch_stride + i.mod * row_stride + j.mod * col_stride] : scalar_t(0);
}

void triu_tril_kernel_cuda(const Tensor& result, const Tensor& self, int64_t k, bool upper) {
    DeviceGuard guard(self.device());
    const int64_t numel = result.numel();
    RAXIS_CHECK(numel <= std::numeric_limits<int32_t>::max(),
        upper ? "triu" : "tril", ": tensors over 2^31 elements are not supported on CUDA yet");
    const int64_t ndim = self.ndim();
    const int64_t n = self.shape()[ndim - 2];
    const int64_t m = self.shape()[ndim - 1];
    const int64_t batch_stride = ndim > 2 ? self.strides()[ndim - 3] : 0;
    const auto blocks = static_cast<unsigned>((numel + kThreads - 1) / kThreads);
    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        triu_tril_kernel<scalar_t><<<blocks, kThreads, 0, cuda::current_stream()>>>(
            result.mutable_data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), static_cast<uint32_t>(numel), k, upper,
            cuda::IntDivider(static_cast<uint32_t>(m)), cuda::IntDivider(static_cast<uint32_t>(n)),
            batch_stride, self.strides()[ndim - 2], self.strides()[ndim - 1]);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // anonymous namespace

REGISTER_DISPATCH(triu_tril_stub, &triu_tril_kernel_cuda);

} // namespace raxis
