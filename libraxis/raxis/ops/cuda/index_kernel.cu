#include <cstdint>
#include <limits>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/integer_divider.cuh>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>


// index_select and masked_fill on the GPU.
// From: aten/src/ATen/native/cuda/Indexing.cu (index_select_out_cuda_impl),
// IndexKernel.cu (masked_fill_kernel)

namespace raxis {
namespace {

constexpr int kThreads = 256;

// self viewed as [outer, dim_size, inner] (contiguous), result as
// [outer, num_indices, inner]: one thread per result element. PyTorch picks
// between a kernel per index (large slices) and one per element; per
// element is the general one, and consecutive threads still read
// consecutive elements of a slice
template <typename scalar_t>
__global__ void index_select_kernel(
    scalar_t* result, const scalar_t* self, const int64_t* index, uint32_t n,
    cuda::IntDivider inner_div, cuda::IntDivider num_indices_div, int64_t dim_size
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const auto k = inner_div.divmod(i);
    const auto j = num_indices_div.divmod(k.div);
    const int64_t src = index[j.mod];
    RAXIS_CUDA_KERNEL_ASSERT(src >= 0 && src < dim_size && "index out of range in self");
    result[i] = self[(static_cast<int64_t>(j.div) * dim_size + src) * inner_div.divisor + k.mod];
}

void index_select_kernel_cuda(const Tensor& result, const Tensor& self_, int64_t dim, const Tensor& index) {
    DeviceGuard guard(self_.device());
    const int64_t n = result.numel();
    if (n == 0) {
        return;
    }
    const Tensor self = self_.contiguous();
    RAXIS_CHECK(n <= std::numeric_limits<int32_t>::max() && self.numel() <= std::numeric_limits<int32_t>::max(),
        "index_select: tensors over 2^31 elements are not supported on CUDA yet");
    int64_t inner = 1;
    for (int64_t d = dim + 1; d < self.ndim(); ++d) {
        inner *= self.shape()[d];
    }
    const int64_t dim_size = self.ndim() == 0 ? 1 : self.shape()[dim];
    const auto inner_div = cuda::IntDivider(static_cast<uint32_t>(inner));
    const auto num_indices_div = cuda::IntDivider(static_cast<uint32_t>(index.numel()));
    const auto blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        index_select_kernel<scalar_t><<<blocks, kThreads, 0, cuda::current_stream()>>>(
            result.mutable_data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
            static_cast<uint32_t>(n), inner_div, num_indices_div, dim_size);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template <typename T>
struct MaskedFillFunctor {
    T value;
    __device__ T operator()(T self, bool mask) const { return mask ? value : self; }
};

// The iterator is (self, self, mask): mask ? value : self for every element
void masked_fill_kernel_cuda(TensorIterator& iter, Scalar value) {
    iter.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, MaskedFillFunctor<scalar_t>{value.to<scalar_t>()});
    });
}

} // anonymous namespace

REGISTER_DISPATCH(index_select_stub, &index_select_kernel_cuda);
REGISTER_DISPATCH(masked_fill_stub, &masked_fill_kernel_cuda);

} // namespace raxis
