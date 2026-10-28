#include <algorithm>
#include <cstdint>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cat_ops.h>
#include <raxis/ops/cuda/integer_divider.cuh>
#include <raxis/tensor/tensor.h>


// cat of contiguous inputs into a contiguous result, many inputs per launch.
// From: aten/src/ATen/native/cuda/Shape.cu (CatArrayBatchedCopy, parallel_cat)
//
// Copying input by input costs a launch each; clip_grad_norm_ stacks one
// 0-d norm per parameter (~50 in nanoGPT), so the launches would dominate.
// Here the inputs' pointers go in the kernel arguments, blockIdx.y picks the
// input and the x blocks stride over its elements. In the contiguous result
// an input is a [outer, size_k * inner] block that starts at column
// offset_k * inner of rows of cat_size * inner elements, so element i of
// input k goes to (i / row_k) * out_row + offset_k * inner + i % row_k, one
// magic-number division per element. The copy moves raw bytes of the item
// size: cat never converts (all inputs share the result's dtype here).

namespace raxis {
namespace {

constexpr int kThreads = 256;
constexpr int kInputsPerLaunch = 64;

struct CatInputs {
    const void* data[kInputsPerLaunch];
    uint32_t numel[kInputsPerLaunch];
    uint32_t column[kInputsPerLaunch];          // offset_k * inner
    cuda::IntDivider row[kInputsPerLaunch];     // size_k * inner
};

static_assert(sizeof(CatInputs) < 3900, "the kernel arguments must fit in 4 KB");

template <typename T>
__global__ void __launch_bounds__(kThreads) cat_batched_kernel(T* out, CatInputs inputs, uint32_t out_row) {
    const int k = blockIdx.y;
    const T* in = static_cast<const T*>(inputs.data[k]);
    const uint32_t numel = inputs.numel[k];
    const uint32_t column = inputs.column[k];
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += gridDim.x * blockDim.x) {
        const auto [row, col] = inputs.row[k].divmod(i);
        out[row * out_row + column + col] = in[i];
    }
}

template <typename T>
void launch_cat(const Tensor& result, ArrayRef<Tensor> tensors, int64_t dim) {
    int64_t inner = 1;
    for (int64_t d = dim + 1; d < result.ndim(); d++) {
        inner *= result.shape()[d];
    }
    const auto out_row = static_cast<uint32_t>(result.shape()[dim] * inner);
    // Twice the SMs in x keeps the GPU busy with two large inputs and
    // stops tiny inputs from launching blocks with nothing to copy
    const int max_blocks_x = 2 * cuda::current_device_properties().multiProcessorCount;
    T* out = static_cast<T*>(result.mutable_data_ptr());

    CatInputs inputs;
    int n = 0;
    int64_t offset = 0;
    int64_t max_numel = 0;
    auto launch = [&] {
        const auto blocks_x = static_cast<unsigned>(
            std::clamp<int64_t>((max_numel + kThreads - 1) / kThreads, 1, max_blocks_x));
        cat_batched_kernel<T><<<dim3(blocks_x, n), kThreads, 0, cuda::current_stream()>>>(out, inputs, out_row);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
        n = 0;
        max_numel = 0;
    };
    for (const Tensor& t : tensors) {
        const int64_t size = t.ndim() == result.ndim() ? t.shape()[dim] : 0;   // a skipped 1-D empty tensor
        if (t.numel() > 0) {
            inputs.data[n] = t.data_ptr();
            inputs.numel[n] = static_cast<uint32_t>(t.numel());
            inputs.column[n] = static_cast<uint32_t>(offset * inner);
            inputs.row[n] = cuda::IntDivider(static_cast<uint32_t>(size * inner));
            max_numel = std::max(max_numel, t.numel());
            if (++n == kInputsPerLaunch) {
                launch();
            }
        }
        offset += size;
    }
    if (n > 0) {
        launch();
    }
}

// The inputs and the result are contiguous, of the result's dtype and on its
// device; the result has fewer than 2^31 elements
void cat_batched_kernel_cuda(const Tensor& result, ArrayRef<Tensor> tensors, int64_t dim) {
    DeviceGuard guard(result.device());
    switch (result.dtype().itemsize()) {
        case 1: launch_cat<uint8_t>(result, tensors, dim); break;
        case 2: launch_cat<uint16_t>(result, tensors, dim); break;
        case 4: launch_cat<uint32_t>(result, tensors, dim); break;
        case 8: launch_cat<uint64_t>(result, tensors, dim); break;
        default: RAXIS_CHECK(false, "cat: unsupported item size ", result.dtype().itemsize());
    }
}

} // anonymous namespace

REGISTER_DISPATCH(cat_batched_stub, &cat_batched_kernel_cuda);

} // namespace raxis
