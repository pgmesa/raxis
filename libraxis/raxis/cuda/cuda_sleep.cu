#include <raxis/cuda/cuda_sleep.h>

#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_stream.h>


namespace raxis::cuda {
namespace {

// From: aten/src/ATen/cuda/Sleep.cu
__global__ void spin_kernel(int64_t cycles) {
    const int64_t start = clock64();
    while (clock64() - start < cycles) {
    }
}

} // anonymous namespace

void sleep(int64_t cycles) {
    spin_kernel<<<1, 1, 0, current_stream()>>>(cycles);
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace raxis::cuda
