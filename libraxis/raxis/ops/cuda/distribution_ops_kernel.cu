#include <cstdint>
#include <type_traits>

#include <raxis/core/device_guard.h>
#include <raxis/core/philox.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/distribution_ops.h>
#include <raxis/tensor/tensor.h>


// uniform_, normal_ and randint_ on the GPU with the CPU's counter layout.
// From: aten/src/ATen/native/cuda/DistributionTemplates.h for the
// transformations, ops/cpu/distribution_ops_kernel.cpp for the layout.
//
// The CPU draws batches of 16 consecutive Philox blocks and lays out their
// 64 words lane-major: word w of block (batch * 16 + lane) is word
// w * 16 + lane of the batch. Here thread t computes block offset + t (lane
// t % 16 of batch t / 16) and writes the elements that use its words, as the
// dropout kernel: uniform_ and randint_ are bitwise the CPU's for a seed.
// normal_ uses the same uniforms, but CUDA's log / sin / cos are not Sleef's,
// so it matches the CPU in distribution only.
//
// Element j of a batch takes word j (float, 64 per batch) or words j and
// j + 32 (double and randint, 32 per batch). For thread `lane` that is word w
// for element w * 16 + lane, or words h and h + 2 for element h * 16 + lane.

namespace raxis {
namespace {

constexpr int kThreads = 256;

// No FMA contraction, so value * range + from rounds as on the CPU
__device__ inline float mul_add_rn(float a, float b, float c) { return __fadd_rn(__fmul_rn(a, b), c); }
__device__ inline double mul_add_rn(double a, double b, double c) { return __dadd_rn(__dmul_rn(a, b), c); }

// [0, 1) from 24 random bits (float) or 53 (double), as unit_uniform on the CPU
__device__ inline float unit_uniform(uint32_t lo, uint32_t /*hi*/, std::type_identity<float>) {
    return static_cast<float>(static_cast<int32_t>(lo & ((1u << 24) - 1))) * 0x1p-24f;
}

__device__ inline double unit_uniform(uint32_t lo, uint32_t hi, std::type_identity<double>) {
    const uint64_t bits = (static_cast<uint64_t>(hi) << 32) | lo;
    return static_cast<double>(static_cast<int64_t>(bits & ((uint64_t{1} << 53) - 1))) * 0x1p-53;
}

// Elements per thread: 4 for float (one word each), 2 for double (two words)
template <typename scalar_t>
constexpr int kElementsPerThread = sizeof(scalar_t) == 8 ? 2 : 4;

// The k-th uniform of the thread and the index of its element
template <typename scalar_t>
__device__ inline scalar_t thread_uniform(const PhiloxBlock& r, int k) {
    if constexpr (sizeof(scalar_t) == 8) {
        return unit_uniform(r[k], r[k + 2], std::type_identity<scalar_t>{});
    } else {
        return unit_uniform(r[k], 0u, std::type_identity<scalar_t>{});
    }
}

__device__ inline int64_t element_index(int64_t t, int k, int per_batch) {
    return (t / kPhiloxLanes) * per_batch + k * kPhiloxLanes + t % kPhiloxLanes;
}

template <typename scalar_t>
__global__ void uniform_kernel(scalar_t* out, int64_t n, scalar_t from, scalar_t to, uint64_t seed, uint64_t offset) {
    constexpr int per_thread = kElementsPerThread<scalar_t>;
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element_index(t, 0, per_thread * kPhiloxLanes) >= n) {
        return;
    }
    const PhiloxBlock r = philox4x32_10(offset + static_cast<uint64_t>(t), seed);
    const scalar_t range = to - from;
    #pragma unroll
    for (int k = 0; k < per_thread; ++k) {
        const int64_t i = element_index(t, k, per_thread * kPhiloxLanes);
        if (i < n) {
            const scalar_t value = mul_add_rn(thread_uniform<scalar_t>(r, k), range, from);
            // x * range + from can round up to `to`
            out[i] = value == to ? from : value;
        }
    }
}

// Box-Muller pairs element j with j + 8 of each group of 16 (NormalFill16 on
// the CPU): lanes l and l + 8 of a batch, so the partner's uniform is one
// warp shuffle away. Both lanes compute the same radius and angle; the low
// lane keeps the cosine, the high one the sine. Every thread takes part in
// the shuffle, also past the end of the tensor.
template <typename scalar_t>
__global__ void normal_kernel(scalar_t* out, int64_t n, scalar_t mean, scalar_t stddev, uint64_t seed, uint64_t offset) {
    constexpr int per_thread = kElementsPerThread<scalar_t>;
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const PhiloxBlock r = philox4x32_10(offset + static_cast<uint64_t>(t), seed);
    const bool low_lane = t % kPhiloxLanes < kPhiloxLanes / 2;
    #pragma unroll
    for (int k = 0; k < per_thread; ++k) {
        const scalar_t own = thread_uniform<scalar_t>(r, k);
        const scalar_t partner = __shfl_xor_sync(0xffffffff, own, kPhiloxLanes / 2);
        const scalar_t u1 = low_lane ? own : partner;
        const scalar_t u2 = low_lane ? partner : own;
        // 1 - u1 is in (0, 1], so the log is finite
        const scalar_t radius = ::sqrt(scalar_t(-2) * ::log(scalar_t(1) - u1));
        const scalar_t theta = scalar_t(2.0 * 3.14159265358979323846) * u2;
        const scalar_t z = radius * (low_lane ? ::cos(theta) : ::sin(theta));
        const int64_t i = element_index(t, k, per_thread * kPhiloxLanes);
        if (i < n) {
            out[i] = z * stddev + mean;
        }
    }
}

// low + floor(draw * range / 2^64) on a 64-bit draw, as randint_impl
template <typename scalar_t>
__global__ void randint_kernel(scalar_t* out, int64_t n, int64_t low, uint64_t range, uint64_t seed, uint64_t offset) {
    constexpr int per_batch = kPhiloxBatchWords / kRandintWordsPerElement;
    const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element_index(t, 0, per_batch) >= n) {
        return;
    }
    const PhiloxBlock r = philox4x32_10(offset + static_cast<uint64_t>(t), seed);
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int64_t i = element_index(t, k, per_batch);
        if (i < n) {
            const uint64_t draw = (static_cast<uint64_t>(r[k + 2]) << 32) | r[k];
            out[i] = static_cast<scalar_t>(low + static_cast<int64_t>(__umul64hi(draw, range)));
        }
    }
}

// One thread per Philox block the generator reserved
unsigned grid_for(int64_t n, int words_per_element) {
    const auto threads = static_cast<int64_t>(philox_blocks(n, words_per_element));
    return static_cast<unsigned>((threads + kThreads - 1) / kThreads);
}

void uniform_kernel_cuda(const Tensor& self, double low, double high, uint64_t seed, uint64_t offset) {
    DeviceGuard guard(self.device());
    const int64_t n = self.numel();
    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        uniform_kernel<scalar_t><<<grid_for(n, real_words_per_element(self.dtype())), kThreads, 0,
                                   cuda::current_stream()>>>(
            self.mutable_data_ptr<scalar_t>(), n, static_cast<scalar_t>(low), static_cast<scalar_t>(high),
            seed, offset);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void normal_kernel_cuda(const Tensor& self, double mean, double stddev, uint64_t seed, uint64_t offset) {
    DeviceGuard guard(self.device());
    const int64_t n = self.numel();
    self.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        normal_kernel<scalar_t><<<grid_for(n, real_words_per_element(self.dtype())), kThreads, 0,
                                  cuda::current_stream()>>>(
            self.mutable_data_ptr<scalar_t>(), n, static_cast<scalar_t>(mean), static_cast<scalar_t>(stddev),
            seed, offset);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

void randint_kernel_cuda(const Tensor& self, int64_t low, uint64_t range, uint64_t seed, uint64_t offset) {
    DeviceGuard guard(self.device());
    const int64_t n = self.numel();
    self.dtype().dispatch_integral_and_bool_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        randint_kernel<scalar_t><<<grid_for(n, kRandintWordsPerElement), kThreads, 0, cuda::current_stream()>>>(
            self.mutable_data_ptr<scalar_t>(), n, low, range, seed, offset);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

} // anonymous namespace

REGISTER_DISPATCH(uniform_stub, &uniform_kernel_cuda);
REGISTER_DISPATCH(normal_stub, &normal_kernel_cuda);
REGISTER_DISPATCH(randint_stub, &randint_kernel_cuda);

} // namespace raxis
