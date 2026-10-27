#pragma once

#include <cstdint>


// Reductions of one value per thread to one value per block, for kernels
// where a block owns a row or a channel (layer_norm, batch_norm).
// From: aten/src/ATen/native/cuda/block_reduce.cuh (BlockReduceSum,
// BlockReduce with WelfordOps)
//
// Two levels: each warp reduces its 32 values with shuffles (registers, no
// shared memory), then warp 0 reduces the per-warp results that went through
// shared memory. The block must be 1-D with a multiple of 32 threads; the
// result is valid in thread 0 only.

namespace raxis::cuda {

constexpr int kWarpSize = 32;

template <typename T>
__device__ inline T warp_sum(T value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

// shared must hold kWarpSize values
template <typename T>
__device__ inline T block_sum(T value, T* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warp_sum(value);
    __syncthreads();  // shared may still be read by a previous reduction
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < static_cast<int>(blockDim.x) / kWarpSize ? shared[lane] : T(0);
        value = warp_sum(value);
    }
    return value;
}

// Running mean and sum of squared deviations (m2) of n values. Two partial
// results merge exactly (Chan et al.), so threads can accumulate their own
// share and the shares combine in any tree. A plain aggregate (no default
// member initializers), so it can live in __shared__ memory.
// From: aten/src/ATen/native/SharedReduceOps.h (WelfordData, WelfordOps)
template <typename T>
struct WelfordData {
    T mean;
    T m2;
    int32_t n;
};

template <typename T>
__device__ inline WelfordData<T> welford_update(WelfordData<T> acc, T x) {
    acc.n += 1;
    const T delta = x - acc.mean;
    acc.mean += delta / static_cast<T>(acc.n);
    acc.m2 += delta * (x - acc.mean);
    return acc;
}

template <typename T>
__device__ inline WelfordData<T> welford_combine(WelfordData<T> a, WelfordData<T> b) {
    if (a.n == 0) {
        return b;
    }
    if (b.n == 0) {
        return a;
    }
    const int32_t n = a.n + b.n;
    const T delta = b.mean - a.mean;
    const T b_share = static_cast<T>(b.n) / static_cast<T>(n);
    return {a.mean + delta * b_share, a.m2 + b.m2 + delta * delta * static_cast<T>(a.n) * b_share, n};
}

template <typename T>
__device__ inline WelfordData<T> warp_welford(WelfordData<T> value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        WelfordData<T> other;
        other.mean = __shfl_down_sync(0xffffffff, value.mean, offset);
        other.m2 = __shfl_down_sync(0xffffffff, value.m2, offset);
        other.n = __shfl_down_sync(0xffffffff, value.n, offset);
        value = welford_combine(value, other);
    }
    return value;
}

// shared must hold kWarpSize values
template <typename T>
__device__ inline WelfordData<T> block_welford(WelfordData<T> value, WelfordData<T>* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    value = warp_welford(value);
    __syncthreads();
    if (lane == 0) {
        shared[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        value = lane < static_cast<int>(blockDim.x) / kWarpSize ? shared[lane] : WelfordData<T>{0, 0, 0};
        value = warp_welford(value);
    }
    return value;
}

} // namespace raxis::cuda
