#pragma once

#include <cuda_runtime.h>

#include <raxis/util/exception.h>


// From: c10/cuda/CUDAException.h
#define RAXIS_CUDA_CHECK(EXPR)                                                \
    do {                                                                      \
        const cudaError_t raxis_cuda_err = (EXPR);                            \
        RAXIS_CHECK(raxis_cuda_err == cudaSuccess,                            \
            "CUDA error: ", cudaGetErrorString(raxis_cuda_err));              \
    } while (0)

// A kernel launch returns nothing: a bad launch configuration (too many
// threads, too much shared memory) only shows up in cudaGetLastError. Errors
// inside the kernel surface later, at the next synchronizing call
#define RAXIS_CUDA_KERNEL_LAUNCH_CHECK() RAXIS_CUDA_CHECK(cudaGetLastError())
