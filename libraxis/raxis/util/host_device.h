#pragma once


// Marks a function shared by CPU code and CUDA kernels: nvcc compiles it for
// both sides, a plain C++ compiler sees an ordinary function.
// From: c10/macros/Macros.h (C10_HOST_DEVICE)
#if defined(__CUDACC__)
#define RAXIS_HOST_DEVICE __host__ __device__
#else
#define RAXIS_HOST_DEVICE
#endif
