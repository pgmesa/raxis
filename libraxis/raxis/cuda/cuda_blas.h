#pragma once

#include <cstdint>

#include <cublas_v2.h>

#include <raxis/util/exception.h>


// From: aten/src/ATen/cuda/Exceptions.h (TORCH_CUDABLAS_CHECK)
#define RAXIS_CUBLAS_CHECK(EXPR)                                              \
    do {                                                                      \
        const cublasStatus_t raxis_cublas_status = (EXPR);                    \
        RAXIS_CHECK(raxis_cublas_status == CUBLAS_STATUS_SUCCESS,             \
            "cuBLAS error: ", cublasGetStatusString(raxis_cublas_status));    \
    } while (0)


namespace raxis::cuda::blas {

/// The cuBLAS handle of the current device, bound to the current stream.
/// One handle per device, created on first use and never destroyed (as the
/// caching allocator: destroying it at exit could run after the driver is
/// gone). Its math mode is set by each gemm (see allow_tf32 below).
/// From: aten/src/ATen/cuda/CublasHandlePool.cpp (getCurrentCUDABlasHandle)
cublasHandle_t current_handle();

/// Column-major C = alpha * op(A) @ op(B) + beta * C, with op(X) = X or X^T
/// as transa / transb ('n' or 't'): BLAS's convention. Row-major callers
/// swap the operands (see ops/cuda/blas.cpp).
/// From: aten/src/ATen/cuda/CUDABlas.cpp (gemm, bgemm)
template <typename T>
void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
          T alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
          T beta, T* c, int64_t ldc, bool allow_tf32);

/// Column-major y = alpha * op(A) @ x + beta * y
template <typename T>
void gemv(char trans, int64_t m, int64_t n, T alpha, const T* a, int64_t lda,
          const T* x, int64_t incx, T beta, T* y, int64_t incy);

/// *result = x . y, with result in device memory (no wait for the GPU)
template <typename T>
void dot(int64_t n, const T* x, int64_t incx, const T* y, int64_t incy, T* result);

/// allow_tf32: float32 products on the tensor cores with inputs rounded
/// to TF32 (CUBLAS_TF32_TENSOR_OP_MATH); ignored for double. Callers pass
/// allow_tf32_matmul() or allow_tf32_conv() (core/context.h).
///
/// gemm on num_batches matrices, each `stride` elements after the previous
/// (a stride of 0 reuses one matrix for every batch)
template <typename T>
void bgemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
           T alpha, const T* a, int64_t lda, int64_t stridea,
           const T* b, int64_t ldb, int64_t strideb,
           T beta, T* c, int64_t ldc, int64_t stridec, int64_t num_batches, bool allow_tf32);

} // namespace raxis::cuda::blas
