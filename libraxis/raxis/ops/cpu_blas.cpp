#include <raxis/ops/cpu_blas.h>

#include <algorithm>
#include <climits>

#include <raxis/core/dtype.h>

#if RAXIS_BUILD_WITH_BLAS
#if RAXIS_BUILD_WITH_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#endif


namespace raxis::cpublas {

DEFINE_DISPATCH(gemm_stub);

namespace {

// From: aten/src/ATen/native/CPUBlas.cpp (normalize_last_dims)
// BLAS rejects a leading dimension smaller than the rows of the matrix even
// when there is a single column and the value is never used: fix it up for
// size 1 dims, where any stride is valid for the tensor
void normalize_last_dims(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t* lda, int64_t* ldb, int64_t* ldc
) {
    if (n == 1) {
        *ldc = m;
    }
    if (transa != TransposeType::NoTranspose) {
        if (m == 1) {
            *lda = k;
        }
    } else if (k == 1) {
        *lda = m;
    }
    if (transb != TransposeType::NoTranspose) {
        if (k == 1) {
            *ldb = n;
        }
    } else if (n == 1) {
        *ldb = k;
    }
}

// From: aten/src/ATen/native/CPUBlas.cpp (use_blas_gemm)
// The sizes fit the int arguments of cblas and the leading dimensions are valid
[[maybe_unused]] bool use_blas_gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc
) {
    const bool transa_ = transa != TransposeType::NoTranspose;
    const bool transb_ = transb != TransposeType::NoTranspose;
    return m <= INT_MAX && n <= INT_MAX && k <= INT_MAX
        && lda <= INT_MAX && ldb <= INT_MAX && ldc <= INT_MAX
        && lda >= std::max(int64_t{1}, transa_ ? k : m)
        && ldb >= std::max(int64_t{1}, transb_ ? n : k)
        && ldc >= std::max(int64_t{1}, m);
}

template <typename scalar_t>
void gemm_batched_with_stride_generic(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t* a, int64_t lda, int64_t batch_stride_a,
    const scalar_t* b, int64_t ldb, int64_t batch_stride_b,
    scalar_t beta,
    scalar_t* c, int64_t ldc, int64_t batch_stride_c
) {
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        gemm(transa, transb, m, n, k, alpha,
             a + batch_stride_a * batch, lda,
             b + batch_stride_b * batch, ldb,
             beta, c + batch_stride_c * batch, ldc);
    }
}

#if RAXIS_BUILD_WITH_MKL
CBLAS_TRANSPOSE to_cblas(TransposeType t) {
    return t == TransposeType::Transpose ? CblasTrans : CblasNoTrans;
}
#endif

} // namespace

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float* a, int64_t lda,
    const float* b, int64_t ldb,
    float beta,
    float* c, int64_t ldc)
{
#if RAXIS_BUILD_WITH_BLAS
    auto to_cblas = [](TransposeType t) {
        return t == TransposeType::Transpose ? CblasTrans : CblasNoTrans;
    };
    cblas_sgemm(
        CblasColMajor, to_cblas(transa), to_cblas(transb),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb),
        beta, c, static_cast<int>(ldc));
#else
    gemm<float>(
        transa, transb, m, n, k,
        alpha, a, lda, b, ldb,
        beta, c, ldc);
#endif
}

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    const double* a, int64_t lda,
    const double* b, int64_t ldb,
    double beta,
    double* c, int64_t ldc)
{
#if RAXIS_BUILD_WITH_BLAS
    auto to_cblas = [](TransposeType t) {
        return t == TransposeType::Transpose ? CblasTrans : CblasNoTrans;
    };
    cblas_dgemm(
        CblasColMajor, to_cblas(transa), to_cblas(transb),
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
        alpha, a, static_cast<int>(lda), b, static_cast<int>(ldb),
        beta, c, static_cast<int>(ldc));
#else
    gemm<double>(
        transa, transb, m, n, k,
        alpha, a, lda, b, ldb,
        beta, c, ldc);
#endif
}


float dot_impl(int64_t n, const float* x, int64_t incx, const float* y, int64_t incy) {
    if (n == 1) { incx = 1; incy = 1; }
#if RAXIS_BUILD_WITH_BLAS
    if (n <= INT_MAX && incx <= INT_MAX && incy <= INT_MAX)
        return cblas_sdot(
            static_cast<int>(n), x, static_cast<int>(incx),
            y, static_cast<int>(incy));
#endif
    return detail::dot_naive(n, x, incx, y, incy);
}
 
double dot_impl(int64_t n, const double* x, int64_t incx, const double* y, int64_t incy) {
    if (n == 1) { incx = 1; incy = 1; }
#if RAXIS_BUILD_WITH_BLAS
    if (n <= INT_MAX && incx <= INT_MAX && incy <= INT_MAX)
        return cblas_ddot(
            static_cast<int>(n), x, static_cast<int>(incx),
            y, static_cast<int>(incy));
#endif
    return detail::dot_naive(n, x, incx, y, incy);
}
 
 
void gemv(
    TransposeType trans, int64_t m, int64_t n,
    float alpha, const float* a, int64_t lda,
    const float* x, int64_t incx,
    float beta, float* y, int64_t incy)
{
#if RAXIS_BUILD_WITH_BLAS
    auto to_cblas = [](TransposeType t) {
        return t == TransposeType::Transpose ? CblasTrans : CblasNoTrans;
    };
    cblas_sgemv(
        CblasColMajor, to_cblas(trans),
        static_cast<int>(m), static_cast<int>(n),
        alpha, a, static_cast<int>(lda),
        x, static_cast<int>(incx),
        beta, y, static_cast<int>(incy));
#else
    detail::gemv_naive(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#endif
}
 
void gemv(
    TransposeType trans, int64_t m, int64_t n,
    double alpha, const double* a, int64_t lda,
    const double* x, int64_t incx,
    double beta, double* y, int64_t incy)
{
#if RAXIS_BUILD_WITH_BLAS
    auto to_cblas = [](TransposeType t) {
        return t == TransposeType::Transpose ? CblasTrans : CblasNoTrans;
    };
    cblas_dgemv(
        CblasColMajor, to_cblas(trans),
        static_cast<int>(m), static_cast<int>(n),
        alpha, a, static_cast<int>(lda),
        x, static_cast<int>(incx),
        beta, y, static_cast<int>(incy));
#else
    detail::gemv_naive(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
#endif
}

// From: aten/src/ATen/native/CPUBlas.cpp (gemm_batched_with_stride)
// PyTorch builds arrays of pointers for cblas_?gemm_batch; the strided
// variant (MKL 2020+) takes the batch strides directly.
void gemm_batched_with_stride(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    float alpha,
    const float* a, int64_t lda, int64_t batch_stride_a,
    const float* b, int64_t ldb, int64_t batch_stride_b,
    float beta,
    float* c, int64_t ldc, int64_t batch_stride_c)
{
    if (batch_size == 1) {
        return gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if RAXIS_BUILD_WITH_MKL
    if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
        cblas_sgemm_batch_strided(
            CblasColMajor, to_cblas(transa), to_cblas(transb),
            m, n, k, alpha, a, lda, batch_stride_a, b, ldb, batch_stride_b,
            beta, c, ldc, batch_stride_c, batch_size);
        return;
    }
#endif
    gemm_batched_with_stride_generic(
        transa, transb, batch_size, m, n, k, alpha, a, lda, batch_stride_a,
        b, ldb, batch_stride_b, beta, c, ldc, batch_stride_c);
}

void gemm_batched_with_stride(
    TransposeType transa, TransposeType transb,
    int64_t batch_size, int64_t m, int64_t n, int64_t k,
    double alpha,
    const double* a, int64_t lda, int64_t batch_stride_a,
    const double* b, int64_t ldb, int64_t batch_stride_b,
    double beta,
    double* c, int64_t ldc, int64_t batch_stride_c)
{
    if (batch_size == 1) {
        return gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    normalize_last_dims(transa, transb, m, n, k, &lda, &ldb, &ldc);
#if RAXIS_BUILD_WITH_MKL
    if (use_blas_gemm(transa, transb, m, n, k, lda, ldb, ldc)) {
        cblas_dgemm_batch_strided(
            CblasColMajor, to_cblas(transa), to_cblas(transb),
            m, n, k, alpha, a, lda, batch_stride_a, b, ldb, batch_stride_b,
            beta, c, ldc, batch_stride_c, batch_size);
        return;
    }
#endif
    gemm_batched_with_stride_generic(
        transa, transb, batch_size, m, n, k, alpha, a, lda, batch_stride_a,
        b, ldb, batch_stride_b, beta, c, ldc, batch_stride_c);
}

} // namespace raxis::cpublas