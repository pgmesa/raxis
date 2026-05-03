#include <raxis/ops/cpu_blas.h>

#include <raxis/core/dtype.h>

#if RAXIS_BUILD_WITH_BLAS
#include <cblas.h>
#endif


namespace raxis::cpublas {

DEFINE_DISPATCH(gemm_stub);

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

} // namespace raxis::cpublas