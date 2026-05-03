#pragma once

#include <cstdint>

#include <raxis/ops/dispatch_stub.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>
#include <raxis/core/scalar.h>


namespace raxis::cpublas {

enum class TransposeType : int8_t {
    NoTranspose,
    Transpose,
};

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    const float* a, int64_t lda,
    const float* b, int64_t ldb,
    float beta,
    float* c, int64_t ldc);

void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    const double* a, int64_t lda,
    const double* b, int64_t ldb,
    double beta,
    double* c, int64_t ldc);


// Used as the fallback path when no external BLAS is available.
using gemm_fn = void(*)(
    Dtype type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    Scalar alpha,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    Scalar beta,
    void* c, int64_t ldc);

DECLARE_DISPATCH(gemm_fn, gemm_stub);

template <typename scalar_t>
void gemm(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb,
    scalar_t beta,
    scalar_t* c, int64_t ldc
) {
    gemm_stub(
        DeviceType::CPU, Dtype::from_type<scalar_t>(),
        transa, transb, m, n, k,
        alpha, a, lda, b, ldb,
        beta, c, ldc);
}


// BLAS-accelerated overloads
float dot_impl(int64_t n, const float* x, int64_t incx, const float* y, int64_t incy);
double dot_impl(int64_t n, const double* x, int64_t incx, const double* y, int64_t incy);
 
namespace detail {
 
template <typename scalar_t>
scalar_t dot_naive(
    int64_t n, 
    const scalar_t* x, int64_t incx,
    const scalar_t* y, int64_t incy
) {
    constexpr int ilp = 4;
    std::array<scalar_t, ilp> partial{};
 
    int64_t i = 0;
    for (; i + ilp <= n; i += ilp)
        for (int p = 0; p < ilp; ++p)
            partial[p] += x[(i + p) * incx] * y[(i + p) * incy];
    for (; i < n; ++i)
        partial[0] += x[i * incx] * y[i * incy];
 
    scalar_t sum = partial[0];
    for (int p = 1; p < ilp; ++p) sum += partial[p];
    return sum;
}
 
} // namespace detail
 
template <typename scalar_t>
scalar_t dot_impl(
    int64_t n, 
    const scalar_t* x, int64_t incx,
    const scalar_t* y, int64_t incy
) {
    if (n == 1) { incx = 1; incy = 1; }
    return detail::dot_naive(n, x, incx, y, incy);
}
 
// BLAS-accelerated overloads
void gemv(
    TransposeType trans, int64_t m, int64_t n,
    float alpha, const float* a, int64_t lda,
    const float* x, int64_t incx,
    float beta, float* y, int64_t incy);
 
void gemv(
    TransposeType trans, int64_t m, int64_t n,
    double alpha, const double* a, int64_t lda,
    const double* x, int64_t incx,
    double beta, double* y, int64_t incy);
 
namespace detail {
 
template <typename scalar_t>
void gemv_naive(
    TransposeType trans, int64_t m, int64_t n,
    scalar_t alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* x, int64_t incx,
    scalar_t beta,
    scalar_t* y, int64_t incy
) {
    if (trans == TransposeType::Transpose) {
        constexpr int ilp = 4;
        for (int64_t i = 0; i < n; ++i) {
            const scalar_t* row = a + lda * i;
            std::array<scalar_t, ilp> partial{};
            int64_t j = 0;
            for (; j + ilp <= m; j += ilp)
                for (int p = 0; p < ilp; ++p)
                    partial[p] += x[(j + p) * incx] * row[j + p];
            for (; j < m; ++j)
                partial[0] += x[j * incx] * row[j];
 
            scalar_t sum = partial[0];
            for (int p = 1; p < ilp; ++p) sum += partial[p];
 
            if (beta == scalar_t(0))
                y[i * incy] = alpha * sum;
            else
                y[i * incy] = beta * y[i * incy] + alpha * sum;
        }
    } else {
        if (beta != scalar_t(1)) {
            for (int64_t i = 0; i < m; ++i)
                y[i * incy] = (beta == scalar_t(0)) ? scalar_t(0) : y[i * incy] * beta;
        }
 
        for (int64_t j = 0; j < n; ++j) {
            const scalar_t* column = a + lda * j;
            scalar_t z = alpha * x[j * incx];
            for (int64_t i = 0; i < m; ++i)
                y[i * incy] += z * column[i];
        }
    }
}
 
} // namespace detail
 
template <typename scalar_t>
void gemv(
    TransposeType trans, int64_t m, int64_t n,
    scalar_t alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* x, int64_t incx,
    scalar_t beta,
    scalar_t* y, int64_t incy
) {
    detail::gemv_naive(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

} // namespace raxis::cpublas