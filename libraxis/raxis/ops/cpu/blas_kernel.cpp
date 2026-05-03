#include <raxis/ops/cpu_blas.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>

#include <array>
#include <cstdint>


namespace raxis {

namespace {

// ---------------------------------------------------------------------------
// Naive GEMM kernels
//
// Column-major layout: element (i,j) of an MxN matrix with leading
// dimension ld is at pointer[j * ld + i].
//
// Loop ordering and manual unrolling follow PyTorch's BlasKernel.cpp for
// cache friendliness and ILP.
// ---------------------------------------------------------------------------

using cpublas::TransposeType;

template <typename scalar_t>
void scale_(int64_t m, int64_t n, scalar_t alpha, scalar_t* a, int64_t lda) {
    if (alpha == scalar_t(1)) return;

    if (alpha == scalar_t(0)) {
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < m; ++i)
                a[j * lda + i] = scalar_t(0);
        return;
    }

    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            a[j * lda + i] *= alpha;
}

// C = alpha * A * B + beta * C  (no transposes)
// k-j-i loop: broadcast scalar from B against column of A into column of C.
template <typename scalar_t>
void gemm_notrans_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb,
    scalar_t beta,
    scalar_t* c, int64_t ldc
) {
    scale_(m, n, beta, c, ldc);

    const int64_t i_m = m / 4;
    for (int64_t l = 0; l < k; ++l) {
        for (int64_t j = 0; j < n; ++j) {
            scalar_t val = b[l + j * ldb] * alpha;
            int64_t i = 0;
            for (int64_t ii = 0; ii < i_m; ++ii, i += 4) {
                c[j * ldc + i + 0] += a[i + 0 + l * lda] * val;
                c[j * ldc + i + 1] += a[i + 1 + l * lda] * val;
                c[j * ldc + i + 2] += a[i + 2 + l * lda] * val;
                c[j * ldc + i + 3] += a[i + 3 + l * lda] * val;
            }
            for (; i < m; ++i)
                c[j * ldc + i] += a[i + l * lda] * val;
        }
    }
}

// C = alpha * A^T * B + beta * C
template <typename scalar_t>
void gemm_transa_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb,
    scalar_t beta,
    scalar_t* c, int64_t ldc
) {
    constexpr int ilp = 4;

    const scalar_t* a_ = a;
    for (int64_t i = 0; i < m; ++i) {
        const scalar_t* b_ = b;
        for (int64_t j = 0; j < n; ++j) {
            std::array<scalar_t, ilp> partial{};
            int64_t l = 0;
            for (; l + ilp <= k; l += ilp)
                for (int p = 0; p < ilp; ++p)
                    partial[p] += a_[l + p] * b_[l + p];
            for (; l < k; ++l)
                partial[0] += a_[l] * b_[l];

            scalar_t dot = partial[0];
            for (int p = 1; p < ilp; ++p) dot += partial[p];

            b_ += ldb;
            if (beta == scalar_t(0))
                c[j * ldc + i] = alpha * dot;
            else
                c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
        }
        a_ += lda;
    }
}

// C = alpha * A * B^T + beta * C
template <typename scalar_t>
void gemm_transb_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb,
    scalar_t beta,
    scalar_t* c, int64_t ldc
) {
    scale_(m, n, beta, c, ldc);

    const int64_t i_m = m / 4;
    for (int64_t l = 0; l < k; ++l) {
        for (int64_t j = 0; j < n; ++j) {
            scalar_t val = b[j + l * ldb] * alpha;
            int64_t i = 0;
            for (int64_t ii = 0; ii < i_m; ++ii, i += 4) {
                c[j * ldc + i + 0] += a[i + 0 + l * lda] * val;
                c[j * ldc + i + 1] += a[i + 1 + l * lda] * val;
                c[j * ldc + i + 2] += a[i + 2 + l * lda] * val;
                c[j * ldc + i + 3] += a[i + 3 + l * lda] * val;
            }
            for (; i < m; ++i)
                c[j * ldc + i] += a[i + l * lda] * val;
        }
    }
}

// C = alpha * A^T * B^T + beta * C
template <typename scalar_t>
void gemm_transab_(
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb,
    scalar_t beta,
    scalar_t* c, int64_t ldc
) {
    constexpr int ilp = 4;

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            std::array<scalar_t, ilp> partial{};
            int64_t l = 0;
            for (; l + ilp <= k; l += ilp)
                for (int p = 0; p < ilp; ++p)
                    partial[p] += a[i * lda + l + p] * b[(l + p) * ldb + j];
            for (; l < k; ++l)
                partial[0] += a[i * lda + l] * b[l * ldb + j];

            scalar_t dot = partial[0];
            for (int p = 1; p < ilp; ++p) dot += partial[p];

            if (beta == scalar_t(0))
                c[j * ldc + i] = alpha * dot;
            else
                c[j * ldc + i] = beta * c[j * ldc + i] + alpha * dot;
        }
    }
}

template <typename scalar_t>
void gemm_core_(
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    const scalar_t* a, int64_t lda,
    const scalar_t* b, int64_t ldb,
    scalar_t beta,
    scalar_t* c, int64_t ldc
) {
    if (transa == TransposeType::NoTranspose && transb == TransposeType::NoTranspose)
        gemm_notrans_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else if (transa == TransposeType::Transpose && transb == TransposeType::NoTranspose)
        gemm_transa_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else if (transa == TransposeType::NoTranspose && transb == TransposeType::Transpose)
        gemm_transb_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    else
        gemm_transab_(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void gemm_kernel(
    Dtype type,
    TransposeType transa, TransposeType transb,
    int64_t m, int64_t n, int64_t k,
    Scalar alpha,
    const void* a, int64_t lda,
    const void* b, int64_t ldb,
    Scalar beta,
    void* c, int64_t ldc
) {
    type.dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        gemm_core_(
            transa, transb, m, n, k,
            alpha.to<scalar_t>(),
            static_cast<const scalar_t*>(a), lda,
            static_cast<const scalar_t*>(b), ldb,
            beta.to<scalar_t>(),
            static_cast<scalar_t*>(c), ldc);
    });
}

} // namespace
 
REGISTER_DISPATCH(cpublas::gemm_stub, &gemm_kernel);

} // namespace raxis