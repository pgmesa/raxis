#include <raxis/cuda/cuda_blas.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <vector>

#include <raxis/cuda/cuda_functions.h>


namespace raxis::cuda::blas {

cublasHandle_t current_handle() {
    // PyTorch keeps a pool of handles per (thread, device): a handle must not
    // be used by two host threads at once. raxis launches from the thread
    // that runs the op (autograd included), so one per device is enough
    static std::mutex mutex;
    static auto* handles = new std::vector<cublasHandle_t>(device_count(), nullptr);
    const DeviceIndex device = current_device();
    cublasHandle_t handle;
    {
        std::lock_guard<std::mutex> lock(mutex);
        cublasHandle_t& slot = (*handles)[device];
        if (!slot) {
            RAXIS_CUBLAS_CHECK(cublasCreate(&slot));
            RAXIS_CUBLAS_CHECK(cublasSetMathMode(slot, CUBLAS_DEFAULT_MATH));
        }
        handle = slot;
    }
    RAXIS_CUBLAS_CHECK(cublasSetStream(handle, current_stream()));
    return handle;
}

namespace {

cublasOperation_t to_cublas_op(char trans) {
    return (trans == 't' || trans == 'T') ? CUBLAS_OP_T : CUBLAS_OP_N;
}

int to_int(int64_t value, const char* name) {
    RAXIS_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
        "cuBLAS: ", name, " = ", value, " does not fit in an int");
    return static_cast<int>(value);
}

// cuBLAS rejects a leading dimension smaller than the rows of the matrix even
// when a dimension of size 1 never uses it (a (1, k) row-major view can have
// any stride). Fixed up the same way as PyTorch.
// From: aten/src/ATen/cuda/detail/CublasLtUtils.h (cublasAdjustLdLevel3)
void adjust_ld(char transa, char transb, int64_t m, int64_t n, int64_t k,
               int64_t* lda, int64_t* ldb, int64_t* ldc) {
    const bool ta = to_cublas_op(transa) == CUBLAS_OP_T;
    const bool tb = to_cublas_op(transb) == CUBLAS_OP_T;
    if (n <= 1) {
        *ldc = std::max<int64_t>(m, 1);
    }
    if (ta) {
        if (m <= 1) *lda = std::max<int64_t>(k, 1);
    } else {
        if (k <= 1) *lda = std::max<int64_t>(m, 1);
    }
    if (tb) {
        if (k <= 1) *ldb = std::max<int64_t>(n, 1);
    } else {
        if (n <= 1) *ldb = std::max<int64_t>(k, 1);
    }
}

template <typename T> struct CublasFns;

template <> struct CublasFns<float> {
    template <typename... Args> static cublasStatus_t gemm(Args... args) { return cublasSgemm(args...); }
    template <typename... Args> static cublasStatus_t bgemm(Args... args) { return cublasSgemmStridedBatched(args...); }
    template <typename... Args> static cublasStatus_t gemv(Args... args) { return cublasSgemv(args...); }
    template <typename... Args> static cublasStatus_t dot(Args... args) { return cublasSdot(args...); }
};

template <> struct CublasFns<double> {
    template <typename... Args> static cublasStatus_t gemm(Args... args) { return cublasDgemm(args...); }
    template <typename... Args> static cublasStatus_t bgemm(Args... args) { return cublasDgemmStridedBatched(args...); }
    template <typename... Args> static cublasStatus_t gemv(Args... args) { return cublasDgemv(args...); }
    template <typename... Args> static cublasStatus_t dot(Args... args) { return cublasDdot(args...); }
};

} // anonymous namespace

template <typename T>
void gemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
          T alpha, const T* a, int64_t lda, const T* b, int64_t ldb,
          T beta, T* c, int64_t ldc) {
    adjust_ld(transa, transb, m, n, k, &lda, &ldb, &ldc);
    RAXIS_CUBLAS_CHECK(CublasFns<T>::gemm(
        current_handle(), to_cublas_op(transa), to_cublas_op(transb),
        to_int(m, "m"), to_int(n, "n"), to_int(k, "k"),
        &alpha, a, to_int(lda, "lda"), b, to_int(ldb, "ldb"),
        &beta, c, to_int(ldc, "ldc")));
}

template <typename T>
void bgemm(char transa, char transb, int64_t m, int64_t n, int64_t k,
           T alpha, const T* a, int64_t lda, int64_t stridea,
           const T* b, int64_t ldb, int64_t strideb,
           T beta, T* c, int64_t ldc, int64_t stridec, int64_t num_batches) {
    adjust_ld(transa, transb, m, n, k, &lda, &ldb, &ldc);
    RAXIS_CUBLAS_CHECK(CublasFns<T>::bgemm(
        current_handle(), to_cublas_op(transa), to_cublas_op(transb),
        to_int(m, "m"), to_int(n, "n"), to_int(k, "k"),
        &alpha, a, to_int(lda, "lda"), stridea, b, to_int(ldb, "ldb"), strideb,
        &beta, c, to_int(ldc, "ldc"), stridec, to_int(num_batches, "num_batches")));
}

template <typename T>
void gemv(char trans, int64_t m, int64_t n, T alpha, const T* a, int64_t lda,
          const T* x, int64_t incx, T beta, T* y, int64_t incy) {
    // From: aten/src/ATen/cuda/CUDABlas.cpp (_cublasAdjustLdLevel2)
    if (n <= 1) {
        lda = std::max<int64_t>(m, 1);
    }
    RAXIS_CUBLAS_CHECK(CublasFns<T>::gemv(
        current_handle(), to_cublas_op(trans), to_int(m, "m"), to_int(n, "n"),
        &alpha, a, to_int(lda, "lda"), x, to_int(incx, "incx"), &beta, y, to_int(incy, "incy")));
}

// cuBLAS returns a scalar result through a host pointer by default, which
// makes the call wait for the GPU; in device pointer mode it writes it to
// GPU memory like any kernel output. From: aten/src/ATen/native/cuda/Blas.cpp
// (dot_cuda, PointerModeGuard)
template <typename T>
void dot(int64_t n, const T* x, int64_t incx, const T* y, int64_t incy, T* result) {
    cublasHandle_t handle = current_handle();
    RAXIS_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
    const cublasStatus_t status = CublasFns<T>::dot(
        handle, to_int(n, "n"), x, to_int(incx, "incx"), y, to_int(incy, "incy"), result);
    RAXIS_CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
    RAXIS_CUBLAS_CHECK(status);
}

template void gemv<float>(char, int64_t, int64_t, float, const float*, int64_t,
                          const float*, int64_t, float, float*, int64_t);
template void gemv<double>(char, int64_t, int64_t, double, const double*, int64_t,
                           const double*, int64_t, double, double*, int64_t);
template void dot<float>(int64_t, const float*, int64_t, const float*, int64_t, float*);
template void dot<double>(int64_t, const double*, int64_t, const double*, int64_t, double*);
template void gemm<float>(char, char, int64_t, int64_t, int64_t, float, const float*, int64_t,
                          const float*, int64_t, float, float*, int64_t);
template void gemm<double>(char, char, int64_t, int64_t, int64_t, double, const double*, int64_t,
                           const double*, int64_t, double, double*, int64_t);
template void bgemm<float>(char, char, int64_t, int64_t, int64_t, float, const float*, int64_t, int64_t,
                           const float*, int64_t, int64_t, float, float*, int64_t, int64_t, int64_t);
template void bgemm<double>(char, char, int64_t, int64_t, int64_t, double, const double*, int64_t, int64_t,
                            const double*, int64_t, int64_t, double, double*, int64_t, int64_t, int64_t);

} // namespace raxis::cuda::blas
