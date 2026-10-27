#include <algorithm>
#include <cstdint>

#include <raxis/core/context.h>
#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_blas.h>
#include <raxis/ops/device_check.h>
#include <raxis/ops/linear_algebra.h>
#include <raxis/tensor/tensor.h>


// addmm and bmm on the GPU with cuBLAS.
// From: aten/src/ATen/native/cuda/Blas.cpp (addmm_out_cuda_impl,
// baddbmm_out_cuda_impl) and cuBlasCommonArgs.h
//
// cuBLAS, as Fortran BLAS, reads column-major matrices. A row-major matrix
// is the transpose of a column-major one with the same memory, so instead
// of copying, every operand gets a transpose flag from its strides, and a
// row-major result is computed through C^T = B^T A^T: cuBLAS writes C^T in
// column-major, which is C in row-major. Only a matrix with no unit stride
// (a strided slice) is copied first.
//
// PyTorch fuses the bias of a 1-D self into the gemm epilogue (cublasLt,
// what nn.Linear hits); here the bias is copied into the result and the
// gemm adds to it (beta = 1), one extra pass over the output.

namespace raxis {
namespace {

void check_floating(const char* op, const Tensor& result) {
    RAXIS_CHECK_NOT_IMPLEMENTED(result.dtype().is_floating(),
        op, " on CUDA is only implemented for floating types (as in PyTorch), got ", result.dtype());
}

// From: aten/src/ATen/native/cuda/cuBlasCommonArgs.h (prepare_matrix_for_cublas)
// Returns the matrix cuBLAS reads (a copy only if no stride is 1) and
// whether cuBLAS must take it as transposed: a row-major matrix is a
// transposed column-major one
Tensor prepare_matrix_for_cublas(const Tensor& tensor, bool& transpose_tensor) {
    if (tensor.is_non_overlapping_and_dense()) {
        transpose_tensor = tensor.is_contiguous();
        return tensor;
    }
    IntArrayRef strides = tensor.strides();
    IntArrayRef sizes = tensor.shape();
    if (strides[0] == 1 && strides[1] >= std::max<int64_t>(1, sizes[0])) {
        transpose_tensor = false;
        return tensor;
    }
    if (strides[1] == 1 && strides[0] >= std::max<int64_t>(1, sizes[1])) {
        transpose_tensor = true;
        return tensor;
    }
    transpose_tensor = true;
    return tensor.clone(MemoryFormat::Contiguous);
}

// result = beta * self + alpha * (m1 @ m2); self is already expanded to
// the result's shape
void addmm_kernel_cuda(
    Tensor& result, const Tensor& self, Tensor m1, Tensor m2, const Scalar& beta, const Scalar& alpha
) {
    check_same_device("addmm", result, self, m1, m2);
    check_floating("addmm", result);
    RAXIS_CHECK(m1.dtype() == m2.dtype(),
        "expected m1 and m2 to have the same dtype, but got: ", m1.dtype(), " != ", m2.dtype());
    RAXIS_CHECK(m1.shape()[1] == m2.shape()[0],
        "mat1 and mat2 shapes cannot be multiplied (",
        m1.shape()[0], "x", m1.shape()[1], " and ", m2.shape()[0], "x", m2.shape()[1], ")");
    DeviceGuard guard(result.device());

    const bool beta_is_zero = beta.to<double>() == 0.0;
    if (!beta_is_zero && !result.is_same(self)) {
        result.copy_(self);
    }
    if (result.numel() == 0) {
        return;
    }
    if (m1.shape()[1] == 0) {
        if (beta_is_zero) {
            result.zero_();
        } else {
            result.mul_(beta);
        }
        return;
    }

    // From: cublasCommonArgs. With a row-major result the operands swap
    // places (C^T = m2^T m1^T) and their transpose flags flip
    bool transpose_result = false, transpose_a = false, transpose_b = false;
    Tensor c = prepare_matrix_for_cublas(result, transpose_result);
    Tensor a = prepare_matrix_for_cublas(transpose_result ? m2 : m1, transpose_a);
    Tensor b = prepare_matrix_for_cublas(transpose_result ? m1 : m2, transpose_b);
    if (transpose_result) {
        transpose_a = !transpose_a;
        transpose_b = !transpose_b;
    }
    const int64_t m = a.shape()[transpose_result ? 1 : 0];
    const int64_t k = a.shape()[transpose_result ? 0 : 1];
    const int64_t n = b.shape()[transpose_result ? 0 : 1];
    const int64_t lda = a.strides()[(transpose_a == transpose_result) ? 1 : 0];
    const int64_t ldb = b.strides()[(transpose_b == transpose_result) ? 1 : 0];
    const int64_t ldc = c.strides()[transpose_result ? 0 : 1];

    result.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::blas::gemm<scalar_t>(
            transpose_a ? 't' : 'n', transpose_b ? 't' : 'n', m, n, k,
            alpha.to<scalar_t>(), a.data_ptr<scalar_t>(), lda, b.data_ptr<scalar_t>(), ldb,
            beta_is_zero ? scalar_t(0) : beta.to<scalar_t>(), c.mutable_data_ptr<scalar_t>(), ldc,
            allow_tf32_matmul());
    });
    if (!c.is_same(result)) {
        result.copy_(c);
    }
}

// From: aten/src/ATen/native/cuda/Blas.cpp (prepare_batch_matrix_for_cublas)
// The same choice for each matrix of a batch; the batch dim keeps its
// stride (cuBLAS's strided batched gemm takes it as is)
Tensor prepare_batch_matrix_for_cublas(
    const Tensor& tensor, bool& transpose_tensor, int64_t& ld_tensor,
    bool transpose_result, int64_t m, int64_t n
) {
    IntArrayRef strides = tensor.strides();
    const int fast_dim = transpose_result ? 2 : 1;
    const int leading_dim = transpose_result ? 1 : 2;
    if (strides[fast_dim] == 1 && strides[leading_dim] >= std::max<int64_t>(1, m)) {
        transpose_tensor = false;
        ld_tensor = strides[leading_dim];
        return tensor;
    }
    if (strides[leading_dim] == 1 && strides[fast_dim] >= std::max<int64_t>(1, n)) {
        transpose_tensor = true;
        ld_tensor = strides[fast_dim];
        return tensor;
    }
    transpose_tensor = !transpose_result;
    // A gemm needs non-zero leading dimensions
    const bool is_stride_non_zero = strides[1] != 0 && strides[2] != 0;
    Tensor result = tensor.is_contiguous() && is_stride_non_zero ? tensor : tensor.clone(MemoryFormat::Contiguous);
    ld_tensor = result.strides()[1];
    return result;
}

// result[i] = beta * result[i] + alpha * (batch1[i] @ batch2[i]); with
// beta != 0 the result already holds self
void baddbmm_kernel_cuda(
    Tensor& result, const Tensor& batch1, const Tensor& batch2,
    const Scalar& beta, const Scalar& alpha, bool /*is_bmm_out*/
) {
    check_same_device("bmm", result, batch1, batch2);
    check_floating("bmm", result);
    DeviceGuard guard(result.device());
    if (result.numel() == 0) {
        return;
    }
    const bool beta_is_zero = beta.to<double>() == 0.0;
    if (batch1.shape()[2] == 0) {
        if (beta_is_zero) {
            result.zero_();
        } else {
            result.mul_(beta);
        }
        return;
    }

    bool transpose_result = false;
    Tensor result_;
    IntArrayRef result_strides = result.strides();
    IntArrayRef result_sizes = result.shape();
    if (result_strides[1] == 1
            && (result_sizes[2] == 1 || result_strides[2] >= std::max<int64_t>(1, result_sizes[1]))) {
        result_ = result;
    } else if (result_strides[2] == 1
            && (result_sizes[1] == 1 || result_strides[1] >= std::max<int64_t>(1, result_sizes[2]))) {
        transpose_result = true;
        result_ = result;
    } else {
        result_ = result.transpose(1, 2).clone(MemoryFormat::Contiguous).transpose(1, 2);
    }

    const int leading_dim = transpose_result ? 1 : 2;
    const int64_t m = result_sizes[transpose_result ? 2 : 1];
    const int64_t n = result_sizes[leading_dim];
    const int64_t k = (transpose_result ? batch2 : batch1).shape()[leading_dim];
    int64_t lda = 0, ldb = 0;
    bool transpose_batch1 = false, transpose_batch2 = false;
    const Tensor batch1_ = prepare_batch_matrix_for_cublas(
        transpose_result ? batch2 : batch1, transpose_batch1, lda, transpose_result, m, k);
    const Tensor batch2_ = prepare_batch_matrix_for_cublas(
        transpose_result ? batch1 : batch2, transpose_batch2, ldb, transpose_result, k, n);
    const int64_t ldc = result_.strides()[leading_dim];
    const int64_t num_batches = result_.shape()[0];

    result.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const char transa = transpose_batch1 ? 't' : 'n';
        const char transb = transpose_batch2 ? 't' : 'n';
        const scalar_t alpha_val = alpha.to<scalar_t>();
        const scalar_t beta_val = beta_is_zero ? scalar_t(0) : beta.to<scalar_t>();
        if (num_batches == 1) {
            cuda::blas::gemm<scalar_t>(
                transa, transb, m, n, k, alpha_val,
                batch1_.data_ptr<scalar_t>(), lda, batch2_.data_ptr<scalar_t>(), ldb,
                beta_val, result_.mutable_data_ptr<scalar_t>(), ldc, allow_tf32_matmul());
        } else {
            cuda::blas::bgemm<scalar_t>(
                transa, transb, m, n, k, alpha_val,
                batch1_.data_ptr<scalar_t>(), lda, batch1_.strides()[0],
                batch2_.data_ptr<scalar_t>(), ldb, batch2_.strides()[0],
                beta_val, result_.mutable_data_ptr<scalar_t>(), ldc, result_.strides()[0], num_batches,
                allow_tf32_matmul());
        }
    });
    if (!result_.is_same(result)) {
        result.copy_(result_);
    }
}

// From: aten/src/ATen/native/cuda/Blas.cpp (addmv_out_cuda): a column-major
// mat is gemv 'n', a row-major one gemv 't' of its transpose
void addmv_kernel_cuda(Tensor& out, const Tensor& mat, const Tensor& vec, const Scalar& beta, const Scalar& alpha) {
    check_floating("addmv", out);
    DeviceGuard guard(out.device());
    // A zero stride is not a valid increment for cuBLAS
    const Tensor vec_contiguous = vec.stride(0) == 0 ? vec.contiguous() : vec;
    const int64_t vec_stride = std::max<int64_t>(vec_contiguous.stride(0), 1);
    const int64_t r_stride = out.stride(0);
    const int64_t rows = mat.shape()[0];
    const int64_t cols = mat.shape()[1];
    out.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t a = alpha.to<scalar_t>();
        const scalar_t b = beta.to<double>() == 0.0 ? scalar_t(0) : beta.to<scalar_t>();
        const scalar_t* v = vec_contiguous.data_ptr<scalar_t>();
        scalar_t* y = out.mutable_data_ptr<scalar_t>();
        if (mat.stride(0) == 1 && mat.stride(1) >= std::max<int64_t>(1, rows)) {
            cuda::blas::gemv<scalar_t>('n', rows, cols, a, mat.data_ptr<scalar_t>(), mat.stride(1),
                                       v, vec_stride, b, y, r_stride);
        } else if (mat.stride(1) == 1 && mat.stride(0) >= std::max<int64_t>(1, cols)) {
            cuda::blas::gemv<scalar_t>('t', cols, rows, a, mat.data_ptr<scalar_t>(), mat.stride(0),
                                       v, vec_stride, b, y, r_stride);
        } else {
            const Tensor cmat = mat.contiguous();
            cuda::blas::gemv<scalar_t>('t', cols, rows, a, cmat.data_ptr<scalar_t>(), cmat.stride(0),
                                       v, vec_stride, b, y, r_stride);
        }
    });
}

void dot_kernel_cuda(Tensor& result, const Tensor& self, const Tensor& other) {
    check_floating("dot", result);
    DeviceGuard guard(result.device());
    const int64_t n = self.numel();
    // A single element: any stride works, cuBLAS wants a positive one
    const int64_t incx = n == 1 ? 1 : self.stride(0);
    const int64_t incy = n == 1 ? 1 : other.stride(0);
    result.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::blas::dot<scalar_t>(n, self.data_ptr<scalar_t>(), incx, other.data_ptr<scalar_t>(), incy,
                                  result.mutable_data_ptr<scalar_t>());
    });
}

} // anonymous namespace

REGISTER_CUDA_DISPATCH(addmm_stub, &addmm_kernel_cuda)
REGISTER_CUDA_DISPATCH(baddbmm_stub, &baddbmm_kernel_cuda)
REGISTER_CUDA_DISPATCH(addmv_stub, &addmv_kernel_cuda)
REGISTER_CUDA_DISPATCH(dot_stub, &dot_kernel_cuda)

} // namespace raxis
