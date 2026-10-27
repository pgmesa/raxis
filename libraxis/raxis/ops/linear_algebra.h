#pragma once

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

RAXIS_API Tensor mm(const Tensor& mat1, const Tensor& mat2);
RAXIS_API Tensor& mm_out(const Tensor& mat1, const Tensor& mat2, Tensor& out);

RAXIS_API Tensor addmm(
    const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    Scalar beta = 1, Scalar alpha = 1);

RAXIS_API Tensor& addmm_out(
    const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    Scalar beta, Scalar alpha, Tensor& out);

RAXIS_API Tensor bmm(const Tensor& self, const Tensor& mat2);
RAXIS_API Tensor& bmm_out(const Tensor& self, const Tensor& mat2, Tensor& out);
 
RAXIS_API Tensor addbmm(
    const Tensor& self, const Tensor& batch1, const Tensor& batch2,
    Scalar beta = 1, Scalar alpha = 1);
 
RAXIS_API Tensor& addbmm_out(
    const Tensor& self, const Tensor& batch1, const Tensor& batch2,
    Scalar beta, Scalar alpha, Tensor& out);

RAXIS_API Tensor mv(const Tensor& mat, const Tensor& vec);
RAXIS_API Tensor& mv_out(const Tensor& mat, const Tensor& vec, Tensor& out);
 
RAXIS_API Tensor addmv(
    const Tensor& self, const Tensor& mat, const Tensor& vec,
    Scalar beta = 1, Scalar alpha = 1);
 
RAXIS_API Tensor& addmv_out(
    const Tensor& self, const Tensor& mat, const Tensor& vec,
    Scalar beta, Scalar alpha, Tensor& out);

RAXIS_API Tensor dot(const Tensor& self, const Tensor& other);

RAXIS_API Tensor matmul(const Tensor& tensor1, const Tensor& tensor2);

// y = x @ weight.T + bias. weight: [out_features, in_features]
RAXIS_API Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias = {});


namespace ops {

RAXIS_API Tensor mm(const Tensor& mat1, const Tensor& mat2);

RAXIS_API Tensor addmm(
    const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    Scalar beta = 1, Scalar alpha = 1);

RAXIS_API Tensor bmm(const Tensor& self, const Tensor& mat2);
 
RAXIS_API Tensor addbmm(
    const Tensor& self, const Tensor& batch1, const Tensor& batch2,
    Scalar beta = 1, Scalar alpha = 1);
 
RAXIS_API Tensor mv(const Tensor& mat, const Tensor& vec);
 
RAXIS_API Tensor addmv(
    const Tensor& self, const Tensor& mat, const Tensor& vec,
    Scalar beta = 1, Scalar alpha = 1);

RAXIS_API Tensor dot(const Tensor& self, const Tensor& other);

}

// result = beta * self + alpha * (m1 @ m2), and its batched form. The CPU
// implementations (BLAS) are in linear_algebra.cpp, the CUDA ones (cuBLAS)
// in ops/cuda/blas.cpp: PyTorch registers them as the CPU and CUDA kernels
// of addmm / baddbmm (addmm_out_cpu, addmm_out_cuda_impl).
using addmm_fn = void(*)(
    Tensor& result, const Tensor& self, Tensor m1, Tensor m2,
    const Scalar& beta, const Scalar& alpha);
using baddbmm_fn = void(*)(
    Tensor& self_or_result, const Tensor& batch1, const Tensor& batch2,
    const Scalar& beta, const Scalar& alpha, bool is_bmm_out);
// out = beta * out + alpha * (mat @ vec); out already holds self
using addmv_fn = void(*)(
    Tensor& out, const Tensor& mat, const Tensor& vec, const Scalar& beta, const Scalar& alpha);
// result (0-d) = self . other
using dot_fn = void(*)(Tensor& result, const Tensor& self, const Tensor& other);
DECLARE_DISPATCH(addmm_fn, addmm_stub);
DECLARE_DISPATCH(baddbmm_fn, baddbmm_stub);
DECLARE_DISPATCH(addmv_fn, addmv_stub);
DECLARE_DISPATCH(dot_fn, dot_stub);

} // namespace raxis