#pragma once

#include <raxis/export.h>
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

} // namespace raxis