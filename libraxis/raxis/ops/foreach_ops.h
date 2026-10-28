#pragma once

#include <vector>

#include <raxis/export.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/array_ref.h>


namespace raxis {

// The norms _foreach_norm has a fused kernel for (torch's NormType)
enum class ForeachNormType { L0, L1, L2, LInf };

// out[i] = the norm of tensors[i]. The tensors are contiguous, of one
// floating dtype and on one device; out is a vector of that dtype there.
using foreach_norm_fn = void(*)(ArrayRef<Tensor> tensors, ForeachNormType norm, const Tensor& out);
// tensors[i] *= scalar, with scalar a 0-d tensor of their dtype on their device
using foreach_mul_fn = void(*)(ArrayRef<Tensor> tensors, const Tensor& scalar);

DECLARE_DISPATCH(foreach_norm_fn, foreach_norm_stub);
DECLARE_DISPATCH(foreach_mul_fn, foreach_mul_stub);

// Ops over lists of tensors (torch._foreach_*): the list in a few launches
// instead of one op per tensor. The fused kernels are GPU only and take
// lists of contiguous tensors of one dtype on one device; anything else
// takes the slow path, one op per tensor, as PyTorch's
// foreach_tensor_*_slow. No autograd: they serve optimizers and
// clip_grad_norm_, which run under no_grad.
// From: aten/src/ATen/native/ForeachOpsKernels.cpp, native/cuda/ForeachReduceOp.cu,
// native/cuda/ForeachBinaryOpScalarTensor.cu

/// The ord-norm of every tensor (flattened): 0-d tensors. On the fused
/// path they are views of one vector of norms.
RAXIS_API std::vector<Tensor> foreach_norm(ArrayRef<Tensor> tensors, double ord = 2.0);

/// tensors[i] *= scalar in place, for a 0-d (or one element) tensor scalar
RAXIS_API void foreach_mul_(ArrayRef<Tensor> tensors, const Tensor& scalar);

} // namespace raxis
