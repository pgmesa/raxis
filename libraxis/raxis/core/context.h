#pragma once

#include <raxis/export.h>


// Process-wide switches that change how kernels compute, set from Python
// (raxis.backends). From: aten/src/ATen/Context.h (allowTF32CuBLAS,
// allowTF32CuDNN)

namespace raxis {

/// TF32 (float32 inputs rounded to a 10-bit mantissa, products accumulated
/// in float32) on the tensor cores of Ampere and newer GPUs for float32
/// matmuls: mm, addmm, bmm, matmul, linear. Off by default, as
/// torch.backends.cuda.matmul.allow_tf32: since PyTorch 1.12 a float32
/// matmul gives float32 precision unless asked otherwise.
RAXIS_API bool allow_tf32_matmul();
RAXIS_API void set_allow_tf32_matmul(bool allow);

/// The same for the gemms inside a float32 convolution. On by default, as
/// torch.backends.cudnn.allow_tf32: training noise dwarfs the rounding, and
/// convolutions are where the speed-up pays. raxis has no cuDNN; the flag
/// keeps torch's meaning (convolutions) for its own im2col + cuBLAS path.
RAXIS_API bool allow_tf32_conv();
RAXIS_API void set_allow_tf32_conv(bool allow);

} // namespace raxis
