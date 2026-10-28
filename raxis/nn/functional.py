"""Functional interface of raxis.nn, mirroring torch.nn.functional.

Most functions call C++ ops (top-level raxis ops or raxis._C._nn, as torch
does with torch._C._nn). leaky_relu and selu are composed in Python.
"""

import math
from typing import Sequence

import raxis
from raxis import Tensor
from raxis._C import _nn


# -- Activations --

def relu(input: Tensor, inplace: bool = False) -> Tensor:
    return raxis.relu_(input) if inplace else raxis.relu(input)


def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    # max(0, x) + slope * min(0, x), with min(0, x) = -relu(-x)
    return raxis.relu(input) - raxis.relu(-input) * negative_slope


SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946


def selu(input: Tensor) -> Tensor:
    # scale * (max(0, x) + alpha * (exp(min(0, x)) - 1)), with min(0, x) = -relu(-x)
    negative = (raxis.exp(-raxis.relu(-input)) - 1.0) * SELU_ALPHA
    return (raxis.relu(input) + negative) * SELU_SCALE


def gelu(input: Tensor, approximate: str = "none") -> Tensor:
    return _nn.gelu(input, approximate=approximate)


def sigmoid(input: Tensor) -> Tensor:
    return raxis.sigmoid(input)


def tanh(input: Tensor) -> Tensor:
    return raxis.tanh(input)


def softmax(input: Tensor, dim: int) -> Tensor:
    return raxis.softmax(input, dim)


def log_softmax(input: Tensor, dim: int) -> Tensor:
    return raxis.log_softmax(input, dim)


# -- Layers --

def linear(input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    return _nn.linear(input, weight, bias)


# From: torch/nn/functional.py (embedding)
def embedding(input: Tensor, weight: Tensor, padding_idx: int | None = None, max_norm: float | None = None,
              norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False) -> Tensor:
    if padding_idx is not None:
        if padding_idx > 0:
            assert padding_idx < weight.size(0), "Padding_idx must be within num_embeddings"
        elif padding_idx < 0:
            assert padding_idx >= -weight.size(0), "Padding_idx must be within num_embeddings"
            padding_idx = weight.size(0) + padding_idx
    else:
        padding_idx = -1
    if max_norm is not None:
        raise NotImplementedError("embedding: max_norm is not supported yet")
    if sparse:
        raise NotImplementedError("embedding: raxis has no sparse gradients")
    return raxis.embedding(weight, input, padding_idx, scale_grad_by_freq)


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return raxis.dropout(input, p, training)


# -- Attention --

# From: aten/src/ATen/native/transformers/attention.cpp (_safe_softmax)
def _safe_softmax(input: Tensor, dim: int) -> Tensor:
    """softmax, but a row that is all -inf (every position masked) gives
    zeros instead of NaN."""
    out = raxis.softmax(input, dim)
    masked_rows = (input == -math.inf).all(dim, keepdim=True)
    return raxis.where(masked_rows, 0.0, out)


# From: aten/src/ATen/native/transformers/sdp_utils_cpp.cpp (use_flash_attention_cpp)
def _use_flash_attention_cpp(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None,
                             dropout_p: float) -> bool:
    """Whether the fused CPU kernel takes these inputs (torch's checks, in
    its order); otherwise the math composition runs."""
    if query.is_cuda:
        return False
    if dropout_p > 0.0:
        return False
    if not query.ndim == key.ndim == value.ndim == 4:
        return False
    if not (query.shape[0] == key.shape[0] == value.shape[0]
            and query.shape[1] == key.shape[1] == value.shape[1]):
        return False
    L, S = query.shape[2], key.shape[2]
    if attn_mask is not None:
        # The kernel gives the mask no gradient
        if attn_mask.requires_grad or attn_mask.ndim not in (2, 4):
            return False
        if attn_mask.shape[-2] not in (L, 1) or attn_mask.shape[-1] not in (S, 1):
            return False
        if attn_mask.ndim == 4 and (attn_mask.shape[0] not in (query.shape[0], 1)
                                    or attn_mask.shape[1] not in (query.shape[1], 1)):
            return False
    if not query.shape[-1] == key.shape[-1] == value.shape[-1]:
        return False
    if L == 0 or S == 0:
        return False
    if not query.strides[-1] == key.strides[-1] == value.strides[-1] == 1:
        return False
    return query.dtype in (raxis.float32, raxis.float64) and key.dtype == value.dtype == query.dtype


# From: aten/src/ATen/native/transformers/cuda/sdp_utils.cpp (use_mem_efficient_attention)
def _use_mem_efficient_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None) -> bool:
    """Whether the fused GPU kernel takes these inputs. torch tries its
    FlashAttention-2 kernel first, but that one only takes fp16 / bf16, so
    float32 lands here (or on the math path)."""
    if not query.is_cuda:
        return False
    if not query.ndim == key.ndim == value.ndim == 4:
        return False
    if not (query.shape[0] == key.shape[0] == value.shape[0]
            and query.shape[1] == key.shape[1] == value.shape[1]):
        return False
    if not query.dtype == key.dtype == value.dtype == raxis.float32:
        return False
    # raxis's kernel keeps head dims up to 64 in its tiles (torch's goes further)
    if not query.shape[-1] == key.shape[-1] == value.shape[-1] <= 64:
        return False
    if query.shape[2] == 0 or key.shape[2] == 0:
        return False
    if attn_mask is not None:
        # The kernel gives the bias no gradient
        if attn_mask.requires_grad or attn_mask.ndim not in (2, 4):
            return False
        if attn_mask.shape[-2] not in (query.shape[2], 1) or attn_mask.shape[-1] not in (key.shape[2], 1):
            return False
        if attn_mask.ndim == 4 and (attn_mask.shape[0] not in (query.shape[0], 1)
                                    or attn_mask.shape[1] not in (query.shape[1], 1)):
            return False
    # Its tiles take up to 85 KB of shared memory per block
    return raxis.cuda.get_device_capability(query.device) >= (8, 0)


# From: aten/src/ATen/native/transformers/attention.cpp (validate_sdpa_input)
def _validate_sdpa_input(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None,
                         is_causal: bool) -> None:
    if not query.dtype == key.dtype == value.dtype:
        raise RuntimeError(
            f"Expected query, key, and value to have the same dtype, but got query.dtype: {query.dtype} "
            f"key.dtype: {key.dtype} and value.dtype: {value.dtype} instead.")
    if query.ndim < 2 or key.ndim < 2 or value.ndim < 2:
        raise RuntimeError(
            f"Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: "
            f"{query.ndim} key.dim: {key.ndim} and value.dim: {value.ndim} instead.")
    if attn_mask is not None:
        if is_causal:
            raise RuntimeError(
                "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
        if attn_mask.dtype not in (raxis.bool, raxis.float32, query.dtype):
            raise RuntimeError(
                f"Expected attn_mask dtype to be bool or float or to match query dtype, but got "
                f"attn_mask.dtype: {attn_mask.dtype} and  query.dtype: {query.dtype} instead.")


# From: aten/src/ATen/native/transformers/attention.cpp (scaled_dot_product_attention,
# _scaled_dot_product_attention_math)
def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None,
                                 dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None,
                                 enable_gqa: bool = False) -> Tensor:
    """softmax(query @ key^T * scale + mask) @ value over the last two dims:
    query (..., L, E), key (..., S, E), value (..., S, Ev) -> (..., L, Ev).

    attn_mask is a bool mask (True takes part) or an additive float mask;
    is_causal masks the positions after each query (a lower triangular
    mask). scale defaults to 1 / sqrt(E).

    As in torch, 4-D float inputs without dropout run the fused flash
    attention kernel on the CPU (blocks of keys with an online softmax, the
    (L, S) scores are never stored; grad_fn
    ScaledDotProductFlashAttentionForCpuBackward0), and 4-D float32 inputs on
    the GPU the fused kernel torch calls memory-efficient attention, dropout
    included (grad_fn ScaledDotProductEfficientAttentionBackward0). The rest
    goes through torch's math backend, a composition of ops that autograd
    differentiates on its own. They give the same result up to rounding
    (dropout draws different masks on each path).
    """
    if enable_gqa:
        raise NotImplementedError("scaled_dot_product_attention: enable_gqa is not supported yet")
    # Unbatched inputs get a batch dim, so they can take the fused kernels too
    if query.ndim == key.ndim == value.ndim == 3:
        while attn_mask is not None and attn_mask.ndim < 4:
            attn_mask = attn_mask.unsqueeze(0)
        return scaled_dot_product_attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), attn_mask,
                                            dropout_p, is_causal, scale).squeeze(0)
    _validate_sdpa_input(query, key, value, attn_mask, is_causal)
    if _use_flash_attention_cpp(query, key, value, attn_mask, dropout_p):
        return raxis._C._scaled_dot_product_flash_attention_for_cpu(
            query, key, value, dropout_p, is_causal, attn_mask=attn_mask, scale=scale)[0]
    if _use_mem_efficient_attention(query, key, value, attn_mask):
        compute_log_sumexp = query.requires_grad or key.requires_grad or value.requires_grad
        return raxis._C._scaled_dot_product_efficient_attention(
            query, key, value, attn_mask, compute_log_sumexp, dropout_p, is_causal, scale=scale)[0]
    if scale is None:
        scale = 1.0 / math.sqrt(query.size(-1))
    # q and k are each scaled by sqrt(scale) before the matmul, which keeps
    # the products in range for low precision dtypes
    scaling_factor = math.sqrt(abs(scale))
    q = query * (-scaling_factor if scale < 0 else scaling_factor)
    if is_causal:
        attn_mask = raxis.ones((query.size(-2), key.size(-2)), dtype=raxis.bool, device=query.device).tril()
    attn = q.matmul(key.transpose(-2, -1) * scaling_factor)
    if attn_mask is not None:
        if attn_mask.dtype == raxis.bool:
            attn = attn.masked_fill(attn_mask.logical_not(), -math.inf)
        else:
            attn = attn + attn_mask
    attn = _safe_softmax(attn, -1)
    if dropout_p > 0.0:
        attn = raxis.dropout(attn, dropout_p, True)
    return attn.matmul(value)


# -- Convolution and pooling --

def _expand(value: int | Sequence[int], n: int) -> list[int]:
    return list(value) if isinstance(value, (tuple, list)) else [value] * n


# From: aten/src/ATen/native/Convolution.cpp (_convolution_mode, convolution_same)
def _conv_padding_from_string(input: Tensor, padding: str, weight: Tensor, stride,
                              dilation) -> tuple[Tensor, list[int]]:
    """The input (padded when needed) and the symmetric padding for the
    convolution that the "valid" / "same" padding strings mean."""
    spatial_dims = weight.dim() - 2
    if padding == "valid":
        return input, [0] * spatial_dims
    if padding != "same":
        raise RuntimeError(f"Invalid padding string: '{padding}'")
    if any(s != 1 for s in _expand(stride, spatial_dims)):
        raise RuntimeError("padding='same' is not supported for strided convolutions")
    # The output keeps the input size when the d * (k - 1) positions the
    # kernel overhangs are split between both sides. With an odd total the
    # right side gets one more: the input is padded on the right by the
    # difference and the convolution pads both sides by the left amount.
    pads_l = []
    extra_r = [0] * (2 * spatial_dims)
    for i, (d, k) in enumerate(zip(_expand(dilation, spatial_dims), weight.shape[2:])):
        total = d * (k - 1)
        pads_l.append(total // 2)
        extra_r[2 * (spatial_dims - 1 - i) + 1] = total - 2 * (total // 2)
    if any(extra_r):
        input = pad(input, extra_r)
    return input, pads_l


def conv1d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1,
           padding: int | Sequence[int] | str = 0, dilation=1, groups: int = 1) -> Tensor:
    if isinstance(padding, str):
        input, padding = _conv_padding_from_string(input, padding, weight, stride, dilation)
    return raxis.conv1d(input, weight, bias, stride, padding, dilation, groups)


def conv2d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1,
           padding: int | Sequence[int] | str = 0, dilation=1, groups: int = 1) -> Tensor:
    if isinstance(padding, str):
        input, padding = _conv_padding_from_string(input, padding, weight, stride, dilation)
    return raxis.conv2d(input, weight, bias, stride, padding, dilation, groups)


def max_pool2d(input: Tensor, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode: bool = False, return_indices: bool = False):
    if return_indices:
        return raxis.max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)
    return raxis.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)


def avg_pool2d(input: Tensor, kernel_size, stride=None, padding=0, ceil_mode: bool = False,
               count_include_pad: bool = True, divisor_override: int | None = None) -> Tensor:
    return raxis.avg_pool2d(input, kernel_size, stride, padding, ceil_mode,
                            count_include_pad, divisor_override)


# -- Padding --

def pad(input: Tensor, pad: Sequence[int], mode: str = "constant", value: float | None = None) -> Tensor:
    """Pads input with a (left, right) pair per dim, starting from the last
    dim. mode is "constant" (with value, 0 by default), "reflect",
    "replicate" or "circular". Negative pads crop."""
    return _nn.pad(input, list(pad), mode, value)


# -- Normalization --

# From: torch/nn/functional.py (_verify_batch_size, batch_norm)
def batch_norm(input: Tensor, running_mean: Tensor | None, running_var: Tensor | None,
               weight: Tensor | None = None, bias: Tensor | None = None, training: bool = False,
               momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    if training:
        # A single value per channel has zero variance: nothing to normalize
        if math.prod(input.shape[2:]) * input.shape[0] == 1:
            raise ValueError(
                f"Expected more than 1 value per channel when training, got input size {input.shape}")
        if eps <= 0.0:
            raise ValueError(f"batch_norm eps must be positive during training, but got {eps}")
    elif eps < 0.0:
        raise ValueError(f"batch_norm eps must be non-negative, but got {eps}")
    return raxis.batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps)


def layer_norm(input: Tensor, normalized_shape, weight: Tensor | None = None,
               bias: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    return raxis.layer_norm(input, normalized_shape, weight, bias, eps)


# -- Losses --

def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return _nn.mse_loss(input, target, reduction)


def nll_loss(input: Tensor, target: Tensor, weight: Tensor | None = None, *,
             ignore_index: int = -100, reduction: str = "mean") -> Tensor:
    return _nn.nll_loss(input, target, weight, reduction, ignore_index)


def cross_entropy(input: Tensor, target: Tensor, weight: Tensor | None = None, *,
                  ignore_index: int = -100, reduction: str = "mean") -> Tensor:
    return _nn.cross_entropy(input, target, weight, reduction, ignore_index)


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return _nn.binary_cross_entropy(input, target, reduction)


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor,
                                     reduction: str = "mean") -> Tensor:
    return _nn.binary_cross_entropy_with_logits(input, target, reduction)
