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


def dropout(input: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    return raxis.dropout(input, p, training)


# -- Convolution and pooling --

def _expand(value: int | Sequence[int], n: int) -> list[int]:
    return list(value) if isinstance(value, (tuple, list)) else [value] * n


# From: aten/src/ATen/native/Convolution.cpp (_convolution_mode, convolution_same)
def _padding_from_string(padding: str, weight: Tensor, stride, dilation) -> list[int]:
    spatial_dims = weight.dim() - 2
    if padding == "valid":
        return [0] * spatial_dims
    if padding != "same":
        raise RuntimeError(f"Invalid padding string: '{padding}'")
    if any(s != 1 for s in _expand(stride, spatial_dims)):
        raise RuntimeError("padding='same' is not supported for strided convolutions")
    # The output keeps the input size when the d * (k - 1) positions the
    # kernel overhangs are split between both sides
    pads = []
    for d, k in zip(_expand(dilation, spatial_dims), weight.shape[2:]):
        total = d * (k - 1)
        if total % 2 != 0:
            raise NotImplementedError(
                "padding='same' with an even kernel size needs more padding on one side "
                "than on the other, which raxis does not support yet")
        pads.append(total // 2)
    return pads


def conv1d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1,
           padding: int | Sequence[int] | str = 0, dilation=1, groups: int = 1) -> Tensor:
    if isinstance(padding, str):
        padding = _padding_from_string(padding, weight, stride, dilation)
    return raxis.conv1d(input, weight, bias, stride, padding, dilation, groups)


def conv2d(input: Tensor, weight: Tensor, bias: Tensor | None = None, stride=1,
           padding: int | Sequence[int] | str = 0, dilation=1, groups: int = 1) -> Tensor:
    if isinstance(padding, str):
        padding = _padding_from_string(padding, weight, stride, dilation)
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
