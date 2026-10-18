"""Functional interface of raxis.nn, mirroring torch.nn.functional.

Most functions call C++ ops (top-level raxis ops or raxis._C._nn, as torch
does with torch._C._nn). leaky_relu and selu are composed in Python.
"""

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


# -- Losses --

def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return _nn.mse_loss(input, target, reduction)


def nll_loss(input: Tensor, target: Tensor, ignore_index: int = -100,
             reduction: str = "mean") -> Tensor:
    return _nn.nll_loss(input, target, reduction, ignore_index)


def cross_entropy(input: Tensor, target: Tensor, ignore_index: int = -100,
                  reduction: str = "mean") -> Tensor:
    return _nn.cross_entropy(input, target, reduction, ignore_index)


def binary_cross_entropy(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    return _nn.binary_cross_entropy(input, target, reduction)


def binary_cross_entropy_with_logits(input: Tensor, target: Tensor,
                                     reduction: str = "mean") -> Tensor:
    return _nn.binary_cross_entropy_with_logits(input, target, reduction)
