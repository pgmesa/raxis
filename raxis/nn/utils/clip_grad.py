import math
from typing import Iterable

import raxis
from raxis import Tensor


def _vector_norm(t: Tensor, norm_type: float) -> Tensor:
    """The p-norm of t flattened. Stands in for torch.linalg.vector_norm
    until raxis has a norm reduction (torch's adds |x|^p in one pass;
    here it takes an elementwise op and a sum)."""
    if norm_type == math.inf:
        return t.abs().amax() if t.numel() else raxis.tensor(0.0, dtype=t.dtype)
    if norm_type == -math.inf:
        return t.abs().amin()
    if norm_type == 0:
        return (t != 0).sum().to(t.dtype)
    if norm_type == 2:
        return (t * t).sum().sqrt()
    if norm_type == 1:
        return t.abs().sum()
    return t.abs().pow(norm_type).sum().pow(1.0 / norm_type)


# From: torch/nn/utils/clip_grad.py (_get_total_norm)
def get_total_norm(tensors: Tensor | Iterable[Tensor], norm_type: float = 2.0,
                   error_if_nonfinite: bool = False) -> Tensor:
    """The norm of all the tensors together, as if they were concatenated into
    one vector: the norm of their norms."""
    tensors = [tensors] if isinstance(tensors, Tensor) else list(tensors)
    norm_type = float(norm_type)
    if not tensors:
        return raxis.tensor(0.0)
    with raxis.no_grad():
        total_norm = _vector_norm(raxis.stack([_vector_norm(t, norm_type) for t in tensors]), norm_type)
    if error_if_nonfinite and not math.isfinite(total_norm.item()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`")
    return total_norm


# From: torch/nn/utils/clip_grad.py (_clip_grads_with_norm_)
def clip_grads_with_norm_(parameters: Tensor | Iterable[Tensor], max_norm: float, total_norm: Tensor) -> None:
    """Scales the gradients in place by min(max_norm / (total_norm + 1e-6), 1)."""
    parameters = [parameters] if isinstance(parameters, Tensor) else parameters
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    with raxis.no_grad():
        # Multiplying by a coefficient clamped to 1 instead of branching on
        # it, as torch, keeps the result a tensor (no sync on a GPU)
        clip_coef = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
        for g in grads:
            g.mul_(clip_coef)


# From: torch/nn/utils/clip_grad.py
def clip_grad_norm_(parameters: Tensor | Iterable[Tensor], max_norm: float, norm_type: float = 2.0,
                    error_if_nonfinite: bool = False) -> Tensor:
    """Rescales the gradients so that their total norm is at most max_norm.
    Returns the total norm before clipping."""
    parameters = [parameters] if isinstance(parameters, Tensor) else list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = get_total_norm(grads, norm_type, error_if_nonfinite)
    clip_grads_with_norm_(parameters, max_norm, total_norm)
    return total_norm


# From: torch/nn/utils/clip_grad.py
def clip_grad_value_(parameters: Tensor | Iterable[Tensor], clip_value: float) -> None:
    """Clamps every gradient element to [-clip_value, clip_value] in place."""
    parameters = [parameters] if isinstance(parameters, Tensor) else parameters
    clip_value = float(clip_value)
    with raxis.no_grad():
        for p in parameters:
            if p.grad is not None:
                p.grad.clamp_(min=-clip_value, max=clip_value)
