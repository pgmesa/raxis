import math
from typing import Iterable

import raxis
from raxis import Tensor


def _group_by_device_and_dtype(tensors: list[Tensor]) -> list[list[Tensor]]:
    """One list per (device, dtype), as torch's _group_tensors_by_device_and_dtype:
    a foreach op takes tensors of one device and dtype."""
    groups = raxis._C._group_indices_by_device_and_dtype(tensors)
    if len(groups) == 1:
        return [tensors]
    return [[tensors[i] for i in group] for group in groups]


def _vector_norm(t: Tensor, norm_type: float) -> Tensor:
    """The p-norm of t flattened. Stands in for torch.linalg.vector_norm
    until raxis has a norm reduction (torch's adds |x|^p in one pass;
    here it takes an elementwise op and a sum)."""
    if norm_type == math.inf:
        return t.abs().amax()
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
                   error_if_nonfinite: bool = False, foreach: bool | None = None) -> Tensor:
    """The norm of all the tensors together, as if they were concatenated into
    one vector: the norm of their norms. With foreach (the default), the norms
    of each (device, dtype) group come from one _foreach_norm (two kernels on
    a GPU for the ord 0, 1, 2 and inf, instead of a few launches per tensor)."""
    tensors = [tensors] if isinstance(tensors, Tensor) else list(tensors)
    norm_type = float(norm_type)
    if not tensors:
        return raxis.tensor(0.0)
    groups = _group_by_device_and_dtype(tensors)
    norms = []
    with raxis.no_grad():
        for group in groups:
            if foreach is None or foreach:
                norms.extend(raxis._C._foreach_norm(group, norm_type))
            else:
                norms.extend(_vector_norm(t, norm_type) for t in group)
        if len(groups) > 1:
            first_device = tensors[0].device
            norms = [n.to(first_device) for n in norms]
        total_norm = _vector_norm(raxis.stack(norms), norm_type)
    if error_if_nonfinite and not math.isfinite(total_norm.item()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`")
    return total_norm


# From: torch/nn/utils/clip_grad.py (_clip_grads_with_norm_)
def clip_grads_with_norm_(parameters: Tensor | Iterable[Tensor], max_norm: float, total_norm: Tensor,
                          foreach: bool | None = None) -> None:
    """Scales the gradients in place by min(max_norm / (total_norm + 1e-6), 1)."""
    parameters = [parameters] if isinstance(parameters, Tensor) else parameters
    _clip_grads_with_norm_([p.grad for p in parameters if p.grad is not None], max_norm, total_norm, foreach)


def _clip_grads_with_norm_(grads: list[Tensor], max_norm: float, total_norm: Tensor, foreach: bool | None) -> None:
    if not grads:
        return
    with raxis.no_grad():
        # Multiplying by a coefficient clamped to 1 instead of branching on
        # it, as torch, keeps the result a tensor (no sync on a GPU)
        clip_coef = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
        for group in _group_by_device_and_dtype(grads):
            coef = clip_coef.to(group[0].device)
            if foreach is None or foreach:
                raxis._C._foreach_mul_(group, coef)
            else:
                for g in group:
                    g.mul_(coef)


# From: torch/nn/utils/clip_grad.py
def clip_grad_norm_(parameters: Tensor | Iterable[Tensor], max_norm: float, norm_type: float = 2.0,
                    error_if_nonfinite: bool = False, foreach: bool | None = None) -> Tensor:
    """Rescales the gradients so that their total norm is at most max_norm.
    Returns the total norm before clipping."""
    parameters = [parameters] if isinstance(parameters, Tensor) else list(parameters)
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = get_total_norm(grads, norm_type, error_if_nonfinite, foreach)
    _clip_grads_with_norm_(grads, max_norm, total_norm, foreach)
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
