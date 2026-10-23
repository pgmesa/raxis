from typing import Optional, Sequence, Union

import raxis
from raxis import _C
from raxis.autograd import graph
from raxis.autograd.function import Function, FunctionCtx, once_differentiable

_TensorOrTensors = Union["raxis.Tensor", Sequence["raxis.Tensor"]]


def _as_tuple(tensors) -> tuple:
    if tensors is None:
        return ()
    if isinstance(tensors, raxis.Tensor):
        return (tensors,)
    return tuple(tensors)


# From: torch/autograd/__init__.py
def backward(
    tensors: _TensorOrTensors,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
) -> None:
    """Accumulates in .grad of the leaves the sum of the gradients of
    ``tensors``, each weighted by its entry in ``grad_tensors`` (None for a
    scalar means 1). ``retain_graph`` defaults to ``create_graph``; with
    ``create_graph`` the backward is recorded, for higher-order derivatives."""
    tensors = _as_tuple(tensors)
    grad_tensors = _as_tuple(grad_tensors)
    _C._autograd_backward(list(tensors), list(grad_tensors), retain_graph, create_graph)


# From: torch/autograd/__init__.py
def grad(
    outputs: _TensorOrTensors,
    inputs: _TensorOrTensors,
    grad_outputs: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    allow_unused: Optional[bool] = None,
    materialize_grads: bool = False,
) -> tuple:
    """Returns the gradients of ``outputs`` with respect to ``inputs``,
    without touching any .grad. Only the part of the graph between them
    runs. With ``create_graph`` the result can be differentiated again
    (Hessian-vector products, gradient penalties).

    An input the outputs do not depend on raises, unless ``allow_unused``
    (it then gets None, or zeros with ``materialize_grads``)."""
    if allow_unused is None:
        allow_unused = materialize_grads
    if materialize_grads and not allow_unused:
        raise ValueError("Expected allow_unused to be True or not passed when materialize_grads=True, "
                         "but got: allow_unused=False.")
    outputs = _as_tuple(outputs)
    inputs = _as_tuple(inputs)
    grad_outputs = _as_tuple(grad_outputs)
    result = _C._autograd_grad(list(outputs), list(inputs), list(grad_outputs),
                               retain_graph, create_graph, allow_unused)
    if materialize_grads:
        result = tuple(raxis.zeros_like(x) if g is None else g for g, x in zip(result, inputs))
    return result
