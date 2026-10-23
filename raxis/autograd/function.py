"""Custom differentiable operations written in Python.

From: torch/autograd/function.py
"""

import functools
from typing import Any, Callable

import raxis
from raxis import _C


class FunctionCtx:
    """The ``ctx`` passed to ``forward`` / ``setup_context`` and ``backward``.

    It carries what forward hands to backward: tensors saved with
    :meth:`save_for_backward` (read back as :attr:`saved_tensors`), any
    other attribute set on it, and what the forward marked (dirty and
    non-differentiable outputs). In PyTorch it is the backward node itself;
    here the node (a C++ PyNode) keeps it alive until the graph is freed.

    Store tensors with :meth:`save_for_backward`, not as attributes: an
    output stored on ctx owns the node through its grad_fn while the node
    owns ctx, and that cycle is never freed.
    """

    # Grads of outputs that got no gradient arrive as zeros (False: None)
    materialize_grads: bool = True
    # Filled by the node during backward only
    _saved_tensors = None

    def save_for_backward(self, *tensors: Any) -> None:
        """Saves tensors (or None) for backward. Inputs, outputs and
        intermediates are all allowed; an in-place change afterwards is
        detected when backward reads them."""
        self.to_save = tensors

    @property
    def saved_tensors(self) -> tuple:
        if self._saved_tensors is None:
            raise RuntimeError("saved_tensors can only be read inside backward")
        return self._saved_tensors

    def mark_dirty(self, *args: Any) -> None:
        """Marks inputs that forward modified in place. They must be returned
        as outputs: their history is rebased on this function."""
        self.dirty_tensors = args

    def mark_non_differentiable(self, *args: Any) -> None:
        """Marks outputs that do not require grad (indices, masks). Their
        gradient in backward is still passed (as zeros or None)."""
        self.non_differentiable = args

    def set_materialize_grads(self, value: bool) -> None:
        """With False, outputs that got no gradient arrive as None instead of
        zeros, which saves the allocation when backward can skip them."""
        self.materialize_grads = value


class Function:
    """Base class for custom autograd operations.

    Subclass it with static ``forward`` and ``backward`` methods and call
    ``apply``::

        class Exp(raxis.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                result = x.exp()
                ctx.save_for_backward(result)
                return result

            @staticmethod
            def backward(ctx, grad):
                result, = ctx.saved_tensors
                return grad * result

        y = Exp.apply(x)

    ``forward`` runs without recording a graph; ``apply`` makes its outputs
    differentiable through ``backward``, which returns one gradient per
    input of forward (None for inputs that are not tensors or need none,
    see ``ctx.needs_input_grad``). Alternatively ``forward(*args)`` can take
    no ctx and a static ``setup_context(ctx, inputs, output)`` fills it.
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{type(self).__name__} should not be instantiated: call "
            f"{type(self).__name__}.apply(...) instead")

    @staticmethod
    def forward(ctx: FunctionCtx, *args: Any) -> Any:
        raise NotImplementedError("You must implement the forward function for custom autograd.Function.")

    @staticmethod
    def setup_context(ctx: FunctionCtx, inputs: tuple, output: Any) -> None:
        raise NotImplementedError("setup_context is not implemented.")

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: Any) -> Any:
        raise NotImplementedError("You must implement the backward function for custom autograd.Function.")

    # From: torch/csrc/autograd/python_function.cpp (THPFunction_apply)
    @classmethod
    def apply(cls, *args: Any) -> Any:
        ctx = FunctionCtx()
        ctx.needs_input_grad = tuple(isinstance(a, raxis.Tensor) and a.requires_grad for a in args)
        is_executable = raxis.is_grad_enabled() and any(ctx.needs_input_grad)

        with raxis.no_grad():
            if cls.setup_context is not Function.setup_context:
                outputs = cls.forward(*args)
                cls.setup_context(ctx, args, outputs)
            else:
                outputs = cls.forward(ctx, *args)

        return _C._wrap_function_outputs(
            ctx, cls.backward, cls.__name__ + "Backward", args, outputs, is_executable)


def once_differentiable(fn: Callable) -> Callable:
    """Decorator for a backward that is not differentiable itself: it runs
    without recording a graph. Under create_graph its results get a grad_fn
    that raises, so a second derivative through it fails instead of
    treating it as a constant."""

    @functools.wraps(fn)
    def wrapper(ctx, *args):
        with raxis.no_grad():
            outputs = fn(ctx, *args)

        if not raxis.is_grad_enabled():
            return outputs
        if not any(isinstance(arg, raxis.Tensor) and arg.requires_grad for arg in args):
            return outputs

        is_tuple = isinstance(outputs, tuple)
        outputs_ = outputs if is_tuple else (outputs,)
        # The node needs inputs that require grad to be part of the graph
        fakes = [None if out is None else out.detach().requires_grad_() for out in outputs_]
        errors = iter(_C._delayed_error(
            "trying to differentiate twice a function that was marked with @once_differentiable",
            [f for f in fakes if f is not None]))
        wrapped = tuple(None if f is None else next(errors) for f in fakes)
        return wrapped if is_tuple else wrapped[0]

    return wrapper
