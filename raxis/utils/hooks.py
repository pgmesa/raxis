import weakref
from typing import Any, Callable

import raxis


# From: torch/utils/hooks.py
class RemovableHandle:
    """Returned by the register_*_hook methods: remove() unregisters the hook.

    Holds the hooks dict weakly, so a handle kept around does not keep a
    deleted module alive.
    """

    next_id: int = 0

    def __init__(self, hooks_dict: dict) -> None:
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, *args: Any) -> None:
        self.remove()


class BackwardHookFunction(raxis.autograd.Function):
    """Identity on tensors: its node in the graph is where Module backward
    hooks see the gradients of the module inputs and outputs.

    From: torch/nn/modules/_functions.py
    """

    @staticmethod
    def forward(ctx, *args):
        ctx.mark_non_differentiable(*[arg for arg in args if not arg.requires_grad])
        return args

    @staticmethod
    def backward(ctx, *args):
        return args


# From: torch/utils/hooks.py
class BackwardHook:
    """Implements Module.register_full_backward_hook / _pre_hook.

    The tensor inputs and outputs of the module go through a
    BackwardHookFunction each. A post hook on the output node captures the
    gradients of the outputs (and runs the pre hooks, which may change
    them); a post hook on the input node runs the user hooks with the
    gradients of the inputs and of the outputs. Non-tensor arguments show up
    as None.
    """

    def __init__(self, module, user_hooks: list, user_pre_hooks: list) -> None:
        self.user_hooks = user_hooks
        self.user_pre_hooks = user_pre_hooks
        self.module = module

        self.grad_outputs = None
        self.n_outputs = -1
        self.output_tensors_index = None
        self.n_inputs = -1
        self.input_tensors_index = None

    def _pack_with_none(self, indices, values, size):
        res = [None] * size
        for idx, val in zip(indices, values, strict=True):
            res[idx] = val
        return tuple(res)

    def _unpack_none(self, indices, values):
        return tuple(values[idx] for idx in indices)

    def _set_user_hook(self, grad_fn) -> None:
        def hook(grad_input, _):
            if self.grad_outputs is None:
                # The gradient reached the inputs without going through the
                # outputs of the module
                return None
            res = self._pack_with_none(self.input_tensors_index, grad_input, self.n_inputs)
            for user_hook in self.user_hooks:
                out = user_hook(self.module, res, self.grad_outputs)
                if out is None:
                    continue
                if len(out) != len(res):
                    raise RuntimeError("Backward hook returned an invalid number of grad_input, "
                                       f"got {len(out)}, but expected {len(res)}")
                res = out
            self.grad_outputs = None
            return self._unpack_none(self.input_tensors_index, res)

        grad_fn.register_hook(hook)

    def _apply_on_tensors(self, fn: Callable, args: tuple):
        tensors_idx = [i for i, arg in enumerate(args) if isinstance(arg, raxis.Tensor)]
        tensors = [args[i] for i in tensors_idx]
        requires_grad = any(t.requires_grad for t in tensors)
        if not (requires_grad and raxis.is_grad_enabled()):
            return args, None

        new_tensors = BackwardHookFunction.apply(*tensors)
        if len(new_tensors) == 0:
            raise RuntimeError("Cannot set Module backward hook for a Module with no input Tensors.")
        grad_fns = [t.grad_fn for t in new_tensors
                    if t.grad_fn is not None and t.grad_fn.name() == "BackwardHookFunctionBackward"]
        fn(grad_fns[0])

        arg_list = list(args)
        for idx, val in zip(tensors_idx, new_tensors, strict=True):
            arg_list[idx] = val
        return tuple(arg_list), tensors_idx

    def setup_input_hook(self, args: tuple) -> tuple:
        res, input_idx = self._apply_on_tensors(self._set_user_hook, args)
        self.n_inputs = len(args)
        self.input_tensors_index = input_idx
        return res

    def setup_output_hook(self, args):
        def fn(grad_fn) -> None:
            def hook(_, grad_output):
                self.grad_outputs = self._pack_with_none(
                    self.output_tensors_index, grad_output, self.n_outputs)
                for user_pre_hook in self.user_pre_hooks:
                    hook_grad_outputs = user_pre_hook(self.module, self.grad_outputs)
                    if hook_grad_outputs is None:
                        continue
                    if len(hook_grad_outputs) != len(self.grad_outputs):
                        raise RuntimeError("Backward pre hook returned an invalid number of grad_output, "
                                           f"got {len(hook_grad_outputs)}, but expected {len(self.grad_outputs)}")
                    self.grad_outputs = hook_grad_outputs
                local_grad_outputs = self.grad_outputs

                # No input requires grad: the input hook never runs, so the
                # user hooks run here with no input gradients
                if self.input_tensors_index is None:
                    grad_inputs = self._pack_with_none([], [], self.n_inputs)
                    for user_hook in self.user_hooks:
                        res = user_hook(self.module, grad_inputs, self.grad_outputs)
                        if res is not None and not (isinstance(res, tuple) and all(el is None for el in res)):
                            raise RuntimeError("Backward hook for Modules where no input requires "
                                               "gradient should always return None or None for all gradients.")
                    self.grad_outputs = None

                return tuple(local_grad_outputs[i] for i in self.output_tensors_index)

            grad_fn.register_hook(hook)

        is_tuple = isinstance(args, tuple)
        if not is_tuple:
            args = (args,)
        res, output_idx = self._apply_on_tensors(fn, args)
        self.n_outputs = len(args)
        self.output_tensors_index = output_idx
        return res if is_tuple else res[0]
