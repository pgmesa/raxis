from collections import OrderedDict, namedtuple
from typing import Any, Callable, Iterator, Mapping

import raxis
from raxis import Tensor
from raxis.utils.hooks import BackwardHook, RemovableHandle


class Parameter(Tensor):
    """A Tensor that is registered as a learnable parameter when assigned to
    a Module attribute. It aliases the data of the tensor it wraps."""

    def __init__(self, data: Tensor, requires_grad: bool = True):
        super().__init__(data)
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"


# From: torch/nn/modules/module.py
class _IncompatibleKeys(namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])):
    """Returned by Module.load_state_dict."""

    __slots__ = ()

    def __repr__(self) -> str:
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()


class Module:
    """Base class for all neural network modules.

    Submodules and parameters assigned as attributes are registered
    automatically, so parameters() and train()/eval() reach the whole tree.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_non_persistent_buffers_set", set())
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "_forward_pre_hooks", OrderedDict())
        object.__setattr__(self, "_forward_hooks", OrderedDict())
        object.__setattr__(self, "_backward_pre_hooks", OrderedDict())
        object.__setattr__(self, "_backward_hooks", OrderedDict())

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required 'forward' function")

    # From: torch/nn/modules/module.py (Module._call_impl)
    def __call__(self, *args, **kwargs) -> Tensor:
        if not (self._forward_pre_hooks or self._forward_hooks
                or self._backward_pre_hooks or self._backward_hooks):
            return self.forward(*args, **kwargs)

        for hook in tuple(self._forward_pre_hooks.values()):
            args_result = hook(self, args)
            if args_result is not None:
                args = args_result if isinstance(args_result, tuple) else (args_result,)

        bw_hook = None
        if self._backward_hooks or self._backward_pre_hooks:
            bw_hook = BackwardHook(self, list(self._backward_hooks.values()),
                                   list(self._backward_pre_hooks.values()))
            args = bw_hook.setup_input_hook(args)

        result = self.forward(*args, **kwargs)
        for hook in tuple(self._forward_hooks.values()):
            hook_result = hook(self, args, result)
            if hook_result is not None:
                result = hook_result

        if bw_hook:
            result = bw_hook.setup_output_hook(result)
        return result

    # -- Hooks --

    def register_forward_pre_hook(self, hook: Callable) -> RemovableHandle:
        """hook(module, args) runs before forward. Returning something other
        than None replaces the positional args (a single value is wrapped in a
        tuple)."""
        handle = RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        return handle

    def register_forward_hook(self, hook: Callable) -> RemovableHandle:
        """hook(module, args, output) runs after forward. Returning something
        other than None replaces the output."""
        handle = RemovableHandle(self._forward_hooks)
        self._forward_hooks[handle.id] = hook
        return handle

    def register_full_backward_pre_hook(self, hook: Callable, prepend: bool = False) -> RemovableHandle:
        """hook(module, grad_output) runs when the gradients of the outputs are
        computed. Returning something other than None replaces them (one entry
        per output, None for non-tensor outputs)."""
        handle = RemovableHandle(self._backward_pre_hooks)
        self._backward_pre_hooks[handle.id] = hook
        if prepend:
            self._backward_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_full_backward_hook(self, hook: Callable, prepend: bool = False) -> RemovableHandle:
        """hook(module, grad_input, grad_output) runs when the gradients of the
        inputs are computed. Returning something other than None replaces
        grad_input. The module's inputs and outputs become views (made by a
        BackwardHookFunction), so they cannot be modified in place."""
        handle = RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        if prepend:
            self._backward_hooks.move_to_end(handle.id, last=False)
        return handle

    def __setattr__(self, name: str, value: Any) -> None:
        if "_modules" not in self.__dict__:
            raise RuntimeError("cannot assign attributes before Module.__init__() call")
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
        elif name in self._buffers:
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"cannot assign '{type(value).__name__}' as buffer '{name}'")
            self._buffers[name] = value
            object.__setattr__(self, name, value)
        else:
            # Reassigning a registered name to something else unregisters it
            self._parameters.pop(name, None)
            self._modules.pop(name, None)
            object.__setattr__(self, name, value)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"cannot assign '{type(param).__name__}' as parameter '{name}'")
        self._parameters[name] = param
        object.__setattr__(self, name, param)

    def register_buffer(self, name: str, tensor: Tensor | None, persistent: bool = True) -> None:
        """A tensor that is part of the module's state but is not trained,
        such as the running mean of BatchNorm. Persistent buffers are saved
        in the state_dict."""
        if tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(f"cannot assign '{type(tensor).__name__}' as buffer '{name}'")
        self._buffers[name] = tensor
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)
        object.__setattr__(self, name, tensor)

    def register_module(self, name: str, module: "Module | None") -> None:
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"{type(module).__name__} is not a Module subclass")
        self._modules[name] = module
        object.__setattr__(self, name, module)

    # -- Traversal --

    def children(self) -> Iterator["Module"]:
        for module in self._modules.values():
            if module is not None:
                yield module

    def named_modules(self, prefix: str = "") -> Iterator[tuple[str, "Module"]]:
        yield prefix, self
        for name, module in self._modules.items():
            if module is not None:
                yield from module.named_modules(f"{prefix}.{name}" if prefix else name)

    def modules(self) -> Iterator["Module"]:
        for _, module in self.named_modules():
            yield module

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        for module_name, module in self.named_modules(prefix):
            for name, param in module._parameters.items():
                if param is not None:
                    yield (f"{module_name}.{name}" if module_name else name), param

    def parameters(self) -> Iterator[Parameter]:
        for _, param in self.named_parameters():
            yield param

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Tensor]]:
        for module_name, module in self.named_modules(prefix):
            for name, buf in module._buffers.items():
                if buf is not None:
                    yield (f"{module_name}.{name}" if module_name else name), buf

    def buffers(self) -> Iterator[Tensor]:
        for _, buf in self.named_buffers():
            yield buf

    def apply(self, fn: Callable[["Module"], None]) -> "Module":
        """Applies fn to every submodule (children first) and then to self,
        typically to initialize parameters."""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    # -- State dict --

    def _local_state(self) -> dict[str, Tensor]:
        state = {name: p for name, p in self._parameters.items() if p is not None}
        for name, buf in self._buffers.items():
            if buf is not None and name not in self._non_persistent_buffers_set:
                state[name] = buf
        return state

    # From: torch/nn/modules/module.py (Module.state_dict)
    def state_dict(self, destination: dict | None = None, prefix: str = "",
                   keep_vars: bool = False) -> "OrderedDict[str, Tensor]":
        """Parameters and persistent buffers by their dotted names
        ("0.weight", "body.fc.bias", ...).

        The values are detached tensors that share memory with the module, so
        the dict follows later updates; clone it to keep a snapshot.
        keep_vars=True returns the parameters themselves.
        """
        if destination is None:
            destination = OrderedDict()
        for name, value in self._local_state().items():
            destination[prefix + name] = value if keep_vars else value.detach()
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + ".", keep_vars)
        return destination

    # From: torch/nn/modules/module.py (Module.load_state_dict, _load_from_state_dict)
    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> _IncompatibleKeys:
        """Copies the tensors of state_dict into the parameters and buffers
        with the same names.

        With strict=True, a missing or unexpected key raises; otherwise they
        are returned. A shape mismatch always raises.
        """
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        error_msgs: list[str] = []

        def load(module, prefix):
            local_state = module._local_state()
            for name, param in local_state.items():
                key = prefix + name
                if key not in state_dict:
                    missing_keys.append(key)
                    continue
                input_param = state_dict[key]
                if not isinstance(input_param, Tensor):
                    error_msgs.append(
                        f'While copying the parameter named "{key}", expected Tensor '
                        f"from checkpoint but received {type(input_param)}")
                    continue
                if tuple(input_param.shape) != tuple(param.shape):
                    error_msgs.append(
                        f"size mismatch for {key}: copying a param with shape "
                        f"{tuple(input_param.shape)} from checkpoint, the shape in current "
                        f"model is {tuple(param.shape)}.")
                    continue
                with raxis.no_grad():
                    param.copy_(input_param)

            for key in state_dict:
                if key.startswith(prefix):
                    input_name = key[len(prefix):].split(".", 1)[0]
                    if input_name not in module._modules and input_name not in local_state:
                        unexpected_keys.append(key)

            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(self, "")

        if strict:
            if unexpected_keys:
                error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(
                    ", ".join(f'"{k}"' for k in unexpected_keys)))
            if missing_keys:
                error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(
                    ", ".join(f'"{k}"' for k in missing_keys)))
        if error_msgs:
            raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(
                type(self).__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    # -- Modes and gradients --

    def train(self, mode: bool = True) -> "Module":
        for module in self.modules():
            object.__setattr__(module, "training", mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.parameters():
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    def requires_grad_(self, requires_grad: bool = True) -> "Module":
        for p in self.parameters():
            p.requires_grad = requires_grad
        return self

    def to(self, device) -> "Module":
        if not str(device).startswith("cpu"):
            raise NotImplementedError(f"raxis only supports the CPU for now, got device '{device}'")
        return self

    def cpu(self) -> "Module":
        return self.to("cpu")

    # -- Representation --

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = []
        for name, module in self._modules.items():
            child = repr(module).replace("\n", "\n  ")
            lines.append(f"({name}): {child}")
        extra = self.extra_repr()
        if not lines:
            return f"{type(self).__name__}({extra})"
        body = "\n  ".join(([extra] if extra else []) + lines)
        return f"{type(self).__name__}(\n  {body}\n)"


class Sequential(Module):
    """Chains modules: the output of each one is the input of the next."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        if len(modules) == 1 and isinstance(modules[0], OrderedDict):
            for name, module in modules[0].items():
                self.register_module(name, module)
        else:
            for idx, module in enumerate(modules):
                self.register_module(str(idx), module)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x
