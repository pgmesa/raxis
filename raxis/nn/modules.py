from collections import OrderedDict
from typing import Any, Callable, Iterator

import raxis
from raxis import Tensor
from raxis.utils.hooks import RemovableHandle


class Parameter(Tensor):
    """A Tensor that is registered as a learnable parameter when assigned to
    a Module attribute. It aliases the data of the tensor it wraps."""

    def __init__(self, data: Tensor, requires_grad: bool = True):
        super().__init__(data)
        self.requires_grad = requires_grad

    def __repr__(self) -> str:
        return f"Parameter containing:\n{super().__repr__()}"


class Module:
    """Base class for all neural network modules.

    Submodules and parameters assigned as attributes are registered
    automatically, so parameters() and train()/eval() reach the whole tree.
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "training", True)
        object.__setattr__(self, "_forward_pre_hooks", OrderedDict())
        object.__setattr__(self, "_forward_hooks", OrderedDict())

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required 'forward' function")

    # From: torch/nn/modules/module.py (Module._call_impl)
    def __call__(self, *args, **kwargs) -> Tensor:
        if not (self._forward_pre_hooks or self._forward_hooks):
            return self.forward(*args, **kwargs)

        for hook in tuple(self._forward_pre_hooks.values()):
            args_result = hook(self, args)
            if args_result is not None:
                args = args_result if isinstance(args_result, tuple) else (args_result,)
        result = self.forward(*args, **kwargs)
        for hook in tuple(self._forward_hooks.values()):
            hook_result = hook(self, args, result)
            if hook_result is not None:
                result = hook_result
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

    def __setattr__(self, name: str, value: Any) -> None:
        if "_modules" not in self.__dict__:
            raise RuntimeError("cannot assign attributes before Module.__init__() call")
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
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

    def apply(self, fn: Callable[["Module"], None]) -> "Module":
        """Applies fn to every submodule (children first) and then to self,
        typically to initialize parameters."""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

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
