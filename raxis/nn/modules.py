import operator
from collections import OrderedDict, namedtuple
from typing import Any, Callable, Iterable, Iterator, Mapping

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

    # A module or tensor reachable through several names (shared layers, tied
    # weights such as a language model's embedding and output projection)
    # is yielded once, under its first name, unless remove_duplicate=False:
    # otherwise an optimizer would update a tied weight twice per step.
    # From: torch/nn/modules/module.py (named_modules, _named_members)

    def named_children(self) -> Iterator[tuple[str, "Module"]]:
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def children(self) -> Iterator["Module"]:
        for _, module in self.named_children():
            yield module

    def named_modules(self, memo: set | None = None, prefix: str = "",
                      remove_duplicate: bool = True) -> Iterator[tuple[str, "Module"]]:
        if memo is None:
            memo = set()
        if self in memo:
            return
        if remove_duplicate:
            memo.add(self)
        yield prefix, self
        for name, module in self._modules.items():
            if module is not None:
                yield from module.named_modules(memo, f"{prefix}.{name}" if prefix else name, remove_duplicate)

    def modules(self) -> Iterator["Module"]:
        for _, module in self.named_modules():
            yield module

    def _named_members(self, get_members: Callable, prefix: str, recurse: bool,
                       remove_duplicate: bool) -> Iterator[tuple[str, Tensor]]:
        memo = set()
        modules = (self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
                   if recurse else [(prefix, self)])
        for module_name, module in modules:
            for name, value in get_members(module):
                if value is None or value in memo:
                    continue
                if remove_duplicate:
                    memo.add(value)
                yield (f"{module_name}.{name}" if module_name else name), value

    def named_parameters(self, prefix: str = "", recurse: bool = True,
                         remove_duplicate: bool = True) -> Iterator[tuple[str, Parameter]]:
        return self._named_members(lambda m: m._parameters.items(), prefix, recurse, remove_duplicate)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_buffers(self, prefix: str = "", recurse: bool = True,
                      remove_duplicate: bool = True) -> Iterator[tuple[str, Tensor]]:
        return self._named_members(lambda m: m._buffers.items(), prefix, recurse, remove_duplicate)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, buf in self.named_buffers(recurse=recurse):
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

    # -- Device and dtype --

    # From: torch/nn/modules/module.py (Module._apply)
    def _apply(self, fn: Callable[[Tensor], Tensor]) -> "Module":
        """Applies fn to every parameter, gradient and buffer. A parameter
        stays the same object and takes the new data (torch's
        `param.data = fn(param)`), so an optimizer created before moving the
        model keeps updating the moved parameters."""
        for module in self.children():
            module._apply(fn)
        for param in self._parameters.values():
            if param is None:
                continue
            with raxis.no_grad():
                param._set_data(fn(param))
                if param.grad is not None:
                    param.grad._set_data(fn(param.grad))
        # Through __setattr__: a buffer lives in _buffers and as an attribute
        for name, buf in list(self._buffers.items()):
            if buf is not None:
                setattr(self, name, fn(buf))
        return self

    # From: torch/nn/modules/module.py (Module.to)
    def to(self, *args, **kwargs) -> "Module":
        """to(device), to(dtype) or to(device, dtype). The dtype applies to
        floating-point parameters and buffers only, as in torch."""
        device = kwargs.pop("device", None)
        dtype = kwargs.pop("dtype", None)
        if kwargs:
            raise TypeError(f"to() got unexpected keyword arguments {list(kwargs)}")
        for arg in args:
            if isinstance(arg, raxis.dtype):
                dtype = arg
            else:
                device = arg
        if dtype is not None and not dtype.is_floating:
            raise TypeError(f"nn.Module.to only accepts floating point dtypes, but got desired dtype={dtype}")

        def convert(t: Tensor) -> Tensor:
            if dtype is not None and t.dtype.is_floating:
                return t.to(t.device if device is None else device, dtype)
            return t if device is None else t.to(device)

        return self._apply(convert)

    def cuda(self, device: int | None = None) -> "Module":
        return self.to("cuda" if device is None else f"cuda:{device}")

    def cpu(self) -> "Module":
        return self.to("cpu")

    def float(self) -> "Module":
        return self.to(raxis.float32)

    def double(self) -> "Module":
        return self.to(raxis.float64)

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


# From: torch/nn/modules/container.py
class ModuleList(Module):
    """Holds submodules in a list, registered under the names "0", "1", ...
    so parameters() and train()/eval() reach them (a plain Python list
    would hide them)."""

    def __init__(self, modules: Iterable[Module] | None = None) -> None:
        super().__init__()
        if modules is not None:
            self.extend(modules)

    def _abs_index(self, idx: int) -> str:
        idx = operator.index(idx)
        if not -len(self) <= idx < len(self):
            raise IndexError(f"index {idx} is out of range")
        return str(idx % len(self))

    def __getitem__(self, idx: int | slice) -> "Module | ModuleList":
        if isinstance(idx, slice):
            return type(self)(list(self._modules.values())[idx])
        return self._modules[self._abs_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        self.register_module(self._abs_index(idx), module)

    def __delitem__(self, idx: int | slice) -> None:
        modules = list(self._modules.values())
        del modules[idx if isinstance(idx, slice) else int(self._abs_index(idx))]
        # Renumber what is left, so the names stay "0", "1", ...
        for name in list(self._modules):
            object.__delattr__(self, name)
        self._modules.clear()
        self.extend(modules)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> "ModuleList":
        return self.extend(modules)

    def append(self, module: Module) -> "ModuleList":
        self.register_module(str(len(self)), module)
        return self

    def extend(self, modules: Iterable[Module]) -> "ModuleList":
        for module in modules:
            self.append(module)
        return self

    def insert(self, index: int, module: Module) -> None:
        modules = list(self._modules.values())
        modules.insert(index, module)
        self._modules.clear()
        self.extend(modules)

    def pop(self, idx: int | slice) -> "Module | ModuleList":
        module = self[idx]
        del self[idx]
        return module

    def __repr__(self) -> str:
        """Runs of identical submodules are printed once: (0-3): 4 x Block(...)"""
        reprs = [repr(module) for module in self]
        if not reprs:
            return f"{type(self).__name__}()"
        runs = []  # [first index, last index, repr]
        for i, r in enumerate(reprs):
            if runs and runs[-1][2] == r:
                runs[-1][1] = i
            else:
                runs.append([i, i, r])
        lines = []
        for start, end, r in runs:
            r = r.replace("\n", "\n  ")
            lines.append(f"({start}): {r}" if start == end else f"({start}-{end}): {end - start + 1} x {r}")
        return f"{type(self).__name__}(\n  " + "\n  ".join(lines) + "\n)"


# From: torch/nn/modules/container.py
class ModuleDict(Module):
    """Holds submodules in an ordered dict; its keys are their names, so
    model.blocks["encoder"] and model.blocks.encoder are the same module."""

    def __init__(self, modules: Mapping[str, Module] | Iterable[tuple[str, Module]] | None = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.register_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]
        object.__delattr__(self, key)

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def keys(self):
        return self._modules.keys()

    def items(self):
        return self._modules.items()

    def values(self):
        return self._modules.values()

    def pop(self, key: str) -> Module:
        module = self[key]
        del self[key]
        return module

    def clear(self) -> None:
        for key in list(self._modules):
            del self[key]

    def update(self, modules: Mapping[str, Module] | Iterable[tuple[str, Module]]) -> None:
        items = modules.items() if isinstance(modules, Mapping) else modules
        for key, module in items:
            self[key] = module
