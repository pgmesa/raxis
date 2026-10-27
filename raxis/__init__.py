
import os
import sys
import ctypes
import platform
import importlib.util
from pathlib import Path


__version__ = "0.3.0"

package_path = Path(__file__).parent.absolute()

# Add raxis dlls
raxis_lib_dirs = package_path / 'lib'

# Platform-specific shared library loading
if platform.system() == 'Windows':
    os.add_dll_directory(str(raxis_lib_dirs))
    core_lib_path = raxis_lib_dirs / "raxis.dll"
    ctypes.WinDLL(str(core_lib_path))
    extension = '.pyd'
else:
    # Pre-load shared dependencies from lib/ so the linker can find them
    skip = {'libraxis', '_C'}
    for lib in sorted(raxis_lib_dirs.glob('*.so*')):
        if not any(s in lib.stem for s in skip):
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)

    core_lib_path = raxis_lib_dirs / ('libraxis.dylib' if sys.platform == 'darwin' else 'libraxis.so')
    ctypes.CDLL(str(core_lib_path), mode=ctypes.RTLD_GLOBAL)
    extension = '.so'

# The CUDA backend, when it was built (RAXIS_USE_CUDA): loading it registers
# its kernels, allocator and hooks in libraxis. Without it raxis is CPU only
# and raxis.cuda.is_available() is False
_cuda_lib_name = {'Windows': 'raxis_cuda.dll', 'Darwin': 'libraxis_cuda.dylib'}.get(platform.system(), 'libraxis_cuda.so')
if (raxis_lib_dirs / _cuda_lib_name).exists():
    ctypes.CDLL(str(raxis_lib_dirs / _cuda_lib_name), mode=ctypes.RTLD_GLOBAL)

# Dynamically identify and load the `_C` module
_C_module_file = None
for file in raxis_lib_dirs.iterdir():
    if file.suffix == extension and file.stem.startswith('_C'):
        _C_module_file = file
        break

if not _C_module_file:
    raise ImportError("Cannot find the raxis._C shared library file")

# Load the C++ extension as a Python module
spec = importlib.util.spec_from_file_location("raxis._C", _C_module_file)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load spec for {_C_module_file}")

_C = importlib.util.module_from_spec(spec)
sys.modules["raxis._C"] = _C
spec.loader.exec_module(_C)

# Expose everything from the dynamically loaded _C module
from raxis._C import *
from raxis._tensor_str import _str

import builtins as _builtins
import collections as _collections
import functools as _functools

Tensor.__repr__ = _str
# Identity hash, as torch: __eq__ is elementwise, so pybind11 leaves Tensor
# unhashable, but tensors must key dicts (optimizer state, parameter maps)
Tensor.__hash__ = object.__hash__
# float and int are shadowed below by the dtype aliases
Tensor.__float__ = lambda self: _builtins.float(self.item())
Tensor.__int__ = lambda self: _builtins.int(self.item())

# Dtype aliases, as in torch
float = float32
double = float64
int = int32
long = int64


def _return_self(method):
    def inplace(self, *args, **kwargs):
        method(self, *args, **kwargs)
        return self
    inplace.__name__ = method.__name__
    inplace.__doc__ = method.__doc__
    return inplace


# pybind11 wraps the `Tensor&` returned by in-place methods in a new Python
# object. Return `self` instead (as torch does), so `p *= 0.1` keeps `p` the
# same object, and a Parameter stays a Parameter.
for _name in dir(Tensor):
    _is_inplace_method = _name.endswith("_") and not _name.startswith("_")
    _is_inplace_operator = _name in ("__iadd__", "__isub__", "__imul__", "__itruediv__")
    if _is_inplace_method or _is_inplace_operator:
        setattr(Tensor, _name, _return_self(getattr(Tensor, _name)))
del _name, _is_inplace_method, _is_inplace_operator


def _is_int_sequence(args):
    # `int` is the raxis.int dtype here, as in torch
    return len(args) >= 1 and _builtins.all(isinstance(a, _builtins.int) for a in args)


def _sizes_as_varargs(fn):
    """Lets `fn(3, 4)` mean `fn((3, 4))`, as torch's argument parser does for
    shapes (torch.ones(2, 3))."""
    @_functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if _is_int_sequence(args):
            args = (args,)
        return fn(*args, **kwargs)
    return wrapper


def _method_sizes_as_varargs(method):
    """Same as _sizes_as_varargs for methods: x.reshape(2, 3), x.permute(1, 0)."""
    @_functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if _is_int_sequence(args):
            args = (args,)
        return method(self, *args, **kwargs)
    return wrapper


class return_types:
    """Named results, as torch.return_types: values, indices = x.max(0)"""
    max = _collections.namedtuple("max", ["values", "indices"])
    min = _collections.namedtuple("min", ["values", "indices"])
    sort = _collections.namedtuple("sort", ["values", "indices"])
    topk = _collections.namedtuple("topk", ["values", "indices"])


def _named_result(fn, result_type):
    @_functools.wraps(fn)
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result_type(*result) if isinstance(result, tuple) else result
    return wrapper


for _name in ("max", "min", "sort", "topk"):
    _result_type = getattr(return_types, _name)
    globals()[_name] = _named_result(globals()[_name], _result_type)
    setattr(Tensor, _name, _named_result(getattr(Tensor, _name), _result_type))
del _result_type

for _name in ("empty", "zeros", "ones", "rand", "randn"):
    globals()[_name] = _sizes_as_varargs(globals()[_name])
for _name in ("reshape", "view", "permute", "expand", "flip"):
    setattr(Tensor, _name, _method_sizes_as_varargs(getattr(Tensor, _name)))
del _name


# From: torch/utils/_contextlib.py (_DecoratorContextManager)
class _GradModeDecorator:
    """Lets a grad mode context manager decorate a function; every call runs
    in a new instance, so a recursive call does not overwrite the saved mode."""

    def __call__(self, func):
        @_functools.wraps(func)
        def wrapper(*args, **kwargs):
            with type(self)():
                return func(*args, **kwargs)
        return wrapper


class no_grad(_GradModeDecorator):
    """Context manager (or decorator) that disables gradient computation."""
    def __enter__(self):
        self._prev = _C.is_grad_enabled()
        _C.set_grad_enabled(False)
        return self

    def __exit__(self, *args):
        _C.set_grad_enabled(self._prev)


class enable_grad(_GradModeDecorator):
    """Context manager (or decorator) that enables gradient computation."""
    def __enter__(self):
        self._prev = _C.is_grad_enabled()
        _C.set_grad_enabled(True)
        return self

    def __exit__(self, *args):
        _C.set_grad_enabled(self._prev)


from raxis import autograd, cuda, nn, optim
from raxis.serialization import save, load
