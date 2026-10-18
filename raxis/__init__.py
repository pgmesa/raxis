
import os
import sys
import ctypes
import platform
import importlib.util
from pathlib import Path


__version__ = "0.1.0"

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

    core_lib_path = raxis_lib_dirs / 'libraxis.so'
    ctypes.CDLL(str(core_lib_path), mode=ctypes.RTLD_GLOBAL)
    extension = '.so'

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

Tensor.__repr__ = _str

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


class no_grad:
    """Context manager that disables gradient computation."""
    def __enter__(self):
        self._prev = _C.is_grad_enabled()
        _C.set_grad_enabled(False)
        return self

    def __exit__(self, *args):
        _C.set_grad_enabled(self._prev)


class enable_grad:
    """Context manager that enables gradient computation."""
    def __enter__(self):
        self._prev = _C.is_grad_enabled()
        _C.set_grad_enabled(True)
        return self

    def __exit__(self, *args):
        _C.set_grad_enabled(self._prev)


from raxis import nn, optim
