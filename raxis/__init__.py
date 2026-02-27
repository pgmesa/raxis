
import os
import sys
import ctypes
import platform
import importlib.util
from pathlib import Path


__version__ = "0.1.0-alpha.1"

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
    # Load core library first
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
