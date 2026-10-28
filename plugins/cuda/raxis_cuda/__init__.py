"""The CUDA backend of raxis: libraxis_cuda built with one CUDA major version.

Installed as raxis_cuda12 (wheel raxis-cuda12) or raxis_cuda13 (raxis-cuda13)
from this same source. `import raxis` finds the package and calls load(); it
is not meant to be imported on its own. The CUDA runtime is linked
statically and cuBLAS comes from NVIDIA's wheel (nvidia-cublas-cu12 or
nvidia-cublas), so the machine only needs the NVIDIA driver.
"""
import os
import ctypes
import platform
import importlib.util
from pathlib import Path


CUDA_MAJOR = int(__name__.removeprefix('raxis_cuda'))

package_path = Path(__file__).parent.absolute()
library_path = package_path / 'lib' / ('raxis_cuda.dll' if platform.system() == 'Windows' else 'libraxis_cuda.so')
# The toolkit the library was built with, e.g. "13.0.88"
cuda_version = (package_path / 'lib' / 'cuda_version.txt').read_text().strip()


def _cublas_dir() -> Path | None:
    """Where NVIDIA's cuBLAS wheel put its libraries (None without it: then
    the system's, e.g. a CUDA toolkit on the library path).

    The wheels changed layout with CUDA 13: nvidia/cublas/{lib,bin} for 12,
    nvidia/cu13/{lib,bin/x86_64} for 13 (one directory for all its libraries).
    """
    spec = importlib.util.find_spec('nvidia')
    if spec is None or not spec.submodule_search_locations:
        return None
    windows = platform.system() == 'Windows'
    if CUDA_MAJOR == 12:
        relative = Path('cublas', 'bin' if windows else 'lib')
    else:
        relative = Path(f'cu{CUDA_MAJOR}', 'bin/x86_64' if windows else 'lib')
    for root in spec.submodule_search_locations:
        if (Path(root) / relative).is_dir():
            return Path(root) / relative
    return None


def load() -> None:
    """Loads libraxis_cuda, which registers its kernels, allocator and hooks
    in libraxis (already loaded by raxis)."""
    cublas_dir = _cublas_dir()
    if platform.system() == 'Windows':
        # Since Python 3.8 Windows no longer resolves DLL dependencies through PATH
        if cublas_dir is not None:
            os.add_dll_directory(str(cublas_dir))
        ctypes.CDLL(str(library_path))
        return
    # By full path first (as torch's _preload_cuda_deps): libraxis_cuda then
    # finds them already loaded by their soname, whatever its RUNPATH says
    if cublas_dir is not None:
        for name in ('libcublasLt.so', 'libcublas.so'):
            ctypes.CDLL(str(cublas_dir / f'{name}.{CUDA_MAJOR}'), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
