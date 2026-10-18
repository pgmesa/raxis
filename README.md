# Raxis

## Install from pip

```bash
pip install raxis
```

## Build from source

### Prerequisites

- C++20 compiler (MSVC 17+, GCC 12+, Clang 14+)
- CMake 3.22+
- [uv](https://docs.astral.sh/uv/) (manages Python and all Python-side dependencies)

### Clone

```bash
git clone --recursive https://github.com/pgmesa/raxis.git
cd raxis
```

If you already cloned without `--recursive`:

```bash
git submodule update --init
```

### Set up the environment

```bash
uv sync --extra dev
```

This creates `.venv` from `uv.lock` with everything needed to build, test and
benchmark: pybind11, Intel MKL (x86 only), NumPy, pytest and PyTorch (used as
the reference in tests and benchmarks).

#### Intel MKL

On x86, raxis uses Intel MKL for BLAS (matmul and friends) and VML
(vectorized transcendental functions), with the Intel OpenMP runtime (iomp5)
for threading. MKL is installed into the venv from PyPI (`mkl-devel`,
`mkl-static`, `intel-openmp`) as part of the `build` extra, so no system-wide
installation is needed. CMake finds it through `CMAKE_PREFIX_PATH`:

| Platform | `CMAKE_PREFIX_PATH` |
|---|---|
| Windows | `.venv/Library` |
| Linux | `.venv` |

If MKL is not found, raxis falls back to a generic BLAS (e.g. OpenBLAS) and,
if none is available, to its own naive kernels. Check the configure summary
(`BLAS: Intel MKL ...`) to see which one is used.

### Configure and build

From `libraxis/` (Windows paths shown, see the table above for Linux):

```bash
cmake --preset release -DPython_EXECUTABLE=<abs path to .venv/Scripts/python.exe> -DCMAKE_PREFIX_PATH=<abs path to .venv/Library>
cmake --build --preset release --target install
```

The `install` target places the compiled libraries inside the `raxis/` Python
package, which `uv sync` already installed in editable mode.

### Test and benchmark

From the repository root:

```bash
uv run pytest tests
uv run python scripts/bench_ops.py -k sum
```

### Build options

| Option | Default | Description |
|---|---|---|
| `BUILD_PYTHON_BINDINGS` | `ON` | Build the Python module |
| `BUILD_EXAMPLES` | `OFF` | Build C++ examples |
| `RAXIS_USE_PARALLEL` | `ON` | Enable intra-op parallelism |
| `RAXIS_USE_OPENMP` | `ON` | Use OpenMP instead of the native thread pool |
| `RAXIS_USE_SIMD` | `ON` | Enable SIMD vectorized kernels with runtime CPU dispatch |
| `RAXIS_USE_BLAS` | `ON` | Use a BLAS library for linear algebra |
| `RAXIS_USE_MKL` | `ON` | Prefer Intel MKL over generic BLAS (x86 only) |
| `RAXIS_USE_MIMALLOC` | `ON` | Use mimalloc for CPU memory allocation |

Override via command line:

```bash
cmake --preset release -DRAXIS_USE_PARALLEL=OFF
```
