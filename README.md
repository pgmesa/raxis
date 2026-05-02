# Raxis

## Install from pip

```bash
pip install raxis
```

## Build from source

### Prerequisites

- C++20 compiler (MSVC 19.30+, GCC 12+, Clang 15+)
- CMake 3.22+
- Python 3.10+ with pybind11 (`pip install pybind11`)

### Clone

```bash
git clone --recursive https://github.com/pablomusaber/raxis.git
cd raxis/libraxis
```

If you already cloned without `--recursive`:

```bash
git submodule update --init
```

### Configure and build

```bash
cd libraxis
cmake --preset release
cmake --build --preset release
```

To install the Python package:

```bash
cmake --build --preset release --target install
```

### Build options

| Option | Default | Description |
|---|---|---|
| `BUILD_PYTHON_BINDINGS` | `ON` | Build the Python module |
| `BUILD_EXAMPLES` | `OFF` | Build C++ examples |
| `RAXIS_PARALLEL` | `ON` | Enable intra-op parallelism |
| `RAXIS_USE_OPENMP` | `ON` | Use OpenMP instead of native thread pool |
| `RAXIS_ENABLE_SIMD` | `ON` | Enable AVX2 vectorized kernels |

Override via command line:

```bash
cmake --preset release -DRAXIS_PARALLEL=OFF
```