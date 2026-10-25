# Raxis

Raxis is a from-scratch re-implementation of the core of PyTorch, the ATen
tensor library and the autograd engine, in C++20 with Python bindings. It is
distilled from the PyTorch source: it keeps the techniques that make ATen fast
(TensorIterator, vectorized kernels with runtime CPU dispatch, BLAS/MKL)
without the breadth and indirection that make the original hard to read.

The goal is a codebase that someone curious about how PyTorch works inside can
read end to end, while running at PyTorch speed on CPU.

> **Status: alpha.** CPU only. Tensors with views and broadcasting, type
> promotion, elementwise ops, reductions, matmul, autograd, and an `nn`/`optim`
> package with enough to train MLPs, CNNs (Conv1d/2d, pooling, BatchNorm) and a
> small GPT (Embedding, LayerNorm, GELU, scaled dot product attention). CUDA is next.

## Install

```bash
pip install raxis
```

Wheels for Windows and Linux (x86-64), Python 3.10 to 3.14. Raxis has no
runtime dependencies; numpy is optional (`.numpy()`, `raxis.from_numpy`).

```python
import raxis
from raxis import nn, optim

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

x, y = raxis.randn((64, 784)), raxis.randint(0, 10, (64,))
loss = nn.functional.cross_entropy(model(x), y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

The API follows PyTorch's, so most code only needs `torch` replaced by
`raxis`.

`raxis.autograd.graph` exports the autograd graph of a tensor as Graphviz DOT
or Mermaid text, as autograd nodes (`view="backward"`) or as forward ops
(`view="forward"`):

```python
from raxis.autograd import graph

print(graph.to_mermaid(loss, dict(model.named_parameters()), view="forward"))
graph.render(loss, "loss.svg")  # needs pip install 'raxis[viz]' and Graphviz
```

## Inside

| Path | What it holds |
|---|---|
| `libraxis/raxis/tensor` | TensorImpl and storage, views and strides, type promotion, TensorIterator |
| `libraxis/raxis/ops` | Operators and their CPU kernels, compiled once per ISA (AVX2, AVX512, NEON) and picked at runtime |
| `libraxis/raxis/autograd` | Graph, engine and backward functions |
| `libraxis/bindings` | pybind11 module `raxis._C` |
| `raxis/` | Python package: `nn`, `optim`, `autograd.graph`, tensor printing |

Code taken from PyTorch keeps a `// From: aten/src/ATen/...` comment pointing
to the original.

## Build from source

Requirements: a C++20 compiler (MSVC 17+, GCC 12+, Clang 14+), CMake 3.22+ and
[uv](https://docs.astral.sh/uv/).

```bash
git clone --recursive https://github.com/pgmesa/raxis.git
cd raxis
uv sync --extra dev --extra cpu        # or --extra cu130 for CUDA PyTorch
uv run python scripts/build_libraxis.py
uv run pytest tests
```

`uv sync` installs the pinned build and test dependencies from `uv.lock`,
including Intel MKL from PyPI and PyTorch, the reference the tests compare
against. `build_libraxis.py` builds `libraxis` with CMake and installs it into
the `raxis/` package. To work on the C++ side, `libraxis/CMakePresets.json`
has `debug` and `release` presets; `CMakeLists.txt` lists the build options
(`RAXIS_USE_*`).

`scripts/bench_ops.py` benchmarks each op against PyTorch and NumPy,
`scripts/bench_training.py` times a training step of the example models, and
`examples/` has notebooks that train MLPs on two moons and MNIST and a CNN on
MNIST with either Raxis or PyTorch, and `examples/nanogpt` trains
[nanoGPT](https://github.com/karpathy/nanoGPT)'s character-level Shakespeare
model, with its `model.py` unchanged except for the imports, and samples from
it. `raxis.save` and `raxis.load` use PyTorch's checkpoint format, so a model
trained with one library can be sampled with the other.

## License

MIT. Raxis includes code derived from PyTorch (BSD-3-Clause) and links SLEEF,
mimalloc and Intel MKL: see [THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).
