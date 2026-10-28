# Raxis

Raxis is a from-scratch re-implementation of the core of PyTorch, the ATen
tensor library and the autograd engine, in C++20 with Python bindings. It is
distilled from the PyTorch source: it keeps the techniques that make ATen fast
(TensorIterator, vectorized kernels with runtime CPU dispatch, BLAS/MKL)
without the breadth and indirection that make the original hard to read.

The goal is a codebase that someone curious about how PyTorch works inside can
read end to end, while running at PyTorch speed on CPU and close to it on
NVIDIA GPUs (without cuDNN).

> **Status: alpha.** CPU and CUDA. Tensors with views and broadcasting, type
> promotion, elementwise ops, reductions, matmul, autograd, and an `nn`/`optim`
> package with enough to train MLPs, CNNs (Conv1d/2d, pooling, BatchNorm) and a
> small GPT (Embedding, LayerNorm, GELU, scaled dot product attention), on the
> CPU or on an NVIDIA GPU.

## Install

```bash
pip install raxis
```

Wheels for Windows (x86-64), Linux (x86-64 and aarch64, e.g. a 64-bit
Raspberry Pi OS) and macOS (Apple silicon, macOS 14+), Python 3.10 to 3.14.
Raxis has no runtime dependencies; numpy is optional (`.numpy()`,
`raxis.from_numpy`).

For NVIDIA GPUs, install the CUDA backend as well (Windows and Linux x86-64).
It is a separate package, as JAX's CUDA plugins, so the CPU wheel stays small:

```bash
pip install raxis-cuda12   # driver r525+, GPUs from Volta (sm_70) on
pip install raxis-cuda13   # driver r580+, GPUs from Turing (sm_75) on
```

Each pulls the raxis of the same version and NVIDIA's cuBLAS wheel; the CUDA
runtime is linked in, so only the driver is needed. `import raxis` loads it,
and `raxis.cuda.is_available()` tells whether there is a GPU to use.

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

`examples/autograd_graph.py` draws every view and format side by side.

## Performance

Measured on a Ryzen 7 9700X (Zen 5, 8 cores, AVX512) with Windows 11, 8
threads, PyTorch 2.14.1 and Intel MKL in both libraries. The scripts and
notebooks to reproduce them are in the repo (see [Build from source](#build-from-source)).

**Training the examples.** Time of `trainer.fit` in each notebook (training
plus the validation pass of every epoch) with Raxis, PyTorch and the two
earlier projects of the same author:
[synapx](https://github.com/pgmesa/synapx) (autograd in C++ on top of libtorch)
and [synapgrad](https://github.com/pgmesa/synapgrad) (autograd in pure Python
on top of NumPy).

| Notebook | Raxis | PyTorch | synapx | synapgrad | Params | Train set | Batch | Epochs |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| basic_mlp.ipynb | 0.20 s | 1.4 s | 0.53 s | 0.26 s | 337 | 150 | 4 | 50 |
| mnist_mlp.ipynb | 2.9 s | 4.5 s | 5.6 s | 10.0 s | 178,110 | 48,000 | 64 | 10 |
| mnist_cnn.ipynb | 10.8 s | 37.8 s | - | 4 min 55 s | 20,538 | 48,000 | 128 | 5 |

The models reach the same accuracy with every library (MNIST: ~97.6% val with
the MLP, ~98.6% with the CNN). Most of PyTorch's time in the CNN is Dropout:
its CPU random numbers come from a serial mt19937, Raxis draws them from
Philox in parallel. synapx has no Conv2d.

**nanoGPT.** [nanoGPT](https://github.com/karpathy/nanoGPT)'s
character-level Shakespeare model, trained with `examples/nanogpt/train.py`
(nanoGPT's `model.py` with only the imports changed):

| Config | Raxis step | PyTorch step | Raxis val loss | PyTorch val loss |
|---|--:|--:|--:|--:|
| nanoGPT README's CPU run: 4 layers, n_embd 128, block 64, batch 12, 2000 iters | 13.2 ms | 14.2 ms | 1.9065 | 1.9121 |
| `train_shakespeare_char.py`: 6 layers, n_embd 384, block 256, batch 64, dropout 0.2 | 2.61 s | 4.00 s | 1.4553 | 1.4680 |

Median time of a training step (forward, backward, gradient clipping and
AdamW). The full config would take ~5.1 h with Raxis and ~7.2 h with
PyTorch on this CPU for its 5000 iterations, but the val loss stops
improving around iteration 2000 and the model overfits afterwards, so both
were stopped early: the table has the best val loss up to iteration 2000
(1750 for Raxis, 2000 for PyTorch), against 1.4697 in nanoGPT's README.
Random numbers differ between the libraries, so the losses do too.

**On the GPU.** RTX 5070 Ti (CUDA 13.0, driver 610), float32, PyTorch
2.14.1+cu130, the same TF32 flags in both libraries (nanoGPT turns TF32 on
for matmuls, as its train.py does). Median training step:

| Model | Raxis | PyTorch |
|---|--:|--:|
| MNIST MLP, batch 64 (`scripts/bench_training.py --device cuda`) | 0.54 ms | 0.45 ms |
| MNIST CNN, batch 128 (same) | 1.32 ms | 0.96 ms |
| nanoGPT `train_shakespeare_char.py`, 5000 iters | 47.3 ms | 45.7 ms |

The full nanoGPT run takes 6.0 min with Raxis and 5.8 min with PyTorch, best
val loss 1.4668 and 1.4612 (both at iteration 1750). PyTorch convolves
with cuDNN; Raxis does not link it (a closed library of several hundred MB
that would hide the most interesting kernels) and runs im2col plus cuBLAS.
Its attention is its own FlashAttention-2 style kernel, against PyTorch's
memory-efficient one. At batch 64 the MLP is bound by kernel launches on
any GPU.

**Ops.** A selection of `scripts/bench_ops.py` (float32, forward only unless
noted). Over its 388 cases the median ratio Raxis/PyTorch is 0.85x; the only
case beyond 2x is the large 3x3 convolution, where PyTorch uses oneDNN's
direct convolution and Raxis im2col plus an MKL gemm.

| Op | Shape | Raxis (ms) | PyTorch (ms) |
|---|---|--:|--:|
| add | 1000 x 1000 | 0.017 | 0.017 |
| exp | 1000 x 1000 | 0.123 | 0.130 |
| sum(dim=1) | 1000 x 1000 | 0.010 | 0.018 |
| mm | 512 x 512 @ 512 x 512 | 0.459 | 0.464 |
| softmax | 16384 x 384 | 0.872 | 0.885 |
| layer_norm | 16384 x 384 | 0.726 | 1.182 |
| gelu | 16384 x 384 | 0.814 | 0.800 |
| sdpa (causal) | 64 x 6 x 256 x 64 | 13.98 | 15.23 |
| sort | 1000 x 1000 | 5.98 | 11.03 |
| x[idx] backward | 1000 x 1000 | 0.050 | 0.317 |
| normal_ | 1000 x 1000 | 0.548 | 3.111 |
| dropout | 1000 x 1000 | 0.123 | 6.880 |
| conv2d 3x3, 32 -> 64 ch | 64 x 32 x 26 x 26 | 2.98 | 1.49 |
| conv2d 3x3, 64 -> 64 ch | 64 x 64 x 32 x 32 | 8.94 | 4.32 |

**ARM.** On GitHub's ARM runners (Apple M1 and Neoverse, shared machines,
so no table) the median ratio Raxis/PyTorch over the same cases is 0.79x to
0.88x in float32 and float64.

**Against NumPy.** Raxis wheels are 8-10 MB, smaller than NumPy's (13-17 MB),
while PyTorch's CPU wheels take 110-180 MB. On 1000 x 1000 elementwise ops
Raxis and PyTorch both run 3 to 30x faster than NumPy (`add` 0.017 vs 0.565
ms, `exp` 0.123 vs 0.944 ms), mostly because they split the work between
threads: NumPy's own functions are
[single-threaded by design](https://numpy.org/doc/stable/reference/global_state.html),
only its BLAS calls use several. On 100 x 100 arrays the three are on par.

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
