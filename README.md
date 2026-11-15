<div align="center">
  <img src="docs/assets/raxis-header.svg" alt="Raxis" width="440">
  <br><br>
  <p><em>PyTorch's core, reforged one axis at a time.</em></p>
  <a href="https://pypi.org/project/raxis/"><img src="https://img.shields.io/pypi/v/raxis?label=pypi&labelColor=3e2731&color=f77622" alt="PyPI version"></a>
  <a href="https://pypi.org/project/raxis/"><img src="https://img.shields.io/pypi/pyversions/raxis?labelColor=3e2731&color=f77622" alt="Python versions"></a>
  <a href="https://pypi.org/project/raxis-cuda13/"><img src="https://img.shields.io/badge/backends-CPU%20%7C%20CUDA-f77622?labelColor=3e2731" alt="Backends: CPU and CUDA"></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/raxis?labelColor=3e2731&color=f77622" alt="License"></a>
</div>

---

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
> package with enough to train MLPs, CNNs (Conv1d/2d, pooling, BatchNorm),
> vision transformers and GPT-2 (Embedding, LayerNorm, GELU, scaled dot product
> attention), on the CPU or on an NVIDIA GPU.

## Install

```bash
pip install raxis
```

Wheels for Windows (x86-64), Linux (x86-64 and aarch64, e.g. a 64-bit
Raspberry Pi OS) and macOS (Apple silicon, macOS 14+), Python 3.10 to 3.14.
Raxis has no runtime dependencies; numpy is optional (`.numpy()`,
`raxis.from_numpy`).

For NVIDIA GPUs, install it with the CUDA backend (Windows and Linux x86-64):

```bash
pip install "raxis[cuda12]"   # driver r525+, GPUs from Volta (sm_70) on
pip install "raxis[cuda13]"   # driver r580+, GPUs from Turing (sm_75) on
```

The backend is a package of its own, as JAX's CUDA plugins, so the CPU wheel
stays small: `pip install raxis-cuda12` / `raxis-cuda13` is the same. It
pulls the raxis of its version and NVIDIA's cuBLAS wheel; the CUDA runtime is
linked in, so only the driver is needed. `import raxis` loads it, and
`raxis.cuda.is_available()` tells whether there is a GPU to use.

## Examples

Each example runs the same code with Raxis or with PyTorch (`--backend raxis`
or `--backend torch`), saves its model in PyTorch's checkpoint format, and has
a script that uses the trained model. On an RTX 5070 Ti (CUDA 13.0, driver
610), float32 with TF32 matmuls and convolutions in both libraries, PyTorch
2.14.1+cu130:

| Example | Raxis | PyTorch |
|---|--:|--:|
| [ResNet-9](examples/cifar10) from scratch on CIFAR-10, 24 epochs | 94.1%, 3.6 min | 93.9%, 2.0 min |
| [ViT-B/16](examples/cifar10) pretrained on ImageNet, fine-tuned on CIFAR-10, 3 epochs | 98.85%, 10.8 min | 98.85%, 9.8 min |
| [GPT-2 124M](examples/nanogpt) fine-tuned on Shakespeare, 20 iterations | val loss 3.273, 29 s | 3.273, 27 s |
| [nanoGPT](examples/nanogpt)'s character-level Shakespeare model from scratch, 5000 iterations | val loss 1.467, 6.0 min | 1.461, 5.8 min |

Test accuracy for CIFAR-10, the best val loss for the GPTs; the times are
whole runs. The ResNet is convolutions and batch norm: PyTorch runs them with
cuDNN, 4.6 s per epoch, and Raxis with im2col and a cuBLAS gemm, 8.3 s. Raxis
does not use cuDNN, so that the way a convolution becomes a matrix product
stays readable in the repo. The ViT and the GPTs are matrix multiplications,
layer norm and attention (Raxis's own FlashAttention-2 style kernel, against
PyTorch's memory-efficient one), and there Raxis is within 10%. GPT-2's
losses agree to four decimals: the same weights and batches give the same
training. Random numbers differ between the libraries, so the other results
do too.

```bash
cd examples/cifar10
python train.py --model resnet9 --backend raxis    # downloads CIFAR-10
python prepare_vit.py                              # timm's ViT-B/16 weights
python train.py --model vit --backend raxis
python infer.py out-raxis-vit --show 16 --grid predictions.png
python infer.py out-raxis-vit my_cat.jpg

cd ../nanogpt
python prepare_gpt2.py                             # GPT-2's weights and BPE Shakespeare
python sample.py --out-dir out-gpt2 --device cuda --start "Hello, I'm a language model,"
python train.py --backend raxis --finetune --device cuda
python sample.py --out-dir out-raxis-gpt2 --device cuda --start "ROMEO:"

python prepare.py                                  # character-level Shakespeare
python train.py --backend raxis --full --device cuda
python sample.py --device cuda
```

The pretrained weights come from Hugging Face as safetensors files, read with
numpy, so running them needs only Raxis, numpy and, for GPT-2's tokenizer,
tiktoken (PyTorch only for `--backend torch`). Details and the recipes are in
each example's directory. `examples/` also has notebooks that train MLPs on
two moons and MNIST and a CNN on MNIST, on the CPU (timed in
[Against my earlier projects](#performance)).

## Performance

**On the GPU.** The same RTX 5070 Ti, with the same TF32 flags in both
libraries: their defaults in `bench_training.py` (TF32 for convolutions only),
TF32 matmuls too in nanoGPT, as its train.py does. Median training step:

| Model | Raxis | PyTorch |
|---|--:|--:|
| MNIST MLP, batch 64 (`scripts/bench_training.py --device cuda`) | 0.54 ms | 0.45 ms |
| MNIST CNN, batch 128 (same) | 1.32 ms | 0.96 ms |
| nanoGPT `train_shakespeare_char.py` | 47.3 ms | 45.7 ms |

The CNN is where cuDNN pays off, as in the ResNet. At batch 64 the MLP is small
enough that its step is bound by the kernel launches, not by the GPU.

**On the CPU.** Measured on a Ryzen 7 9700X (Zen 5, 8 cores, AVX512) with
Windows 11, 8 threads, PyTorch 2.14.1 and Intel MKL in both libraries. The
scripts and notebooks to reproduce them are in the repo (see
[Build from source](#build-from-source)). nanoGPT's character-level
Shakespeare model, trained with `examples/nanogpt/train.py` (nanoGPT's
`model.py` with only the imports changed):

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

**Ops.** A selection of `scripts/bench_ops.py` (float32, forward only unless
noted). Over its 388 cases the median ratio Raxis/PyTorch is 0.85x; in that
run the only case beyond 2x was the large 3x3 convolution, where PyTorch uses oneDNN's
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
Raxis and PyTorch both run 3 to 30x faster than NumPy here (`add` 0.017 vs
0.565 ms, `exp` 0.123 vs 0.944 ms). Part of it is threads: NumPy's own
functions are
[single-threaded by design](https://numpy.org/doc/stable/reference/global_state.html),
only its BLAS calls use several. The rest, for the cheapest ops, is memory:
on Windows each new NumPy output comes fresh from the system and pays its page
faults, while Raxis and PyTorch reuse memory through mimalloc. On 100 x 100
arrays the three are on par.

**Against my earlier projects.** Time of `trainer.fit` in each notebook
(training plus the validation pass of every epoch), on the same CPU, with
Raxis, PyTorch and the two projects that came before Raxis:
[SynapX](https://github.com/pgmesa/synapx) (autograd in C++ on top of libtorch)
and [SynapGrad](https://github.com/pgmesa/synapgrad) (autograd in pure Python
on top of NumPy).

| Notebook | Raxis | PyTorch | SynapX | SynapGrad | Params | Train set | Batch | Epochs |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| basic_mlp.ipynb | 0.22 s | 0.50 s | 2.0 s | 0.33 s | 337 | 150 | 4 | 50 |
| mnist_mlp.ipynb | 3.0 s | 4.6 s | 6.3 s | 9.4 s | 178,110 | 48,000 | 64 | 10 |
| mnist_cnn.ipynb | 12.5 s | 47.2 s | 3 min 7 s | 5 min 6 s | 20,538 | 48,000 | 128 | 5 |

The models reach the same accuracy with every library (MNIST: ~97.6% val with
the MLP, ~98.6% with the CNN). Most of PyTorch's time in the CNN is Dropout:
its CPU random numbers come from a serial mt19937, Raxis draws them from
Philox in parallel, the generator PyTorch itself uses on CUDA.

## Inside

| Path | What it holds |
|---|---|
| `libraxis/raxis/tensor` | TensorImpl and storage, views and strides, type promotion, TensorIterator |
| `libraxis/raxis/ops` | Operators and their CPU kernels, compiled once per ISA (AVX2, AVX512, NEON) and picked at runtime |
| `libraxis/raxis/ops/cuda` | The CUDA kernels: elementwise loops, reductions, attention, convolutions, sort, indexing |
| `libraxis/raxis/cuda` | CUDA runtime: streams and events, caching allocator, pinned host memory, cuBLAS handles |
| `libraxis/raxis/autograd` | Graph, engine and backward functions |
| `libraxis/bindings` | pybind11 module `raxis._C` |
| `raxis/` | Python package: `nn`, `optim`, `cuda`, `autograd.graph`, tensor printing |
| `plugins/cuda` | The `raxis-cuda12` / `raxis-cuda13` packages |

Much of the code comes from PyTorch, and many places carry a `// From: aten/src/ATen/...` comment pointing
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
the `raxis/` package; with `--cuda` it also builds the CUDA backend (needs
the CUDA toolkit, 12.8 or newer, which brings cuBLAS), and `import raxis`
loads that build before any installed `raxis-cuda*` package. To work on the
C++ side, `libraxis/CMakePresets.json`
has `debug` and `release` presets; `CMakeLists.txt` lists the build options
(`RAXIS_USE_*`).

`scripts/bench_ops.py` benchmarks each op against PyTorch and NumPy,
`scripts/bench_training.py` times a training step of the example models,
`scripts/lineage` measures what made my earlier projects slow, and
`examples/` holds the notebooks and the [examples](#examples). `raxis.save`
and `raxis.load` use PyTorch's checkpoint format, so a model trained with one
library can be used with the other.

## Use from C++

`libraxis` is a C++20 library with a CMake package, so a C++ project can use
the tensors, the ops and autograd without Python:

```cmake
find_package(Raxis CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE Raxis::Raxis)
```

```cpp
#include <raxis/raxis.h>

auto a = raxis::arange(0, 6).reshape({2, 3}).to(raxis::dtypes::Float32).requires_grad_(true);
auto y = a.exp().matmul(raxis::ones({3, 2})).sum();
y.backward();
std::cout << a.grad() << "\n";
```

Build and install it from `libraxis/` with CMake (`cmake --install`), shared by
default. `-DRAXIS_BUILD_STATIC=ON` (CMake 3.24+) builds a static library
instead, to link into a program with no `libraxis` next to it: the package then
installs SLEEF and mimalloc with it, finds MKL again in the consuming project,
and `Raxis::Raxis` links the whole archive, because the CPU kernels register
themselves from static initializers that a linker would otherwise drop. The
whole library with static MKL is about 30 MB, against more than 300 MB for
libtorch's CPU library. The CUDA backend needs the shared build. The C++ API
follows PyTorch's ATen, but it is not documented yet and may still change.

The `nn` modules and the optimizers are Python only, as they were in PyTorch
before libtorch's C++ frontend. They are thin layers over ops that C++ already
has (convolution, layer norm, attention, the fused Adam kernel), so a C++
`nn` in the style of libtorch's `torch::nn` would build on those. It has not been
added for now; it may be later, if it proves worth it.

## License

MIT. Raxis includes code derived from PyTorch (BSD-3-Clause) and links SLEEF,
mimalloc, Intel MKL (x86-64), OpenBLAS (Linux aarch64) and, in the CUDA
packages, the CUDA runtime and CUB: see
[THIRD_PARTY_NOTICES.txt](THIRD_PARTY_NOTICES.txt).
