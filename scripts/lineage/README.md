# Lineage benchmarks

What made my earlier projects slow, measured. They back the numbers of the
blog series; they are not part of the library.

Run them on Windows from a venv with PyTorch, on a clone of
[synapgrad](https://github.com/pgmesa/synapgrad) and a built
[synapx](https://github.com/pgmesa/synapx), both on `PYTHONPATH`.

| Script | What it measures |
|---|---|
| `add_loops.py`, `add_loops.cpp` | `a + b` with a modulo per element, with strides at runtime, with kernels picked from the strides and flat, against NumPy |
| `multitensor_check.cpp` | Multitensor's own code: wrong offsets on expanded tensors, and its `add` |
| `training_split.py` | A training step of each library, and the MLP step with only its tensor operations |
| `conv_algorithms.py` | Windows + tensordot, im2col + matmul and `conv2d`, on NumPy and libtorch, and synapgrad's CNN step without its graph |
| `conv_forward_backward.py` | Windows and im2col on NumPy, forward and backward apart |
