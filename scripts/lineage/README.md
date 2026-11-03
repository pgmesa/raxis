# Lineage benchmarks

What made my earlier projects slow, measured. They back the numbers of the
blog series; they are not part of the library.

Run them from the raxis venv, with [synapx](https://github.com/pgmesa/synapx)
0.4.0 from PyPI (it runs on the same torch 2.14 as the tests) and a clone of
[synapgrad](https://github.com/pgmesa/synapgrad) on `PYTHONPATH`:

```bash
PYTHONPATH=<synapgrad clone> uv run --with "synapx==0.4.0" python scripts/lineage/training_split.py synapx:mlp
```

| Script | What it measures |
|---|---|
| `add_loops.py`, `add_loops.cpp` | `a + b` with a modulo per element, with strides at runtime, with kernels picked from the strides and flat, against NumPy |
| `training_split.py` | A training step of each library, and the MLP step with only its tensor operations |
| `conv_algorithms.py` | Windows + tensordot, im2col + matmul and `conv2d`, on NumPy and libtorch; synapgrad's CNN step without its graph, on NumPy and on torch in Python; `matmul` on both |
| `conv_breakdown.py` | Each step of windows and im2col on NumPy and on torch in Python |
| `conv_forward_backward.py` | Windows and im2col on NumPy, forward and backward apart |
