"""Why reading with a stride is slow (Rebuilding PyTorch, part 2).

Two experiments, float32, one thread, out= preallocated, median of many runs:

1. x[::s] + y for a growing stride s, a million elements (out of the
   caches): stride 2 already loses the vector loop, and from 64 bytes on
   every element costs a whole cache line.
2. a + c.t() against a + c for square tensors: the lines of a column of c
   are reused by the next columns while they stay in the caches, so the
   cost of the transpose depends on which cache level holds a column, and
   explodes at n = 1024, where a row is 4096 bytes and every line of a
   column falls into the same cache sets.

    uv run python scripts/articles/strides.py
"""

import statistics
import time

import numpy as np
import torch

import raxis

STRIDES = [1, 2, 4, 8, 16, 64, 1000]
LENGTHS = [1_000_000]
SIZES = [300, 1000, 1024, 4000]


def ms(f, repeat=100):
    for _ in range(5):
        f()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def make(lib, a):
    return a if lib is np else lib.from_numpy(a)


def transpose(lib, x):
    return x.T if lib is np else x.t()


def main():
    torch.set_num_threads(1)
    raxis.set_num_threads(1)
    libs = [np, torch, raxis]
    print(f"numpy {np.__version__}, torch {torch.__version__}, raxis {raxis.__version__}, "
          f"float32, one thread, ns per element (median)")

    rng = np.random.default_rng(0)
    for n in LENGTHS:
        print(f"\nx[::s] + y, {n} elements")
        print(f"{'stride':>12s}" + "".join(f"{lib.__name__:>10s}" for lib in libs))
        y_np = rng.standard_normal(n, dtype=np.float32)
        for s in STRIDES:
            base = rng.standard_normal(n * s, dtype=np.float32)
            cells = []
            for lib in libs:
                x, y = make(lib, base)[::s], make(lib, y_np)
                out = make(lib, np.empty(n, dtype=np.float32))
                cells.append(ms(lambda: lib.add(x, y, out=out)) / n * 1e6)
            print(f"{s:5d} ({4 * s:4d} B)" + "".join(f"{c:10.3f}" for c in cells))

    print("\na + c.t() / a + c, n x n")
    print(f"{'n':>6s}" + "".join(f"{lib.__name__:>22s}" for lib in libs))
    for n in SIZES:
        a_np = rng.standard_normal((n, n), dtype=np.float32)
        c_np = rng.standard_normal((n, n), dtype=np.float32)
        cells = []
        for lib in libs:
            a, c = make(lib, a_np), make(lib, c_np)
            ct = transpose(lib, c)
            out = make(lib, np.empty((n, n), dtype=np.float32))
            contiguous = ms(lambda: lib.add(a, c, out=out))
            transposed = ms(lambda: lib.add(a, ct, out=out))
            cells.append(f"{contiguous / n / n * 1e6:.3f} {transposed / n / n * 1e6:.3f} x{transposed / contiguous:.1f}")
        print(f"{n:6d}" + "".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
