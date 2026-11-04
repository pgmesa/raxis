"""Timings for the article on TensorIterator (Rebuilding PyTorch, part 3).

Each case runs the same operation in NumPy, PyTorch and Raxis on one thread,
writing into an output allocated beforehand unless the case is about the
allocated output, and prints the median of many runs in milliseconds.

    uv run python scripts/articles/tensoriterator.py
"""

import statistics
import time

import numpy as np
import torch

import raxis

torch.set_num_threads(1)
raxis.set_num_threads(1)


def ms(f, repeat=200):
    for _ in range(10):
        f()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def row(label, cells):
    print(f"{label:44s}" + "".join(f"{c:>10s}" for c in cells))


def fmt(t):
    return f"{t:.3f}" if t is not None else "-"


def case(label, make, op, repeat=200):
    """make(lib) -> args; op(lib, *args) runs the operation."""
    cells = []
    for lib in (np, torch, raxis):
        args = make(lib)
        if args is None:
            cells.append(None)
            continue
        cells.append(ms(lambda: op(lib, *args), repeat))
    row(label, [fmt(c) for c in cells])


def rand(lib, *shape, dtype="float32"):
    a = np.random.default_rng(0).standard_normal(shape).astype(dtype)
    if lib is np:
        return a
    return lib.from_numpy(a)


def add_out(lib, a, b, out):
    lib.add(a, b, out=out)


def T(lib, x):
    return x.T if lib is np else x.t()


def permute(lib, x, *dims):
    return np.transpose(x, dims) if lib is np else x.permute(*dims)


def empty(lib, *shape, dtype="float32"):
    if lib is np:
        return np.empty(shape, dtype=dtype)
    return lib.empty(*shape, dtype=getattr(lib, dtype))


print(f"numpy {np.__version__}, torch {torch.__version__}, raxis {raxis.__version__}, one thread, ms (median)")
row("", ["NumPy", "PyTorch", "Raxis"])

print("-- order: reordering finds an order that suits every operand, when there is one")
N = 1000
case("a + b, contiguous", lambda L: (rand(L, N, N), rand(L, N, N), empty(L, N, N)), add_out)
case("a.t() + b.t(), out=c.t()",
     lambda L: (T(L, rand(L, N, N)), T(L, rand(L, N, N)), T(L, empty(L, N, N))), add_out)
case("a + b.t(), out=c", lambda L: (rand(L, N, N), T(L, rand(L, N, N)), empty(L, N, N)), add_out)
case("a + b, out=c.t()", lambda L: (rand(L, N, N), rand(L, N, N), T(L, empty(L, N, N))), add_out)

print("-- coalescing: x[:, ::2] + y[:, ::2], 1-D when M is even, 2-D when it is odd")
case("M = 2000 (1000 x 1000 elements, 1-D)",
     lambda L: (rand(L, N, 2 * N)[:, ::2], rand(L, N, 2 * N)[:, ::2], empty(L, N, N)), add_out)
case("M = 2001 (1000 x 1001 elements, 2-D)",
     lambda L: (rand(L, N, 2 * N + 1)[:, ::2], rand(L, N, 2 * N + 1)[:, ::2], empty(L, N, N + 1)), add_out)

print("-- why merging dims matters: the same million elements in rows of 1000 or of 4")
case("a(1000, 1000) + b(1000, 1): rows of 1000",
     lambda L: (rand(L, N, N), rand(L, N, 1), empty(L, N, N)), add_out)
case("a(250000, 4) + b(250000, 1): rows of 4",
     lambda L: (rand(L, 250000, 4), rand(L, 250000, 1), empty(L, 250000, 4)), add_out)
case("a(250000, 4) + b(1, 1): merged, one row",
     lambda L: (rand(L, 250000, 4), rand(L, 1, 1), empty(L, 250000, 4)), add_out)

# 300^3 is the general case; 256^3 adds the cache-set trap of a power-of-two
# size (rows of 1 KB, planes of 256 KB), which slows down both orders
for M in (300, 256):
    print(f"-- the first operand decides: a + b vs b + a, b = c.permute(2, 0, 1), {M}^3 floats")
    case("a + b (new output)", lambda L: (rand(L, M, M, M), permute(L, rand(L, M, M, M), 2, 0, 1)),
         lambda L, a, b: a + b, repeat=10)
    case("b + a (new output)", lambda L: (rand(L, M, M, M), permute(L, rand(L, M, M, M), 2, 0, 1)),
         lambda L, a, b: b + a, repeat=10)
    case("a + b, out=c", lambda L: (rand(L, M, M, M), permute(L, rand(L, M, M, M), 2, 0, 1), empty(L, M, M, M)),
         add_out, repeat=10)
    case("b + a, out=c", lambda L: (permute(L, rand(L, M, M, M), 2, 0, 1), rand(L, M, M, M), empty(L, M, M, M)),
         add_out, repeat=10)

print("-- types: computing in the common dtype")
case("float64 + float64, out=float64",
     lambda L: (rand(L, N, N, dtype="float64"), rand(L, N, N, dtype="float64"), empty(L, N, N, dtype="float64")), add_out)
case("float32 + float64, out=float64",
     lambda L: (rand(L, N, N), rand(L, N, N, dtype="float64"), empty(L, N, N, dtype="float64")), add_out)
case("float32 + float32, out=float32",
     lambda L: (rand(L, N, N), rand(L, N, N), empty(L, N, N)), add_out)
case("float32 + float32, out=float64",
     lambda L: (rand(L, N, N), rand(L, N, N), empty(L, N, N, dtype="float64")), add_out)
