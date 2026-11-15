"""Reductions and threads (Rebuilding PyTorch, the reductions article).

1. reductions/sum_loops.cpp, built with CMake: a float32 sum written five
   ways (one float accumulator, four, one AVX2 vector, four vectors, one
   double), their speed from the L2 and their error for 1M and 16M numbers.
2. The error of sum() in NumPy, PyTorch and Raxis, for 16M floats in a row
   and down the columns of a 2M x 8 matrix.
3. sum(), sum(0) and sum(1) of a 1000 x 1000 and a 4000 x 4000 float32
   tensor, one thread.
4. The same sums with 1 to 16 threads: does the result change?
5. a + b (out=) around the grain size of 32768 elements, one thread and
   eight: where threads start to pay; and the sums of 3. with 1 and 8
   threads.

    uv run python scripts/articles/reductions.py
"""

import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import raxis

HERE = Path(__file__).resolve().parent
BUILD = HERE / "reductions" / "build"


def ms(f, repeat=50):
    for _ in range(5):
        f()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def set_threads(n):
    torch.set_num_threads(n)
    raxis.set_num_threads(n)


def sum_loops():
    subprocess.run(["cmake", "-S", HERE / "reductions", "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", BUILD, "--config", "Release"], check=True, capture_output=True)
    binary = next(p for p in [BUILD / "Release" / "sum_loops.exe", BUILD / "sum_loops"] if p.exists())
    output = subprocess.run([binary], check=True, capture_output=True, text=True).stdout

    print("Sum of 64Ki float32 from the L2, ns per element")
    for line in output.splitlines():
        kind, *rest = line.split(",")
        if kind == "speed":
            print(f"{rest[0]:>14s}{float(rest[1]):8.3f}")
    print("\nRelative error of the sum, floats from [0, 1)")
    for line in output.splitlines():
        kind, *rest = line.split(",")
        if kind == "error":
            print(f"{int(rest[0]):>10d}{rest[1]:>14s}{float(rest[2]):>14.1f}{float(rest[3]):>11.1e}")
    print("\nSum of 2^25 ones")
    for line in output.splitlines():
        kind, *rest = line.split(",")
        if kind == "ones":
            print(f"{rest[1]:>14s}{float(rest[2]):>14.1f}")


def library_error():
    set_threads(1)
    rng = np.random.default_rng(0)
    x = rng.random(16 * 2**20, dtype=np.float32)
    m = x.reshape(-1, 8)
    exact_x = x.astype(np.float64).sum()
    exact_m = m.astype(np.float64).sum(0)

    tx, rx = torch.from_numpy(x), raxis.from_numpy(x)
    tm, rm = torch.from_numpy(m), raxis.from_numpy(m)

    print("\nRelative error of sum(), 16M float32 from [0, 1), one thread")
    print(f"{'':>26s}{'NumPy':>10s}{'PyTorch':>10s}{'Raxis':>10s}")
    row = [abs(float(s) - exact_x) / exact_x for s in (x.sum(), tx.sum().item(), rx.sum().item())]
    print(f"{'16M in a row':>26s}" + "".join(f"{e:10.1e}" for e in row))

    cols = [m.sum(0), tm.sum(0).numpy(), np.asarray(rm.sum(0).numpy())]
    row = [float(np.max(np.abs(c.astype(np.float64) - exact_m) / exact_m)) for c in cols]
    print(f"{'2M x 8, sum(0), worst col':>26s}" + "".join(f"{e:10.1e}" for e in row))

    ones = np.ones(2**25, dtype=np.float32)
    print(f"{'2^25 ones':>26s}" + "".join(f"{float(s):>10.0f}" for s in (ones.sum(), torch.from_numpy(ones).sum().item(), raxis.from_numpy(ones).sum().item())))


def shapes():
    set_threads(1)
    print("\nsum of a float32 matrix, ms, one thread (NumPy / PyTorch / Raxis)")
    for n in (1000, 4000):
        a = np.random.rand(n, n).astype(np.float32)
        ta, ra = torch.from_numpy(a), raxis.from_numpy(a)
        for name, f_np, f_t, f_r in [
            ("sum()", lambda: a.sum(), lambda: ta.sum(), lambda: ra.sum()),
            ("sum(0)", lambda: a.sum(0), lambda: ta.sum(0), lambda: ra.sum(0)),
            ("sum(1)", lambda: a.sum(1), lambda: ta.sum(1), lambda: ra.sum(1)),
        ]:
            print(f"{n:>5d} x {n:<5d}{name:>8s}{ms(f_np):9.3f}{ms(f_t):9.3f}{ms(f_r):9.3f}")


def thread_bits():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10_000_000, dtype=np.float32)
    m = rng.standard_normal((1000, 1000), dtype=np.float32)
    exact = x.astype(np.float64).sum()
    tx, rx = torch.from_numpy(x), raxis.from_numpy(x)
    tm, rm = torch.from_numpy(m), raxis.from_numpy(m)

    print(f"\nsum() of 10M float32 from a normal distribution, by thread count (exact {exact:.4f})")
    print(f"{'threads':>8s}{'PyTorch':>14s}{'Raxis':>14s}{'sum(0) same':>13s}{'sum(1) same':>13s}")

    first = None
    for t in (1, 2, 3, 4, 6, 8, 16):
        set_threads(t)
        cols = (tm.sum(0).numpy().tobytes(), np.asarray(rm.sum(0).numpy()).tobytes(),
                tm.sum(1).numpy().tobytes(), np.asarray(rm.sum(1).numpy()).tobytes())
        first = first or cols
        same0 = cols[0] == first[0] and cols[1] == first[1] and cols[0] == cols[1]
        same1 = cols[2] == first[2] and cols[3] == first[3] and cols[2] == cols[3]
        print(f"{t:>8d}{tx.sum().item():14.4f}{rx.sum().item():14.4f}{str(same0):>13s}{str(same1):>13s}")


def grain_size(library):
    # Each library runs in a process of its own: on Linux PyTorch brings
    # libgomp and Raxis iomp5, and the threads of one OpenMP runtime can
    # still be spinning, waiting for work, while the other one is timed
    for n in (16_384, 32_768, 32_769, 65_536, 131_072, 1_048_576):
        a = np.random.rand(n).astype(np.float32)
        b = np.random.rand(n).astype(np.float32)

        if library == "torch":
            ta, tb, to = torch.from_numpy(a), torch.from_numpy(b), torch.empty(n)
            f = lambda: torch.add(ta, tb, out=to)
        else:
            ra, rb, ro = raxis.from_numpy(a), raxis.from_numpy(b), raxis.empty([n])
            f = lambda: raxis.add(ra, rb, out=ro)

        cells = []
        for t in (1, 8):
            set_threads(t)
            cells.append(ms(f, repeat=500) * 1e3)
        print(f"{n:>10d}" + "".join(f"{c:9.1f}" for c in cells), flush=True)


def reduction_threads():
    print("\nsum of a float32 matrix, ms: PyTorch 1 / 8 threads, Raxis 1 / 8 threads")
    for n in (1000, 4000):
        a = np.random.rand(n, n).astype(np.float32)
        ta, ra = torch.from_numpy(a), raxis.from_numpy(a)
        for name, f_t, f_r in [
            ("sum()", lambda: ta.sum(), lambda: ra.sum()),
            ("sum(0)", lambda: ta.sum(0), lambda: ra.sum(0)),
            ("sum(1)", lambda: ta.sum(1), lambda: ra.sum(1)),
        ]:
            cells = []
            for f in (f_t, f_r):
                for t in (1, 8):
                    set_threads(t)
                    cells.append(ms(f))
            print(f"{n:>5d} x {n:<5d}{name:>8s}" + "".join(f"{c:9.3f}" for c in cells))


def main():
    if sys.argv[1:2] == ["grain"]:
        grain_size(sys.argv[2])
        return
    sum_loops()
    library_error()
    shapes()
    thread_bits()
    reduction_threads()
    set_threads(8)
    for library, name in [("torch", "PyTorch"), ("raxis", "Raxis")]:
        print(f"\na + b (out=), float32, {name}, us: 1 thread / 8 threads", flush=True)
        subprocess.run([sys.executable, __file__, "grain", library], check=True)


if __name__ == "__main__":
    main()
