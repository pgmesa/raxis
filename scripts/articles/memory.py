"""What the CPU does when a tensor op reads memory (Rebuilding PyTorch, the
memory article).

1. memory/memory_levels.cpp, built with CMake: the latency of one load for
   working sets from 16 KB to 256 MB, in a random order and in order, and
   out[i] = a[i] + b[i] for arrays of growing size. Each step in the
   numbers is a cache level.
2. Independent loads: x[idx] for a million indices into 256 MB, idx in
   order or shuffled. Unlike the pointer chain of 1., every load can start
   before the previous one ends, so many misses overlap.
3. Threads: a + c and a + c.t() with 1, 2, 4, 8 and 16 threads, for
   1000 x 1000 (12 MB, fits in L3) and 4000 x 4000 (192 MB, from memory).
   Run twice in a child process, with the OpenMP threads spread over the
   cores first, and packed two per core (both hardware threads of a core
   before the next one). Without either, where the OS puts them changes
   from run to run.
4. memory/branches.cpp: a filter (keep the x[i] below a threshold) with a
   branch and without one, for 0% to 100% kept in a random order and for
   sorted data, and how much of a repeated random array the branch predictor
   learns.
5. a + b into a new tensor and into an existing one (out=), for 4 MB and
   64 MB results: a new tensor costs a page fault per page the first time
   it is written, unless the allocator reuses memory it already has.
6. Integer // and % on a million int64, with positive dividends and with
   mixed signs, in NumPy, PyTorch and Raxis (Raxis computes them without a
   branch).

    uv run python scripts/articles/memory.py
"""

import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import raxis

HERE = Path(__file__).resolve().parent
BUILD = HERE / "memory" / "build"
THREADS = [1, 2, 4, 8, 16]


def ms(f, repeat=50):
    for _ in range(5):
        f()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def build():
    subprocess.run(["cmake", "-S", HERE / "memory", "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", BUILD, "--config", "Release"], check=True, capture_output=True)


def run(name):
    binary = next(p for p in [BUILD / "Release" / f"{name}.exe", BUILD / name] if p.exists())
    return subprocess.run([binary], check=True, capture_output=True, text=True).stdout


def levels():
    output = run("memory_levels")

    print("Latency of one load, ns")
    print(f"{'working set':>12s}{'random':>10s}{'in order':>10s}")
    rows = {}
    for line in output.splitlines():
        kind, *rest = line.split(",")
        if kind == "latency":
            rows.setdefault(int(rest[0]), {})[rest[1]] = float(rest[2])
    for size, r in rows.items():
        print(f"{size // 1024:>9d} KB{r['random']:10.2f}{r['in order']:10.2f}")

    print("\nout[i] = a[i] + b[i], ns per element")
    print(f"{'n':>10s}{'3 arrays':>12s}{'ns':>8s}{'GB/s':>8s}")
    for line in output.splitlines():
        kind, *rest = line.split(",")
        if kind == "add":
            n, size, ns = int(rest[0]), int(rest[1]), float(rest[2])
            print(f"{n:>10d}{size / 1e6:>9.2f} MB{ns:8.3f}{12 / ns:8.1f}")


def independent_loads():
    torch.set_num_threads(1)
    raxis.set_num_threads(1)
    n = 64 * 1024 * 1024
    x = np.random.rand(n).astype(np.float32)
    print("\nx[idx], a million float32 from 256 MB, ns per element")
    print(f"{'idx':>22s}{'NumPy':>8s}{'PyTorch':>9s}{'Raxis':>8s}")
    rng = np.random.default_rng(0)
    for name, idx in [("in order, step 64", np.arange(0, 64_000_000, 64)), ("shuffled", rng.permutation(n)[:1_000_000])]:
        idx = idx.astype(np.int64)
        out = np.empty(len(idx), dtype=np.float32)
        xt, it, ot = torch.from_numpy(x), torch.from_numpy(idx), torch.from_numpy(out)
        xr, ir, outr = raxis.from_numpy(x), raxis.from_numpy(idx), raxis.from_numpy(out.copy())
        t_np = ms(lambda: np.take(x, idx, out=out))
        t_t = ms(lambda: torch.index_select(xt, 0, it, out=ot))
        t_r = ms(lambda: raxis.index_select(xr, 0, ir, out=outr))
        k = len(idx) / 1e6
        print(f"{name:>22s}{t_np / k:8.2f}{t_t / k:9.2f}{t_r / k:8.2f}")


def fresh_output():
    torch.set_num_threads(1)
    raxis.set_num_threads(1)
    print("\na + b into a new tensor and into an existing one (out=), ms")
    print(f"{'n x n':>12s}{'':>6s}{'NumPy':>8s}{'PyTorch':>9s}{'Raxis':>8s}")
    for n in (1000, 4000):
        a = np.random.rand(n, n).astype(np.float32)
        b = np.random.rand(n, n).astype(np.float32)
        out = np.empty_like(a)
        ta, tb, to = torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(out)
        ra, rb, ro = raxis.from_numpy(a), raxis.from_numpy(b), raxis.from_numpy(out.copy())
        new = [ms(lambda: a + b), ms(lambda: ta + tb), ms(lambda: ra + rb)]
        old = [ms(lambda: np.add(a, b, out=out)), ms(lambda: torch.add(ta, tb, out=to)), ms(lambda: raxis.add(ra, rb, out=ro))]
        for name, t in [("new", new), ("out=", old)]:
            print(f"{n:>5d} x {n:<5d}{name:>6s}{t[0]:8.2f}{t[1]:9.2f}{t[2]:8.2f}")


def branches():
    output = run("branches")
    print("\nFilter x[x < t], ns per element")
    print(f"{'order':>8s}{'kept':>6s}{'branch':>9s}{'no branch':>11s}")
    rows = {}
    for line in output.splitlines():
        kind, *rest = line.split(",")
        if kind == "filter":
            rows.setdefault((rest[0], int(rest[1])), {})[rest[2]] = float(rest[3])
    for (order, kept), r in rows.items():
        print(f"{order:>8s}{kept:>5d}%{r['branch']:9.2f}{r['no branch']:11.2f}")
    print("\nThe random half, from a smaller array run many times, with a branch, ns per element")
    for line in output.splitlines():
        kind, *rest = line.split(",")
        if kind == "repeat":
            print(f"{int(rest[0]):>10d}{float(rest[2]):8.2f}")


def integer_division():
    torch.set_num_threads(1)
    raxis.set_num_threads(1)
    rng = np.random.default_rng(0)
    n = 1_000_000
    b = rng.integers(1, 50, n)
    out = np.empty(n, dtype=np.int64)
    print("\nInteger // and %, a million int64, ms")
    print(f"{'dividends':>12s}{'op':>4s}{'NumPy':>8s}{'PyTorch':>9s}{'Raxis':>8s}")
    for name, a in [("positive", rng.integers(0, 1000, n)), ("mixed signs", rng.integers(-1000, 1000, n))]:
        ta, tb, to = torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(out)
        ra, rb, ro = raxis.from_numpy(a), raxis.from_numpy(b), raxis.from_numpy(out.copy())
        for op in ["floor_divide", "remainder"]:
            t_np = ms(lambda: getattr(np, op)(a, b, out=out))
            t_t = ms(lambda: getattr(torch, op)(ta, tb, out=to))
            t_r = ms(lambda: getattr(raxis, op)(ra, rb, out=ro))
            symbol = "//" if op == "floor_divide" else "%"
            print(f"{name:>12s}{symbol:>4s}{t_np:8.2f}{t_t:9.2f}{t_r:8.2f}")


PLACEMENTS = {
    "spread over the cores": {"OMP_PLACES": "cores", "OMP_PROC_BIND": "spread"},
    "two per core": {"OMP_PLACES": "threads", "OMP_PROC_BIND": "close"},
}


def threads():
    default = torch.get_num_threads()
    print(f"default threads: torch {default}, raxis {raxis.get_num_threads()}; PyTorch / Raxis, ms")
    print(f"{'threads':21s}" + "".join(f"{t:>18d}" for t in THREADS))
    for n in (1000, 4000):
        a = torch.rand(n, n)
        c = torch.rand(n, n)
        out = torch.empty(n, n)
        ar, cr, outr = raxis.from_numpy(a.numpy()), raxis.from_numpy(c.numpy()), raxis.empty([n, n])
        for name, f_t, f_r in [
            ("a + c", lambda: torch.add(a, c, out=out), lambda: raxis.add(ar, cr, out=outr)),
            ("a + c.t()", lambda: torch.add(a, c.t(), out=out), lambda: raxis.add(ar, cr.t(), out=outr)),
        ]:
            cells = []
            for t in THREADS:
                torch.set_num_threads(t)
                raxis.set_num_threads(t)
                cells.append(f"{ms(f_t):.3f} / {ms(f_r):.3f}")
            print(f"{n} x {n} {name:10s}" + "".join(f"{c:>18s}" for c in cells))
    torch.set_num_threads(default)


def main():
    if sys.argv[1:] == ["threads"]:
        threads()
        return
    build()
    levels()
    independent_loads()
    branches()
    fresh_output()
    integer_division()
    for name, env in PLACEMENTS.items():
        print(f"\nThreads, {name}", flush=True)
        subprocess.run([sys.executable, __file__, "threads"], check=True, env={**os.environ, **env})


if __name__ == "__main__":
    main()
