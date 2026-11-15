"""The same machinery on a GPU (Rebuilding PyTorch, the CUDA article).

1. Asynchronous launches: how long a + b takes to return to Python and how
   long until the GPU has finished it, for 1024 floats and for 4096 x 4096.
2. The caching allocator: a 64 MB and a 4 KB tensor allocated and freed
   from the cache, and with the cache emptied first (cudaFree + cudaMalloc).
3. gpu_kernel: a + b on 4096 x 4096 float32, contiguous, with both inputs
   one element off the 16-byte alignment, with a broadcast row, and with a
   transposed input; time and bandwidth.
4. IntDivider: the magic number and shift for a few divisors, and the
   quotient they give, checked against integer division.
5. Copies from the CPU: 64 MB from pageable and from pinned memory, blocking
   and non_blocking (how long the call takes to return), and pin_memory()
   with an empty host cache and with a cached block.
6. cuda/latency.cu, built with CMake (needs the CUDA toolkit): the latency of
   one load by one GPU thread for working sets from 16 KiB to 512 MiB, in a
   random order and in order. Each step in the numbers is a level of the
   GPU's memory: L1, L2, GDDR7.

    uv run python scripts/articles/cuda.py
"""

import statistics
import subprocess
import time
from pathlib import Path

import torch

import raxis

LIBRARIES = [("PyTorch", torch), ("Raxis", raxis)]
HERE = Path(__file__).resolve().parent
BUILD = HERE / "cuda" / "build"


def host_empty_cache(lib):
    if lib is torch:
        torch._C._host_emptyCache()
    else:
        raxis._C._host_empty_cache()


def median_us(f, repeat=200, setup=None):
    for _ in range(10):
        if setup:
            setup()
        f()
    times = []
    for _ in range(repeat):
        if setup:
            setup()
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e6


def launches():
    print("a + b on the GPU, us: until the call returns / until the GPU has finished")
    print(f"{'':>14s}{'PyTorch':>18s}{'Raxis':>18s}")
    for shape in ([1024], [4096, 4096]):
        cells = []
        for _, lib in LIBRARIES:
            a = lib.rand(*shape, device="cuda")
            b = lib.rand(*shape, device="cuda")
            lib.cuda.synchronize()

            def call():
                a + b

            def call_and_wait():
                a + b
                lib.cuda.synchronize()
            returned = median_us(call, setup=lib.cuda.synchronize)
            finished = median_us(call_and_wait, setup=lib.cuda.synchronize)
            cells.append(f"{returned:8.1f} /{finished:8.1f}")
        print(f"{' x '.join(map(str, shape)):>14s}" + "".join(f"{c:>18s}" for c in cells))

    print("\n1000 calls of a + b on 1024 floats in a row, us per call")
    for name, lib in LIBRARIES:
        a = lib.rand(1024, device="cuda")
        b = lib.rand(1024, device="cuda")
        samples = []
        for _ in range(20):
            lib.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(1000):
                a + b
            lib.cuda.synchronize()
            samples.append((time.perf_counter() - t) / 1000 * 1e6)
        print(f"{name:>10s}{statistics.median(samples):8.2f}")


def allocator():
    print("\nAllocating and freeing an empty CUDA tensor, us")
    print(f"{'':>8s}{'':>14s}{'PyTorch':>10s}{'Raxis':>10s}")
    for size, label in ((16 * 2**20, "64 MB"), (1024, "4 KB")):
        rows = {"cached": [], "cache emptied": []}
        for _, lib in LIBRARIES:
            def alloc():
                lib.empty(size, device="cuda")
            rows["cached"].append(median_us(alloc, setup=lib.cuda.synchronize))
            rows["cache emptied"].append(median_us(alloc, repeat=50, setup=lib.cuda.empty_cache))
        for row, cells in rows.items():
            print(f"{label:>8s}{row:>14s}" + "".join(f"{c:10.1f}" for c in cells))


def kernels():
    n = 4096
    mb = n * n * 4 / 1e6
    print(f"\na + b, {n} x {n} float32 (out of place), ms and GB/s")
    print(f"{'':>22s}{'PyTorch':>18s}{'Raxis':>18s}")
    rows = {}
    for _, lib in LIBRARIES:
        a = lib.rand(n, n, device="cuda")
        b = lib.rand(n, n, device="cuda")
        flat_a = lib.rand(n * n + 1, device="cuda")
        flat_b = lib.rand(n * n + 1, device="cuda")
        row = lib.rand(n, device="cuda")
        cases = [
            ("contiguous", lambda: a + b, 3 * mb),
            ("one element off", lambda: flat_a[1:] + flat_b[1:], 3 * mb),
            ("broadcast row", lambda: a + row, 2 * mb),
            ("transposed b", lambda: a + b.t(), 3 * mb),
        ]
        for label, f, moved in cases:
            for _ in range(10):
                f()
            samples = []
            for _ in range(20):
                lib.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(20):
                    f()
                lib.cuda.synchronize()
                samples.append((time.perf_counter() - t) / 20)
            seconds = min(samples)
            rows.setdefault(label, []).append(f"{seconds * 1e3:6.3f} {moved / 1e3 / seconds:5.0f}")
    for label, cells in rows.items():
        print(f"{label:>22s}" + "".join(f"{c:>18s}" for c in cells))


def int_divider():
    print("\nIntDivider: n / d == (umulhi(n, m) + n) >> s")
    print(f"{'d':>8s}{'s':>4s}{'m':>12s}{'n':>12s}{'umulhi':>12s}{'quotient':>10s}{'n // d':>10s}")
    for d, n in ((3, 100), (1000, 123_456), (4096, 10_000_000), (4097, 10_000_000)):
        s = next(s for s in range(33) if (1 << s) >= d)
        m = ((1 << 32) * ((1 << s) - d)) // d + 1
        assert m < (1 << 32)
        hi = (n * m) >> 32
        q = (hi + n) >> s
        print(f"{d:>8d}{s:>4d}{m:>12d}{n:>12d}{hi:>12d}{q:>10d}{n // d:>10d}")
    # Every n below 2^31 for some divisors, in chunks
    import numpy as np
    n = np.arange(0, 2**31, 7919, dtype=np.uint64)
    for d in (3, 7, 1000, 4097, 65535, 2**31 - 1):
        s = next(s for s in range(33) if (1 << s) >= d)
        m = np.uint64(((1 << 32) * ((1 << s) - d)) // d + 1)
        q = (((n * m) >> np.uint64(32)) + n) >> np.uint64(s)
        assert np.array_equal(q, n // np.uint64(d)), d
    print("checked against n // d for", len(n), "values of n per divisor")


def copies():
    mb = 2**20
    nbytes = 64 * mb
    print("\nCopying 64 MB of float32 from the CPU to the GPU, ms until the call returns")
    print(f"{'':>34s}{'PyTorch':>10s}{'Raxis':>10s}")
    rows = {}
    for _, lib in LIBRARIES:
        pageable = lib.rand(nbytes // 4)
        pinned = pageable.pin_memory()
        dst = lib.empty(nbytes // 4, device="cuda")
        cases = [
            ("pageable, blocking", lambda: dst.copy_(pageable)),
            ("pinned, blocking", lambda: dst.copy_(pinned)),
            ("pageable, non_blocking=True", lambda: dst.copy_(pageable, non_blocking=True)),
            ("pinned, non_blocking=True", lambda: dst.copy_(pinned, non_blocking=True)),
        ]
        for label, f in cases:
            rows.setdefault(label, []).append(median_us(f, repeat=30, setup=lib.cuda.synchronize))

        def pin():
            pageable.pin_memory()
        rows.setdefault("pin_memory(), block cached", []).append(median_us(pin, repeat=30))
        rows.setdefault("pin_memory(), cache emptied", []).append(
            median_us(pin, repeat=10, setup=lambda: host_empty_cache(lib)))
        rows.setdefault("clone() on the CPU, for scale", []).append(median_us(lambda: pageable.clone(), repeat=30))
    for label, cells in rows.items():
        print(f"{label:>34s}" + "".join(f"{c / 1000:10.3f}" for c in cells))


def latency():
    subprocess.run(["cmake", "-S", HERE / "cuda", "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", BUILD, "--config", "Release"], check=True, capture_output=True)
    binary = next(p for p in [BUILD / "Release" / "latency.exe", BUILD / "latency"] if p.exists())
    output = subprocess.run([binary], check=True, capture_output=True, text=True).stdout

    print("\nLatency of one load by one GPU thread: cycles / ns")
    print(f"{'working set':>12s}{'random':>16s}{'in order':>16s}")
    rows = {}
    for line in output.splitlines():
        _, size, kind, cycles, ns = line.split(",")
        rows.setdefault(int(size), {})[kind] = f"{float(cycles):.0f} / {float(ns):.0f}"
    for size, r in rows.items():
        label = f"{size // 1024} KiB" if size < 2**20 else f"{size // 2**20} MiB"
        print(f"{label:>12s}{r['random']:>16s}{r['in order']:>16s}")


def main():
    torch.cuda.init()
    print(torch.cuda.get_device_name(0))
    launches()
    allocator()
    kernels()
    int_divider()
    copies()
    latency()


if __name__ == "__main__":
    main()
