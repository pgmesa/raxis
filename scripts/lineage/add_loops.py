"""
add_loops.cpp against NumPy: why a tensor add with strides can match NumPy on
contiguous inputs and fall far behind it as soon as a dimension is broadcast.

Usage:
    python scripts/lineage/add_loops.py
"""

import subprocess
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
CASES = {
    "1000x1000 + 1000x1000": ((1000, 1000), (1000, 1000)),
    "1000x1000 + 1000x1": ((1000, 1000), (1000, 1)),
    "1000x1000 + 1x1000": ((1000, 1000), (1, 1000)),
    "100x100x100 + 100x1x100": ((100, 100, 100), (100, 1, 100)),
}


def numpy_ms(shape_a, shape_b, runs=101):
    a = np.ones(shape_a, dtype=np.float32)
    b = np.ones(shape_b, dtype=np.float32)
    out = np.empty(shape_a, dtype=np.float32)
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        np.add(a, b, out=out)
        samples.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(samples))


def run_binary(name):
    candidates = [BUILD / "Release" / f"{name}.exe", BUILD / name]
    binary = next(path for path in candidates if path.exists())
    output = subprocess.run([binary], check=True, capture_output=True, text=True).stdout
    return {tuple(line.split(",")[:2]): float(line.split(",")[2]) for line in output.splitlines()}


def main():
    subprocess.run(["cmake", "-S", HERE, "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", BUILD, "--config", "Release"], check=True, capture_output=True)

    base = run_binary("add_loops_base")
    avx2 = run_binary("add_loops_avx2")

    print(f"{'case':26s}{'loop':17s}{'base (ms)':>11s}{'AVX2 (ms)':>11s}")
    for case, (shape_a, shape_b) in CASES.items():
        print(f"{case:26s}{'numpy':17s}{numpy_ms(shape_a, shape_b):11.3f}{'':>11s}")
        for (name, loop), ms in base.items():
            if name == case:
                print(f"{case:26s}{loop:17s}{ms:11.3f}{avx2[(name, loop)]:11.3f}")


if __name__ == "__main__":
    main()
