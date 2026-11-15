"""Timings for the article on CPU kernels (Rebuilding PyTorch, part 4).

Runs each operation on one thread with every CPU capability that the
machine supports, forced through RAXIS_CPU_CAPABILITY and
ATEN_CPU_CAPABILITY (each one in a fresh process: the capability is read
once, at the first operation), and prints the median of many runs in
milliseconds. NumPy, which has its own dispatch, runs once as a reference.

    uv run python scripts/articles/kernels.py
"""

import json
import os
import statistics
import subprocess
import sys
import time

CAPABILITIES = ["default", "avx2", "avx512"]
N = 1_000_000


def ms(f, repeat=300):
    for _ in range(20):
        f()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        times.append(time.perf_counter() - t)
    return statistics.median(times) * 1e3


def cases(lib):
    """(label, function) for lib in {numpy, torch, raxis}, float32, out=."""
    import numpy as np
    a_np = np.random.default_rng(0).standard_normal(N).astype(np.float32)
    b_np = np.random.default_rng(1).standard_normal(N).astype(np.float32)
    if lib == "numpy":
        out = np.empty_like(a_np)
        return [
            ("add", lambda: np.add(a_np, b_np, out=out)),
            ("exp", lambda: np.exp(a_np, out=out)),
            ("sigmoid", None),
            ("tanh", lambda: np.tanh(a_np, out=out)),
            ("sin", lambda: np.sin(a_np, out=out)),
        ]
    m = __import__(lib)
    a, b = m.from_numpy(a_np), m.from_numpy(b_np)
    out = m.empty(N)
    return [
        ("add", lambda: m.add(a, b, out=out)),
        ("exp", lambda: m.exp(a, out=out)),
        ("sigmoid", lambda: m.sigmoid(a, out=out)),
        ("tanh", lambda: m.tanh(a, out=out)),
        ("sin", lambda: m.sin(a, out=out)),
    ]


def child(lib):
    if lib == "torch":
        import torch
        torch.set_num_threads(1)
        capability = torch.backends.cpu.get_cpu_capability()
    elif lib == "raxis":
        import raxis
        raxis.set_num_threads(1)
        capability = raxis.backends.cpu.get_cpu_capability()
    else:
        capability = "native"
    result = {label: (ms(f) if f else None) for label, f in cases(lib)}
    print(json.dumps({"capability": capability, "times": result}))


def run(lib, capability):
    env = dict(os.environ, RAXIS_CPU_CAPABILITY=capability, ATEN_CPU_CAPABILITY=capability,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, __file__, "--child", lib], env=env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main():
    import numpy
    import torch
    import raxis

    print(f"numpy {numpy.__version__}, torch {torch.__version__}, raxis {raxis.__version__}, "
          f"{N} float32, one thread, ms (median)")

    columns = [("NumPy", run("numpy", "default"))]
    for lib in ("torch", "raxis"):
        for capability in CAPABILITIES:
            r = run(lib, capability)
            columns.append((f"{lib} {r['capability']}", r))

    labels = list(columns[0][1]["times"])
    print(f"{'':10s}" + "".join(f"{name:>16s}" for name, _ in columns))
    for label in labels:
        cells = []
        for _, r in columns:
            t = r["times"][label]
            cells.append(f"{t:16.3f}" if t is not None else f"{'-':>16s}")
        print(f"{label:10s}" + "".join(cells))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        main()
