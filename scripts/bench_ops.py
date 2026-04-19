"""
Performance comparison: raxis vs PyTorch vs NumPy.

Usage:
    python bench_ops.py                        # all benchmarks, CPU
    python bench_ops.py --device cuda          # all benchmarks, CUDA
    python bench_ops.py -k sum                 # filter by name substring
    python bench_ops.py --repeats 50           # more repeats for stability
    python bench_ops.py --threshold 20         # flag anything >20x slower
    python bench_ops.py --no-color             # disable terminal colors
"""

import argparse
import sys
import time

import numpy as np
import torch
import raxis


# ---------------------------------------------------------------------------
# Terminal colors (ANSI escape codes)
# ---------------------------------------------------------------------------

_COLOR_ENABLED = True


def set_color_enabled(enabled):
    global _COLOR_ENABLED
    _COLOR_ENABLED = enabled


def _ansi(code):
    def wrap(text):
        if not _COLOR_ENABLED:
            return str(text)
        return f"\033[{code}m{text}\033[0m"
    return wrap


# Style functions
bold      = _ansi("1")
dim       = _ansi("2")
green     = _ansi("32")
yellow    = _ansi("33")
red       = _ansi("31")
cyan      = _ansi("36")
magenta   = _ansi("35")
bold_green  = _ansi("1;32")
bold_yellow = _ansi("1;33")
bold_red    = _ansi("1;31")
bold_cyan   = _ansi("1;36")


def color_ratio(ratio, threshold):
    text = f"{ratio:5.2f}x"
    if ratio < threshold*0.9:
        return bold_green(text)
    elif ratio < threshold:
        return yellow(text)
    return bold_red(text)


def color_status(status):
    if status == "OK":
        return bold_green(status)
    elif status == "SLOW":
        return bold_red(status)
    elif status == "SKIP":
        return dim(status)
    return status


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def print_header(title):
    line = f"  {'Name':<24s} {'Shape':>28s}  {'raxis (ms)':>11s}  {'torch (ms)':>11s}  {'numpy (ms)':>11s}  {'ratio':>8s}"
    print(f"\n{bold_cyan('=== ' + title + ' ===')}")
    print(dim(line))


def print_row(name, shape_str, r_ms, t_ms, ratio, threshold, np_ms=None, status=None):
    status = status or ("OK" if ratio < threshold else "SLOW")
    np_str = f"{np_ms:11.3f}" if np_ms is not None else f"{'--':>11s}"
    print(f"  {name:<24s} {shape_str:>28s}  {r_ms:11.3f}  {t_ms:11.3f}  {np_str}  {color_ratio(ratio, threshold)}  {color_status(status)}")


def print_row_skip(name, shape_str, t_ms):
    print(f"  {name:<24s} {shape_str:>28s}  {dim('N/A'):>20s}  {t_ms:11.3f}  {'--':>11s}  {'--':>8s}  {color_status('SKIP')}")


def format_shape(*shapes):
    if len(shapes) == 1:
        return str(shapes[0])
    return " x ".join(str(s) for s in shapes)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:

    def __init__(self, device="cpu", dtype="float32", warmup=5, repeats=20, threshold=10.0,
                 target_ops=None):
        self.device = device
        self.dtype = np.dtype(dtype)
        self.warmup = warmup
        self.repeats = repeats
        self.threshold = threshold
        self.target_ops = target_ops
        self.results = []

    def _should_run(self, name):
        if self.target_ops is None:
            return True
        
        for op_name in self.target_ops:
            if op_name in name.lower():
                return True
        
        return False

    def _make_tensors(self, shape, dtype: np.dtype = None, lo=0.0, hi=1.0, seed=42):
        rng = np.random.RandomState(seed)
        data = (rng.rand(*shape) * (hi - lo) + lo).astype(dtype or self.dtype)
        r = raxis.from_numpy(data)
        t = torch.from_numpy(data.copy())
        if self.device != "cpu":
            r = r.to(self.device)
            t = t.to(self.device)
        return data, r, t

    def _time_fn(self, fn):
        for _ in range(self.warmup):
            fn()

        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(self.repeats):
            fn()

        if self.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

        return (time.perf_counter() - t0) / self.repeats

    def bench(self, name, shapes, raxis_fn, torch_fn, numpy_fn=None,
              data_kw=None, shape_label=None):
        if not self._should_run(name):
            return False

        kw = data_kw or {}
        ran = False

        for shape in shapes:
            data, r, t = self._make_tensors(shape, **kw)

            r_time = self._time_fn(lambda: raxis_fn(r))
            t_time = self._time_fn(lambda: torch_fn(t))
            ratio = r_time / max(t_time, 1e-12)

            np_time = None
            if numpy_fn is not None and self.device == "cpu":
                np_time = self._time_fn(lambda: numpy_fn(data))

            status = "OK" if ratio < self.threshold else "SLOW"
            self.results.append({
                "name": name, "shape": shape,
                "raxis_ms": r_time * 1e3, "torch_ms": t_time * 1e3,
                "ratio": ratio,
                "np_ratio": (r_time / max(np_time, 1e-12)) if np_time else None,
                "status": status,
            })

            s_label = shape_label or format_shape(shape)
            np_ms = np_time * 1e3 if np_time else None
            print_row(name, s_label, r_time * 1e3, t_time * 1e3, ratio,
                      self.threshold, np_ms=np_ms)
            ran = True

        return ran

    def bench_pair(self, name, shape_pairs, raxis_fn, torch_fn, data_kw_a=None,
                   data_kw_b=None):
        """Benchmark ops that need two tensors with different shapes (e.g. broadcast)."""
        if not self._should_run(name):
            return False

        kw_a = data_kw_a or {}
        kw_b = data_kw_b or {}
        ran = False

        for shape_a, shape_b in shape_pairs:
            _, r_a, t_a = self._make_tensors(shape_a, seed=1, **kw_a)
            _, r_b, t_b = self._make_tensors(shape_b, seed=2, **kw_b)

            r_time = self._time_fn(lambda: raxis_fn(r_a, r_b))
            t_time = self._time_fn(lambda: torch_fn(t_a, t_b))
            ratio = r_time / max(t_time, 1e-12)

            status = "OK" if ratio < self.threshold else "SLOW"
            self.results.append({
                "name": name, "shape": (shape_a, shape_b),
                "raxis_ms": r_time * 1e3, "torch_ms": t_time * 1e3,
                "ratio": ratio, "np_ratio": None, "status": status,
            })

            s_label = format_shape(shape_a, shape_b)
            print_row(name, s_label, r_time * 1e3, t_time * 1e3, ratio,
                      self.threshold)
            ran = True

        return ran

    def bench_backward(self, name, shapes, forward_fn, data_kw=None):
        if not self._should_run(name):
            return False

        kw = data_kw or {}
        label = name + " bwd"
        ran = False

        for shape in shapes:
            _, r, t = self._make_tensors(shape, **kw)
            t = t.requires_grad_(True)

            t_fwd = forward_fn(t)
            grad_out_t = torch.ones_like(t_fwd)

            def torch_bwd():
                t.grad = None
                t_fwd.backward(grad_out_t, retain_graph=True)

            t_bw_time = self._time_fn(torch_bwd)

            r_bw_time = None
            try:
                r = r.requires_grad_(True)
                r_fwd = forward_fn(r)
                grad_out_r = raxis.ones_like(r_fwd)

                def raxis_bwd():
                    r.grad = None
                    r_fwd.backward(grad_out_r, retain_graph=True)

                r_bw_time = self._time_fn(raxis_bwd)
            except (RuntimeError, AttributeError):
                pass

            s_label = format_shape(shape)
            if r_bw_time is not None:
                ratio = r_bw_time / max(t_bw_time, 1e-12)
                status = "OK" if ratio < self.threshold else "SLOW"
                self.results.append({
                    "name": label, "shape": shape,
                    "raxis_ms": r_bw_time * 1e3, "torch_ms": t_bw_time * 1e3,
                    "ratio": ratio, "np_ratio": None, "status": status,
                })
                print_row(label, s_label, r_bw_time * 1e3, t_bw_time * 1e3,
                          ratio, self.threshold)
            else:
                print_row_skip(label, s_label, t_bw_time * 1e3)

            ran = True

        return ran

    def summary(self):
        if not self.results:
            print(dim("\nNo benchmarks were run."))
            return

        slow = [r for r in self.results if r["status"] == "SLOW"]
        total = len(self.results)
        ok = total - len(slow)
        ratios = [r["ratio"] for r in self.results]

        print(f"\n{bold_cyan('=' * 70)}")
        print(bold(f"  Summary: {ok}/{total} benchmarks within {self.threshold}x threshold"))
        print()
        print(f"  Ratio stats:  "
              f"median = {bold(f'{np.median(ratios):.2f}x')}   "
              f"mean = {np.mean(ratios):.2f}x   "
              f"min = {bold_green(f'{np.min(ratios):.2f}x')}   "
              f"max = {color_ratio(np.max(ratios), self.threshold)}")

        if slow:
            print(f"\n  {bold_red('SLOW benchmarks')} (>{self.threshold}x):")
            for r in slow:
                ratio_str = f"{r['ratio']:.2f}x"
                ratio_colored = bold_red(ratio_str)
                print(f"    {r['name']:<25} {str(r['shape']):>28}  {ratio_colored}")

        print(f"{bold_cyan('=' * 70)}")


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

def run_all(runner):
    S2D = [(100, 100), (1000, 1000), (4000, 4000)]
    S_REDUCE = [(100, 100), (1000, 1000), (4000, 4000)]

    BCAST_PAIRS = [
        ((1000, 1000), (1000, 1)),
        ((1000, 1000), (1, 1000)),
        ((1000, 1000), (1000,)),
    ]

    # -- Binary arithmetic --
    _run_section(runner, "Binary arithmetic", [
        ("add", S2D, lambda r: r + r, lambda t: t + t, lambda n: n + n),
        ("sub", S2D, lambda r: r - r, lambda t: t - t, lambda n: n - n),
        ("mul", S2D, lambda r: r * r, lambda t: t * t, lambda n: n * n),
        ("div", S2D, lambda r: r / r, lambda t: t / t, lambda n: n / n),
    ])

    # -- Binary broadcast --
    _run_section_pair(runner, "Binary broadcast", [
        ("add_bcast", BCAST_PAIRS, lambda a, b: a + b, lambda a, b: a + b),
        ("mul_bcast", BCAST_PAIRS, lambda a, b: a * b, lambda a, b: a * b),
    ])

    # -- Scalar ops --
    _run_section(runner, "Scalar ops", [
        ("add_scalar", S2D, lambda r: r + 3.14, lambda t: t + 3.14, lambda n: n + 3.14),
        ("mul_scalar", S2D, lambda r: r * 2.0, lambda t: t * 2.0, lambda n: n * 2.0),
    ])

    # -- Unary ops --
    _run_section(runner, "Unary ops", [
        ("neg",     S2D, lambda r: r.neg(),     lambda t: t.neg(),     lambda n: -n),
        ("abs",     S2D, lambda r: r.abs(),     lambda t: t.abs(),     lambda n: np.abs(n)),
        ("exp",     S2D, lambda r: r.exp(),     lambda t: t.exp(),     lambda n: np.exp(n)),
        ("log",     S2D, lambda r: r.log(),     lambda t: t.log(),     lambda n: np.log(n),
         {"lo": 0.01, "hi": 10.0}),
        ("sqrt",    S2D, lambda r: r.sqrt(),    lambda t: t.sqrt(),    lambda n: np.sqrt(n),
         {"lo": 0.01, "hi": 10.0}),
        ("sigmoid", S2D, lambda r: r.sigmoid(), lambda t: t.sigmoid(), None),
    ])

    # -- Comparison ops --
    _run_section(runner, "Comparison ops", [
        ("gt", S2D, lambda r: raxis.gt(r, 0.5), lambda t: torch.gt(t, 0.5), None),
        ("lt", S2D, lambda r: raxis.lt(r, 0.5), lambda t: torch.lt(t, 0.5), None),
        ("eq", S2D, lambda r: raxis.eq(r, 0.5), lambda t: torch.eq(t, 0.5), None),
    ])

    # -- Elementwise binary --
    _run_section(runner, "Elementwise binary", [
        ("pow_scalar", S2D, lambda r: r.pow(2.0), lambda t: t.pow(2.0),
         lambda n: np.power(n, 2.0), {"lo": 0.01, "hi": 10.0}),
    ])

    # -- Reductions (all-reduce) --
    _run_section(runner, "Reductions (all-reduce)", [
        ("sum_all",  S_REDUCE, lambda r: r.sum(),  lambda t: t.sum(),  lambda n: n.sum()),
        ("mean_all", S_REDUCE, lambda r: r.mean(), lambda t: t.mean(), lambda n: n.mean()),
        ("prod_all", [(100, 100), (1000, 1000)],
         lambda r: r.prod(), lambda t: t.prod(), lambda n: n.prod(),
         {"lo": 0.99, "hi": 1.01}),
        ("max_all",  S_REDUCE, lambda r: r.max(),  lambda t: t.max(),  lambda n: n.max()),
        ("min_all",  S_REDUCE, lambda r: r.min(),  lambda t: t.min(),  lambda n: n.min()),
        ("amax_all",  S_REDUCE, lambda r: r.amax(), lambda t: t.amax(), None),
        ("amin_all",  S_REDUCE, lambda r: r.amin(), lambda t: t.amin(), None),
    ])

    # -- Reductions (dim) --
    _run_section(runner, "Reductions (dim)", [
        ("sum_dim0",  S_REDUCE, lambda r: r.sum(dim=0),  lambda t: t.sum(dim=0),
         lambda n: n.sum(axis=0)),
        ("sum_dim1",  S_REDUCE, lambda r: r.sum(dim=1),  lambda t: t.sum(dim=1),
         lambda n: n.sum(axis=1)),
        ("mean_dim1", S_REDUCE, lambda r: r.mean(dim=1), lambda t: t.mean(dim=1),
         lambda n: n.mean(axis=1)),
        ("amax_dim1", S_REDUCE, lambda r: r.amax([1]),   lambda t: t.amax(dim=1),
         lambda n: n.max(axis=1)),
        ("amin_dim0", S_REDUCE, lambda r: r.amin([0]),   lambda t: t.amin(dim=0),
         lambda n: n.min(axis=0)),
    ])

    # -- Shape ops --
    _run_section(runner, "Shape ops", [
        ("transpose",  S2D, lambda r: r.transpose(0, 1), lambda t: t.transpose(0, 1), None),
        ("reshape",    S2D, lambda r: r.reshape((-1,)),   lambda t: t.reshape((-1,)),   None),
        ("contiguous", S2D,
         lambda r: r.transpose(0, 1).contiguous(),
         lambda t: t.transpose(0, 1).contiguous(),
         lambda n: np.ascontiguousarray(n.T)),
    ])

    # -- Clone / copy --
    _run_section(runner, "Clone / copy", [
        ("clone", S2D, lambda r: r.clone(), lambda t: t.clone(), lambda n: n.copy()),
    ])

    # -- Backward --
    has_any = False
    for name, shapes, fn, kw in [
        ("add",  S2D,     lambda t: t + t,   None),
        ("mul",  S2D,     lambda t: t * t,   None),
        ("sum",  S_REDUCE, lambda t: t.sum(), None),
        ("exp",  S2D,     lambda t: t.exp(), None),
    ]:
        if runner._should_run(name + " bwd"):
            if not has_any:
                print_header("Backward")
                has_any = True
            runner.bench_backward(name, shapes, fn, data_kw=kw)


def _run_section(runner, title, benchmarks):
    """Run a list of single-tensor benchmarks, printing the header only if any match."""
    header_printed = False

    for entry in benchmarks:
        name, shapes = entry[0], entry[1]
        raxis_fn, torch_fn = entry[2], entry[3]
        numpy_fn = entry[4] if len(entry) > 4 else None
        data_kw = entry[5] if len(entry) > 5 else None

        # Handle the case where numpy_fn slot is actually data_kw (dict)
        if isinstance(numpy_fn, dict):
            data_kw = numpy_fn
            numpy_fn = None

        if not runner._should_run(name):
            continue

        if not header_printed:
            print_header(title)
            header_printed = True

        runner.bench(name, shapes, raxis_fn, torch_fn, numpy_fn=numpy_fn,
                     data_kw=data_kw)


def _run_section_pair(runner, title, benchmarks):
    """Run a list of two-tensor benchmarks, printing the header only if any match."""
    header_printed = False

    for name, pairs, raxis_fn, torch_fn in benchmarks:
        if not runner._should_run(name):
            continue

        if not header_printed:
            print_header(title)
            header_printed = True

        runner.bench_pair(name, pairs, raxis_fn, torch_fn)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    dtypes = ["bool", "uint8", "int32", "int64", "float32", "float64"]
    parser = argparse.ArgumentParser(description="Raxis performance benchmarks")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="float32", choices=dtypes)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("-k", "--filter", dest="target_ops", nargs='+', default=None)
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    if args.no_color or not sys.stdout.isatty():
        set_color_enabled(False)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[!] CUDA not available for torch")
            exit(1)
        elif not (hasattr(raxis, "cuda") and hasattr(raxis.cuda, "is_available")
                  and raxis.cuda.is_available()):
            print("[!] CUDA not available for raxis")
            exit(1)

    print(bold(f"Benchmarking on: {args.device} - {args.dtype}"))
    print(dim(f"Warmup: {args.warmup}  Repeats: {args.repeats}  "
              f"Threshold: {args.threshold}x"))

    runner = BenchmarkRunner(
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        threshold=args.threshold,
        target_ops=args.target_ops,
    )

    run_all(runner)
    runner.summary()


if __name__ == "__main__":
    main()