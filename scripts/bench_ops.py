"""
Performance comparison: Raxis vs PyTorch vs NumPy.

Usage:
    python bench_ops.py                        # all benchmarks, CPU
    python bench_ops.py --device cuda          # all benchmarks, CUDA
    python bench_ops.py -k sum                 # filter by name substring
    python bench_ops.py --repeats 50           # more repeats for stability
    python bench_ops.py --threshold 2          # flag anything >2x slower
    python bench_ops.py --no-color             # disable terminal colors
"""

import argparse
import sys
import time

import numpy as np
import torch
import raxis

# ---------------------------------------------------------------------------
# Dtype categories
# ---------------------------------------------------------------------------

ALL = "all"
NO_BOOL = "no_bool"
FLOAT = "float"

_FLOAT_DTYPES = {"float32", "float64"}
_BOOL_DTYPES = {"bool"}


def dtype_allowed(dtype_str, category):
    if category == ALL:
        return True
    if category == NO_BOOL:
        return dtype_str not in _BOOL_DTYPES
    if category == FLOAT:
        return dtype_str in _FLOAT_DTYPES
    return True


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
                 target_ops=None, no_numpy=False, forward_only=False, backward_only=False):
        self.device = device
        self.dtype = np.dtype(dtype)
        self.dtype_str = dtype
        self.warmup = warmup
        self.repeats = repeats
        self.threshold = threshold
        self.target_ops = target_ops
        self.results = []
        self.no_numpy = no_numpy
        self.forward_only = forward_only
        self.backward_only = backward_only

    def _should_run(self, name):
        if self.target_ops is None:
            return True

        for op_name in self.target_ops:
            if op_name in name.lower():
                return True

        return False

    def _make_tensors(self, shape, dtype: np.dtype = None, lo=0.0, hi=1.0, seed=42):
        rng = np.random.RandomState(seed)
        dt = dtype or self.dtype

        if np.issubdtype(dt, np.integer):
            lo_i = int(lo) if lo != 0.0 else 0
            hi_i = int(hi) if hi != 1.0 else 100
            data = rng.randint(lo_i, max(hi_i, lo_i + 1), size=shape).astype(dt)
        elif np.issubdtype(dt, np.bool_):
            data = rng.randint(0, 2, size=shape).astype(dt)
        else:
            data = (rng.rand(*shape) * (hi - lo) + lo).astype(dt)

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

        samples = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(self.repeats):
                fn()
            if self.device != "cpu" and torch.cuda.is_available():
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) / self.repeats)

        return min(samples)

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
            if numpy_fn is not None and self.device == "cpu" and not self.no_numpy:
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

    def bench_pair(self, name, shape_pairs, raxis_fn, torch_fn, numpy_fn=None,
               data_kw_a=None, data_kw_b=None):
        if not self._should_run(name):
            return False

        kw_a = data_kw_a or {}
        kw_b = data_kw_b or {}
        ran = False

        for shape_a, shape_b in shape_pairs:
            data_a, r_a, t_a = self._make_tensors(shape_a, seed=1, **kw_a)
            data_b, r_b, t_b = self._make_tensors(shape_b, seed=2, **kw_b)

            r_time = self._time_fn(lambda: raxis_fn(r_a, r_b))
            t_time = self._time_fn(lambda: torch_fn(t_a, t_b))
            ratio = r_time / max(t_time, 1e-12)

            np_time = None
            if numpy_fn is not None and self.device == "cpu" and not self.no_numpy:
                np_time = self._time_fn(lambda: numpy_fn(data_a, data_b))

            status = "OK" if ratio < self.threshold else "SLOW"
            self.results.append({
                "name": name, "shape": (shape_a, shape_b),
                "raxis_ms": r_time * 1e3, "torch_ms": t_time * 1e3,
                "ratio": ratio,
                "np_ratio": (r_time / max(np_time, 1e-12)) if np_time else None,
                "status": status,
            })

            s_label = format_shape(shape_a, shape_b)
            np_ms = np_time * 1e3 if np_time else None
            print_row(name, s_label, r_time * 1e3, t_time * 1e3, ratio,
                    self.threshold, np_ms=np_ms)
            ran = True

        return ran
    
    def bench_multi(self, name, configs, raxis_fn, torch_fn, numpy_fn=None):
        """Benchmark ops that need multiple tensors built from a config.
 
        configs: list of dicts with keys:
            - shapes: list of shapes for input tensors
            - label:  string label for the row
        """
        if not self._should_run(name):
            return False
 
        ran = False
        for cfg in configs:
            shapes = cfg["shapes"]
            label = cfg["label"]
 
            r_tensors = []
            t_tensors = []
            np_arrays = []
            for idx, shape in enumerate(shapes):
                data, r, t = self._make_tensors(shape, seed=42 + idx)
                r_tensors.append(r)
                t_tensors.append(t)
                np_arrays.append(data)
 
            r_time = self._time_fn(lambda: raxis_fn(r_tensors))
            t_time = self._time_fn(lambda: torch_fn(t_tensors))
            ratio = r_time / max(t_time, 1e-12)
 
            np_time = None
            if numpy_fn is not None and self.device == "cpu":
                np_time = self._time_fn(lambda: numpy_fn(np_arrays))
 
            status = "OK" if ratio < self.threshold else "SLOW"
            self.results.append({
                "name": name, "shape": label,
                "raxis_ms": r_time * 1e3, "torch_ms": t_time * 1e3,
                "ratio": ratio,
                "np_ratio": (r_time / max(np_time, 1e-12)) if np_time else None,
                "status": status,
            })
 
            np_ms = np_time * 1e3 if np_time else None
            print_row(name, label, r_time * 1e3, t_time * 1e3, ratio,
                      self.threshold, np_ms=np_ms)
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
    if not runner.backward_only:
        run_forward_benchmarks(runner)
        
    if not runner.forward_only:
        run_backward_benchmarks(runner)

    
def run_forward_benchmarks(runner):
    S2D = [(100, 100), (500, 500), (1000, 1000)]
    S_REDUCE = [(100, 100), (500, 500), (1000, 1000)]

    BCAST_PAIRS = [
        ((1000, 1000), (1000, 1)),
        ((1000, 1000), (1, 1000)),
        ((1000, 1000), (1000,)),
    ]

    # -- Binary arithmetic --
    _run_section(runner, "Binary arithmetic", [
        ("add",  ALL,     S2D, lambda r: r + r, lambda t: t + t, lambda n: n + n),
        ("sub",  NO_BOOL, S2D, lambda r: r - r, lambda t: t - t, lambda n: n - n),
        ("mul",  ALL,     S2D, lambda r: r * r, lambda t: t * t, lambda n: n * n),
        ("div",  FLOAT,   S2D, lambda r: r / r, lambda t: t / t, lambda n: n / n),
    ])

    # -- Binary broadcast --
    _run_section_pair(runner, "Binary broadcast", [
        ("add_bcast", ALL, BCAST_PAIRS, lambda a, b: a + b, lambda a, b: a + b, lambda a, b: a + b),
        ("mul_bcast", ALL, BCAST_PAIRS, lambda a, b: a * b, lambda a, b: a * b, lambda a, b: a * b),
    ])

    # -- Scalar ops --
    _run_section(runner, "Scalar ops", [
        ("add_scalar", ALL,     S2D, lambda r: r + 3, lambda t: t + 3, lambda n: n + 3),
        ("mul_scalar", NO_BOOL, S2D, lambda r: r * 2, lambda t: t * 2, lambda n: n * 2),
    ])

    # -- Unary ops --
    _run_section(runner, "Unary ops", [
        ("neg",     NO_BOOL, S2D, lambda r: r.neg(), lambda t: t.neg(), lambda n: -n),
        ("abs",     NO_BOOL,     S2D, lambda r: r.abs(), lambda t: t.abs(), lambda n: np.abs(n)),
        ("exp",     FLOAT,   S2D, lambda r: r.exp(), lambda t: t.exp(), lambda n: np.exp(n)),
        ("expm1",   FLOAT,   S2D, lambda r: r.expm1(), lambda t: t.expm1(), lambda n: np.expm1(n)),
        ("log",     FLOAT,   S2D, lambda r: r.log(), lambda t: t.log(), lambda n: np.log(n),
         {"lo": 0.01, "hi": 10.0}),
        ("log2",    FLOAT,   S2D, lambda r: r.log2(), lambda t: t.log2(), lambda n: np.log2(n),
         {"lo": 0.01, "hi": 10.0}),
        ("log10",   FLOAT,   S2D, lambda r: r.log10(), lambda t: t.log10(), lambda n: np.log10(n),
         {"lo": 0.01, "hi": 10.0}),
        ("log1p",   FLOAT,   S2D, lambda r: r.log1p(), lambda t: t.log1p(), lambda n: np.log1p(n),
         {"lo": 0.01, "hi": 10.0}),
        ("sqrt",    FLOAT,   S2D, lambda r: r.sqrt(), lambda t: t.sqrt(), lambda n: np.sqrt(n),
         {"lo": 0.01, "hi": 10.0}),
        ("sigmoid", FLOAT,   S2D, lambda r: r.sigmoid(), lambda t: t.sigmoid(), None),
        ("ceil",    FLOAT,   S2D, lambda r: r.ceil(), lambda t: t.ceil(), lambda n: np.ceil(n)),
        ("floor",   FLOAT,   S2D, lambda r: r.floor(), lambda t: t.floor(), lambda n: np.floor(n)),
        ("round",   FLOAT,   S2D, lambda r: r.round(), lambda t: t.round(),lambda n: np.round(n)),
        ("rsqrt",      FLOAT, S2D, lambda r: r.rsqrt(), lambda t: t.rsqrt(),lambda n: 1.0 / np.sqrt(n),
         {"lo": 0.01, "hi": 10.0}),
        ("reciprocal", FLOAT, S2D, lambda r: r.reciprocal(), lambda t: t.reciprocal(), lambda n: 1.0 / n,
         {"lo": 0.1, "hi": 10.0}),
        ("sin",     FLOAT,   S2D, lambda r: r.sin(), lambda t: t.sin(), lambda n: np.sin(n)),
        ("cos",     FLOAT,   S2D, lambda r: r.cos(), lambda t: t.cos(), lambda n: np.cos(n)),
        ("tan",     FLOAT,   S2D, lambda r: r.tan(), lambda t: t.tan(), lambda n: np.tan(n)),
        ("asin",    FLOAT,   S2D, lambda r: r.asin(), lambda t: t.asin(), lambda n: np.arcsin(n),
         {"lo": -0.9, "hi": 0.9}),
        ("acos",    FLOAT,   S2D, lambda r: r.acos(), lambda t: t.acos(), lambda n: np.arccos(n),
         {"lo": -0.9, "hi": 0.9}),
        ("atan",    FLOAT,   S2D, lambda r: r.atan(), lambda t: t.atan(), lambda n: np.arctan(n)),
        ("sinh",    FLOAT,   S2D, lambda r: r.sinh(), lambda t: t.sinh(), lambda n: np.sinh(n),
         {"lo": -2.0, "hi": 2.0}),
        ("cosh",    FLOAT,   S2D, lambda r: r.cosh(), lambda t: t.cosh(), lambda n: np.cosh(n),
         {"lo": -2.0, "hi": 2.0}),
        ("tanh",    FLOAT,   S2D, lambda r: r.tanh(), lambda t: t.tanh(), lambda n: np.tanh(n)),
        ("erf",     FLOAT,   S2D, lambda r: r.erf(), lambda t: t.erf(), None),
        ("erfc",    FLOAT,   S2D, lambda r: r.erfc(), lambda t: t.erfc(), None),
    ])

    # -- Comparison ops --
    _run_section(runner, "Comparison ops", [
        ("gt", ALL, S2D, lambda r: raxis.gt(r, 0.5), lambda t: torch.gt(t, 0.5), None),
        ("ge", ALL, S2D, lambda r: raxis.ge(r, 0.5), lambda t: torch.ge(t, 0.5), None),
        ("lt", ALL, S2D, lambda r: raxis.lt(r, 0.5), lambda t: torch.lt(t, 0.5), None),
        ("le", ALL, S2D, lambda r: raxis.le(r, 0.5), lambda t: torch.le(t, 0.5), None),
        ("eq", ALL, S2D, lambda r: raxis.eq(r, 0.5), lambda t: torch.eq(t, 0.5), None),
        ("ne", ALL, S2D, lambda r: raxis.ne(r, 0.5), lambda t: torch.ne(t, 0.5), None),
    ])

    # -- Elementwise binary --
    _run_section(runner, "Elementwise binary", [
        ("maximum", ALL, S2D,
         lambda r: raxis.maximum(r, r), lambda t: torch.maximum(t, t), lambda n: np.maximum(n, n)),
        ("minimum", ALL, S2D,
         lambda r: raxis.minimum(r, r), lambda t: torch.minimum(t, t), lambda n: np.minimum(n, n)),
        ("pow_sq",      FLOAT, S2D,
         lambda r: r.pow(2.0), lambda t: t.pow(2.0), lambda n: np.power(n, 2.0),
         {"lo": 0.01, "hi": 10.0}),
        ("pow_cube",    FLOAT, S2D,
         lambda r: r.pow(3.0), lambda t: t.pow(3.0), lambda n: np.power(n, 3.0),
         {"lo": 0.01, "hi": 10.0}),
        ("pow_sqrt",    FLOAT, S2D,
         lambda r: r.pow(0.5), lambda t: t.pow(0.5), lambda n: np.power(n, 0.5),
         {"lo": 0.01, "hi": 10.0}),
        ("pow_recip",   FLOAT, S2D,
         lambda r: r.pow(-1.0), lambda t: t.pow(-1.0), lambda n: np.power(n, -1.0),
         {"lo": 0.1, "hi": 10.0}),
        ("pow_generic",  FLOAT, S2D,
         lambda r: r.pow(2.5), lambda t: t.pow(2.5), lambda n: np.power(n, 2.5),
         {"lo": 0.01, "hi": 10.0}),
        ("pow_tensor",  FLOAT, S2D,
         lambda r: raxis.pow(r, r), lambda t: torch.pow(t, t), lambda n: np.power(n, n),
         {"lo": 0.5, "hi": 2.0}),
    ])

    # -- Logical ops --
    _run_section(runner, "Logical ops", [
        ("logical_not", ALL, S2D,
         lambda r: raxis.logical_not(r), lambda t: torch.logical_not(t),
         lambda n: np.logical_not(n)),
        ("logical_and", ALL, S2D,
         lambda r: raxis.logical_and(r, r), lambda t: torch.logical_and(t, t),
         lambda n: np.logical_and(n, n)),
        ("logical_or", ALL, S2D,
         lambda r: raxis.logical_or(r, r), lambda t: torch.logical_or(t, t),
         lambda n: np.logical_or(n, n)),
    ])

    # -- Reductions (all-reduce) --
    _run_section(runner, "Reductions (all-reduce)", [
        ("sum_all",  ALL,   S_REDUCE, lambda r: r.sum(),  lambda t: t.sum(),  lambda n: n.sum()),
        ("mean_all", FLOAT, S_REDUCE, lambda r: r.mean(), lambda t: t.mean(), lambda n: n.mean()),
        ("prod_all", ALL,   [(100, 100)],
         lambda r: r.prod(), lambda t: t.prod(), lambda n: n.prod(),
         {"lo": 0.99, "hi": 1.01}),
        ("max_all",  ALL,   S_REDUCE, lambda r: r.max(),  lambda t: t.max(),  lambda n: n.max()),
        ("min_all",  ALL,   S_REDUCE, lambda r: r.min(),  lambda t: t.min(),  lambda n: n.min()),
        ("amax_all", ALL,   S_REDUCE, lambda r: r.amax(), lambda t: t.amax(), None),
        ("amin_all", ALL,   S_REDUCE, lambda r: r.amin(), lambda t: t.amin(), None),
    ])

    # -- Reductions (dim) --
    _run_section(runner, "Reductions (dim)", [
        ("sum_dim0",  ALL,   S_REDUCE, lambda r: r.sum(dim=0),  lambda t: t.sum(dim=0),
         lambda n: n.sum(axis=0)),
        ("sum_dim1",  ALL,   S_REDUCE, lambda r: r.sum(dim=1),  lambda t: t.sum(dim=1),
         lambda n: n.sum(axis=1)),
        ("mean_dim1", FLOAT, S_REDUCE, lambda r: r.mean(dim=1), lambda t: t.mean(dim=1),
         lambda n: n.mean(axis=1)),
        ("prod_dim0", ALL,   [(100, 100)], lambda r: r.prod(dim=0), lambda t: t.prod(dim=0),
         lambda n: n.prod(axis=0)),
        ("prod_dim1", ALL,   [(100, 100)], lambda r: r.prod(dim=1), lambda t: t.prod(dim=1),
         lambda n: n.prod(axis=1)),
        ("amax_dim1", ALL,   S_REDUCE, lambda r: r.amax([1]),   lambda t: t.amax(dim=1),
         lambda n: n.max(axis=1)),
        ("amin_dim0", ALL,   S_REDUCE, lambda r: r.amin([0]),   lambda t: t.amin(dim=0),
         lambda n: n.min(axis=0)),
    ])

    # -- Shape ops --
    _run_section(runner, "Shape ops", [
        ("transpose",  ALL, S2D, lambda r: r.transpose(0, 1), lambda t: t.transpose(0, 1), None),
        ("reshape",    ALL, S2D, lambda r: r.reshape((-1,)),   lambda t: t.reshape((-1,)),   None),
        ("contiguous", ALL, S2D,
         lambda r: r.transpose(0, 1).contiguous(),
         lambda t: t.transpose(0, 1).contiguous(),
         lambda n: np.ascontiguousarray(n.T)),
    ])

    # -- Clone / copy / fill --
    _run_section(runner, "Clone / copy / fill", [
        ("clone", ALL, S2D, lambda r: r.clone(), lambda t: t.clone(), lambda n: n.copy()),
        ("fill",  ALL, S2D,
         lambda r: r.fill_(3), lambda t: t.fill_(3), lambda n: n.fill(3)),
    ])
    
    # -- Cat / Stack / Unbind --
    CAT_CONFIGS_DIM0 = [
        {"shapes": [(500, 500)] * 2,  "label": "2x(500,500) dim0"},
        {"shapes": [(1000, 1000)] * 2, "label": "2x(1000,1000) dim0"},
        {"shapes": [(100, 1000)] * 10, "label": "10x(100,1000) dim0"},
    ]
 
    CAT_CONFIGS_DIM1 = [
        {"shapes": [(500, 500)] * 2,  "label": "2x(500,500) dim1"},
        {"shapes": [(1000, 1000)] * 2, "label": "2x(1000,1000) dim1"},
        {"shapes": [(1000, 100)] * 10, "label": "10x(1000,100) dim1"},
    ]
 
    STACK_CONFIGS = [
        {"shapes": [(500, 500)] * 2,  "label": "2x(500,500) dim0"},
        {"shapes": [(1000, 1000)] * 2, "label": "2x(1000,1000) dim0"},
        {"shapes": [(100, 100)] * 20, "label": "20x(100,100) dim0"},
    ]
 
    header_printed = False
 
    for name, configs, r_fn, t_fn, np_fn in [
        ("cat_dim0", CAT_CONFIGS_DIM0,
         lambda ts: raxis.cat(ts, dim=0),
         lambda ts: torch.cat(ts, dim=0),
         lambda ns: np.concatenate(ns, axis=0)),
        ("cat_dim1", CAT_CONFIGS_DIM1,
         lambda ts: raxis.cat(ts, dim=1),
         lambda ts: torch.cat(ts, dim=1),
         lambda ns: np.concatenate(ns, axis=1)),
        ("stack_dim0", STACK_CONFIGS,
         lambda ts: raxis.stack(ts, dim=0),
         lambda ts: torch.stack(ts, dim=0),
         lambda ns: np.stack(ns, axis=0)),
    ]:
        if not runner._should_run(name):
            continue
        if not header_printed:
            print_header("Cat / Stack / Unbind")
            header_printed = True
        runner.bench_multi(name, configs, r_fn, t_fn, numpy_fn=np_fn)
 
    # unbind uses the single-tensor bench since it takes one input
    UNBIND_SHAPES = [(10, 1000), (100, 1000), (1000, 1000)]
    if header_printed or runner._should_run("unbind"):
        if not header_printed:
            print_header("Cat / Stack / Unbind")
            header_printed = True
        runner.bench("unbind_dim0", UNBIND_SHAPES,
                     lambda r: raxis.unbind(r, dim=0),
                     lambda t: torch.unbind(t, dim=0),
                     numpy_fn=None)
    
    # -- Dot --
    S_DOT = [(128,), (1024,), (8192,), (65536,)]
 
    _run_section(runner, "Dot product", [
        ("dot", NO_BOOL, S_DOT,
        lambda r: raxis.dot(r, r),
        lambda t: torch.dot(t, t),
        lambda n: np.dot(n, n)),
    ])
 
    # -- MV --
    S_MV = [
        ((64, 64), (64,)),
        ((256, 256), (256,)),
        ((512, 768), (768,)),
        ((1024, 1024), (1024,)),
    ]
 
    _run_section_pair(runner, "Matrix-vector (mv)", [
        ("mv", NO_BOOL, S_MV,
        lambda a, b: raxis.mv(a, b),
        lambda a, b: torch.mv(a, b),
        lambda a, b: np.matvec(a, b)),
    ])
 
    # -- MM --
    S_MM = [
        ((16, 16), (16, 16)),
        ((64, 64), (64, 64)),
        ((128, 128), (128, 128)),
        ((256, 256), (256, 256)),
        ((256, 256), (256, 256)),
        ((512, 512), (512, 512)),
        # ((1024, 1024), (1024, 1024)),
        ((128, 512), (512, 64)),
        #((512, 768), (768, 768)),
    ]
 
    S_MATMUL = [
        ((4, 64, 64), (4, 64, 64)),
        ((8, 128, 128), (8, 128, 128)),
        ((8, 128, 128), (1, 128, 128)),
        ((32, 64, 64), (32, 64, 64)),
        ((8, 128, 64), (8, 64, 128)),
        ((8, 128, 64), (1, 64, 128)),
        ((8, 512, 64), (8, 64, 512)),
    ]
 
    _run_section_pair(runner, "Matmul (mm)", [
        ("mm", NO_BOOL, S_MM,
        lambda a, b: raxis.mm(a, b),
        lambda a, b: torch.mm(a, b)),
    ])
 
    _run_section_pair(runner, "Matmul (batched)", [
        ("matmul_bmm", NO_BOOL, S_MATMUL,
        lambda a, b: raxis.matmul(a, b),
        lambda a, b: torch.matmul(a, b),
        lambda a, b: np.matmul(a, b)),
    ])
                
            
def run_backward_benchmarks(runner):
    S2D = [(100, 100), (500, 500), (1000, 1000)]
    S_REDUCE = [(100, 100), (500, 500), (1000, 1000)]
 
    # -- Binary arithmetic backward --
    BINARY_BWD = [
        ("add bwd",  FLOAT, S2D, lambda t: t + t),
        ("sub bwd",  FLOAT, S2D, lambda t: t - t),
        ("mul bwd",  FLOAT, S2D, lambda t: t * t),
        ("div bwd",  FLOAT, S2D, lambda t: t / (t + 0.1)),
    ]
 
    has_any = False
    for name, cat, shapes, fn in BINARY_BWD:
        if not dtype_allowed(runner.dtype_str, cat):
            continue
        if runner._should_run(name):
            if not has_any:
                print_header("Backward: Binary arithmetic")
                has_any = True
            runner.bench_backward(name, shapes, fn)
 
    # -- Unary backward --
    UNARY_BWD = [
        ("neg bwd",        FLOAT, S2D, lambda t: t.neg()),
        ("abs bwd",        FLOAT, S2D, lambda t: t.abs()),
        ("exp bwd",        FLOAT, S2D, lambda t: t.exp()),
        ("log bwd",        FLOAT, S2D, lambda t: t.log(), {"lo": 0.01, "hi": 10.0}),
        ("sqrt bwd",       FLOAT, S2D, lambda t: t.sqrt(), {"lo": 0.01, "hi": 10.0}),
        ("sigmoid bwd",    FLOAT, S2D, lambda t: t.sigmoid()),
        ("tanh bwd",       FLOAT, S2D, lambda t: t.tanh()),
        ("sin bwd",        FLOAT, S2D, lambda t: t.sin()),
        ("cos bwd",        FLOAT, S2D, lambda t: t.cos()),
        ("reciprocal bwd", FLOAT, S2D, lambda t: t.reciprocal(), {"lo": 0.1, "hi": 10.0}),
    ]
 
    has_any = False
    for entry in UNARY_BWD:
        name, cat, shapes, fn = entry[0], entry[1], entry[2], entry[3]
        kw = entry[4] if len(entry) > 4 else None
        if not dtype_allowed(runner.dtype_str, cat):
            continue
        if runner._should_run(name):
            if not has_any:
                print_header("Backward: Unary ops")
                has_any = True
            runner.bench_backward(name, shapes, fn, data_kw=kw)
 
    # -- Reduction backward --
    REDUCE_BWD = [
        ("sum bwd",      FLOAT, S_REDUCE, lambda t: t.sum()),
        ("sum_dim0 bwd", FLOAT, S_REDUCE, lambda t: t.sum(dim=0)),
        ("sum_dim1 bwd", FLOAT, S_REDUCE, lambda t: t.sum(dim=1)),
        ("mean bwd",     FLOAT, S_REDUCE, lambda t: t.mean()),
        ("mean_dim0 bwd", FLOAT, S_REDUCE, lambda t: t.mean(dim=0)),
    ]
 
    has_any = False
    for name, cat, shapes, fn in REDUCE_BWD:
        if not dtype_allowed(runner.dtype_str, cat):
            continue
        if runner._should_run(name):
            if not has_any:
                print_header("Backward: Reductions")
                has_any = True
            runner.bench_backward(name, shapes, fn)
 
    # -- Pow backward --
    POW_BWD = [
        ("pow_sq bwd",      FLOAT, S2D, lambda t: t.pow(2.0), {"lo": 0.01, "hi": 10.0}),
        ("pow_cube bwd",    FLOAT, S2D, lambda t: t.pow(3.0), {"lo": 0.01, "hi": 10.0}),
        ("pow_tensor bwd",  FLOAT, S2D, lambda t: torch.pow(t, t), {"lo": 0.5, "hi": 2.0}),
    ]
 
    has_any = False
    for entry in POW_BWD:
        name, cat, shapes, fn = entry[0], entry[1], entry[2], entry[3]
        kw = entry[4] if len(entry) > 4 else None
        if not dtype_allowed(runner.dtype_str, cat):
            continue
        if runner._should_run(name):
            if not has_any:
                print_header("Backward: Pow")
                has_any = True
            runner.bench_backward(name, shapes, fn, data_kw=kw)
 
    # -- Matmul backward --
    S_MM = [
        (64, 64), (128, 128), (256, 256),
    ]
 
    has_any = False
    for name, cat, shapes, fn in [
        ("mm bwd", FLOAT, S_MM, lambda t: torch.mm(t, t.transpose(0, 1))),
    ]:
        if not dtype_allowed(runner.dtype_str, cat):
            continue
        if runner._should_run(name):
            if not has_any:
                print_header("Backward: Matmul")
                has_any = True
            runner.bench_backward(name, shapes, fn)
 
    # -- Shape ops backward --
    SHAPE_BWD = [
        ("reshape bwd",   FLOAT, S2D, lambda t: t.reshape(-1)),
        ("transpose bwd", FLOAT, S2D, lambda t: t.transpose(0, 1)),
    ]
 
    has_any = False
    for name, cat, shapes, fn in SHAPE_BWD:
        if not dtype_allowed(runner.dtype_str, cat):
            continue
        if runner._should_run(name):
            if not has_any:
                print_header("Backward: Shape ops")
                has_any = True
            runner.bench_backward(name, shapes, fn)
 
    # -- Composed backward --
    COMPOSED_BWD = [
        ("chain (a*b+c).sum() bwd", FLOAT, S2D, lambda t: (t * t + t).sum()),
        ("sigmoid(mm) bwd",         FLOAT, S_MM, lambda t: torch.mm(t, t.T).sigmoid().sum()),
    ]
 
    has_any = False
    for name, cat, shapes, fn in COMPOSED_BWD:
        if not dtype_allowed(runner.dtype_str, cat):
            continue
        if runner._should_run(name):
            if not has_any:
                print_header("Backward: Composed")
                has_any = True
            runner.bench_backward(name, shapes, fn)


def _run_section(runner, title, benchmarks):
    """Run a list of single-tensor benchmarks, printing the header only if any match."""
    header_printed = False

    for entry in benchmarks:
        name, cat, shapes = entry[0], entry[1], entry[2]
        raxis_fn, torch_fn = entry[3], entry[4]
        numpy_fn = entry[5] if len(entry) > 5 else None
        data_kw = entry[6] if len(entry) > 6 else None

        if isinstance(numpy_fn, dict):
            data_kw = numpy_fn
            numpy_fn = None

        if not dtype_allowed(runner.dtype_str, cat):
            continue

        if not runner._should_run(name):
            continue

        if not header_printed:
            print_header(title)
            header_printed = True

        runner.bench(name, shapes, raxis_fn, torch_fn, numpy_fn=numpy_fn,
                     data_kw=data_kw)


def _run_section_pair(runner, title, benchmarks):
    header_printed = False

    for entry in benchmarks:
        name, cat, pairs = entry[0], entry[1], entry[2]
        raxis_fn, torch_fn = entry[3], entry[4]
        numpy_fn = entry[5] if len(entry) > 5 else None

        if not dtype_allowed(runner.dtype_str, cat):
            continue

        if not runner._should_run(name):
            continue

        if not header_printed:
            print_header(title)
            header_printed = True

        runner.bench_pair(name, pairs, raxis_fn, torch_fn, numpy_fn=numpy_fn)


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
    parser.add_argument("--threshold", type=float, default=2)
    parser.add_argument("-k", "--filter", dest="target_ops", nargs='+', default=None)
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--no-numpy", action="store_true", help="Skip NumPy benchmarks")
    parser.add_argument("--forward-only", action="store_true",
                        help="Run only forward benchmarks")
    parser.add_argument("--backward-only", action="store_true",
                        help="Run only backward benchmarks")
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
    print(f"Raxis: {raxis.__version__} | PyTorch: {torch.__version__} | Numpy: {np.__version__}")
    print(dim(f"Warmup: {args.warmup}  Repeats: {args.repeats}  "
              f"Threshold: {args.threshold}x"))

    runner = BenchmarkRunner(
        device=args.device,
        dtype=args.dtype,
        warmup=args.warmup,
        repeats=args.repeats,
        threshold=args.threshold,
        target_ops=args.target_ops,
        no_numpy=args.no_numpy,
        forward_only=args.forward_only,
        backward_only=args.backward_only,
    )

    run_all(runner)
    runner.summary()


if __name__ == "__main__":
    main()