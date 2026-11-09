"""How a backward is computed (Rebuilding PyTorch, the autograd article).

1. The graph of loss = (a * b + a).sum() in PyTorch and Raxis: the nodes,
   their edges, their sequence numbers and the gradients.
2. The cost of recording: a * b on 16 floats with and without a tensor
   that requires grad, and the forward and backward of a chain of 500
   multiplications, per node, next to a small micrograd-style engine of
   scalars written here.
3. Depth: the recursive topological sort of micrograd against chains of
   growing length, and PyTorch and Raxis on a chain of 100,000 nodes.
4. Memory on the GPU: eight sin() on a 64 MB tensor, what is allocated
   after the forward, at the peak of the backward and after it, with and
   without retain_graph.
5. A reference cycle: an autograd.Function that keeps its output in an
   attribute of ctx instead of save_for_backward.

    uv run python scripts/articles/autograd.py
"""

import gc
import statistics
import sys
import time

import torch

import raxis

LIBRARIES = [("PyTorch", torch), ("Raxis", raxis)]


def us(f, repeat=200, batch=100):
    # Batches of calls: on Windows perf_counter ticks every 100 ns, as long
    # as one call
    for _ in range(50):
        f()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(batch):
            f()
        times.append(time.perf_counter() - t)
    return statistics.median(times) / batch * 1e6


class Value:
    """A scalar with a gradient, as in micrograd: each result keeps its
    inputs and a closure that adds its gradient into theirs."""

    def __init__(self, data, children=()):
        self.data = data
        self.grad = 0.0
        self._prev = children
        self._backward = lambda: None

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def backward(self):
        order, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                order.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


def graph():
    print("loss = (a * b + a).sum(), a = [2, 3], b = [4, 5]")
    for name, lib in LIBRARIES:
        a = lib.tensor([2.0, 3.0], requires_grad=True)
        b = lib.tensor([4.0, 5.0], requires_grad=True)
        loss = (a * b + a).sum()
        print(f"\n{name}")
        stack, seen = [(loss.grad_fn, 0)], set()
        while stack:
            node, depth = stack.pop()
            label = type(node).__name__ if lib is torch else repr(node).strip("<>")
            seq = node._sequence_nr() if hasattr(node, "_sequence_nr") else ""
            edges = [(type(n).__name__ if lib is torch else repr(n).strip("<>"), nr) for n, nr in node.next_functions]
            print(f"{'  ' * depth}{label:<16s} seq {seq:<22} edges {edges}")
            if id(node) in seen:
                continue
            seen.add(id(node))
            for n, _ in reversed(node.next_functions):
                if n is not None and type(n).__name__ != "AccumulateGrad" and "AccumulateGrad" not in repr(n):
                    stack.append((n, depth + 1))
        loss.backward()
        print(f"a.grad {a.grad.tolist()}  b.grad {b.grad.tolist()}")


def record_cost():
    print("\nRecording a node, us per call (16 floats)")
    print(f"{'':>34s}{'PyTorch':>10s}{'Raxis':>10s}")
    rows = {}
    for _, lib in LIBRARIES:
        a, b = lib.rand(16), lib.rand(16)
        w = lib.rand(16).requires_grad_()
        rows.setdefault("a * b, nothing requires grad", []).append(us(lambda: a * b))
        rows.setdefault("w * b, w requires grad", []).append(us(lambda: w * b))
        with lib.no_grad():
            rows.setdefault("w * b under no_grad", []).append(us(lambda: w * b))
    for label, cells in rows.items():
        print(f"{label:>34s}" + "".join(f"{c:10.2f}" for c in cells))

    n = 500
    print(f"\nA chain of {n} multiplications, us per node")
    print(f"{'':>12s}{'forward':>10s}{'backward':>10s}")

    def micrograd_chain():
        x = Value(1.0)
        y = x
        c = Value(1.0001)
        t0 = time.perf_counter()
        for _ in range(n):
            y = y * c
        t1 = time.perf_counter()
        y.backward()
        t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    def tensor_chain(lib):
        def run():
            x = lib.rand(16).requires_grad_()
            c = lib.full([16], 1.0001)
            y = x
            t0 = time.perf_counter()
            for _ in range(n):
                y = y * c
            s = y.sum()
            t1 = time.perf_counter()
            s.backward()
            t2 = time.perf_counter()
            return t1 - t0, t2 - t1
        return run

    for label, run in [("micrograd", micrograd_chain)] + [(name, tensor_chain(lib)) for name, lib in LIBRARIES]:
        for _ in range(5):
            run()
        samples = [run() for _ in range(50)]
        fwd = statistics.median(s[0] for s in samples) / n * 1e6
        bwd = statistics.median(s[1] for s in samples) / n * 1e6
        print(f"{label:>12s}{fwd:10.2f}{bwd:10.2f}")


def depth():
    print(f"\nDepth: micrograd's recursive sort (Python's recursion limit is {sys.getrecursionlimit()})")
    for n in (500, 990, 1000, 2000):
        y = Value(1.0)
        for _ in range(n):
            y = y * Value(1.0)
        try:
            y.backward()
            result = "ok"
        except RecursionError:
            result = "RecursionError"
        print(f"{n:>10d} nodes  {result}")

    n = 100_000
    print(f"\nA chain of {n:,} multiplications: backward and freeing the graph, ms")
    for name, lib in LIBRARIES:
        x = lib.rand(16).requires_grad_()
        c = lib.full([16], 1.0)
        y = x
        for _ in range(n):
            y = y * c
        s = y.sum()
        t0 = time.perf_counter()
        s.backward(retain_graph=True)
        t1 = time.perf_counter()
        del y, s
        t2 = time.perf_counter()
        print(f"{name:>10s}  backward {(t1 - t0) * 1e3:8.1f}  free {(t2 - t1) * 1e3:8.1f}")


def gpu_memory():
    if not (torch.cuda.is_available() and raxis.cuda.is_available()):
        print("\nNo GPU: skipping the memory experiment")
        return
    torch.cuda.init()
    mb = 2**20
    print("\nGPU memory of x -> sin -> ... -> sin (8 times) -> sum, x of 64 MB, MB")
    print(f"{'':>10s}{'':>16s}{'forward':>9s}{'peak':>9s}{'after':>9s}")
    for name, lib in LIBRARIES:
        for retain in (False, True):
            gc.collect()
            lib.cuda.synchronize()
            base = lib.cuda.memory_allocated()
            x = lib.randn(4096, 4096, device="cuda").requires_grad_()
            y = x
            for _ in range(8):
                y = y.sin()
            s = y.sum()
            del y
            lib.cuda.synchronize()
            forward = lib.cuda.memory_allocated() - base
            lib.cuda.reset_peak_memory_stats()
            s.backward(retain_graph=retain)
            lib.cuda.synchronize()
            peak = lib.cuda.max_memory_allocated() - base
            after = lib.cuda.memory_allocated() - base
            label = "retain_graph" if retain else "default"
            print(f"{name:>10s}{label:>16s}{forward / mb:9.0f}{peak / mb:9.0f}{after / mb:9.0f}")
            del x, s


def cycle():
    if not (torch.cuda.is_available() and raxis.cuda.is_available()):
        return
    mb = 2**20
    print("\nMemory left on the GPU after 10 steps of an autograd.Function, x of 4 MB, MB")
    print(f"{'':>10s}{'':>22s}{'no GC':>8s}{'after gc':>10s}")
    gc.disable()
    try:
        for name, lib in LIBRARIES:
            class SavedOnCtx(lib.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    y = x.exp()
                    ctx.y = y
                    return y

                @staticmethod
                def backward(ctx, grad):
                    return grad * ctx.y

            class Saved(lib.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    y = x.exp()
                    ctx.save_for_backward(y)
                    return y

                @staticmethod
                def backward(ctx, grad):
                    (y,) = ctx.saved_tensors
                    return grad * y

            for label, function in [("ctx.y = y", SavedOnCtx), ("save_for_backward(y)", Saved)]:
                x = lib.randn(1024, 1024, device="cuda").requires_grad_()
                function.apply(x).sum().backward()
                gc.collect()
                base = lib.cuda.memory_allocated()
                for _ in range(10):
                    function.apply(x).sum().backward()
                grown = lib.cuda.memory_allocated() - base
                gc.collect()
                left = lib.cuda.memory_allocated() - base
                print(f"{name:>10s}{label:>22s}{grown / mb:8.0f}{left / mb:10.0f}")
                del x
                gc.collect()
    finally:
        gc.enable()


def main():
    torch.set_num_threads(1)
    raxis.set_num_threads(1)
    graph()
    record_cost()
    depth()
    gpu_memory()
    cycle()


if __name__ == "__main__":
    main()
