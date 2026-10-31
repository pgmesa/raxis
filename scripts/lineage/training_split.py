"""Split a training step between tensor operations and autograd + bindings.

full: the library's own model, autograd and optimizer.
ops: the same MLP with the backward written by hand, so only the tensor
operations run (no graph, no engine, no nn.Module).
"""
import cProfile
import pstats
import sys
import time

import numpy as np

B_MLP, B_CNN = 64, 128
rng = np.random.default_rng(0)
X = rng.random((B_MLP, 28, 28), dtype=np.float32)
Y = rng.integers(0, 10, size=B_MLP)
X_C = rng.random((B_CNN, 1, 28, 28), dtype=np.float32)
Y_C = rng.integers(0, 10, size=B_CNN)
X_T = rng.random((4, 2), dtype=np.float32)
Y_T = rng.random((4, 1), dtype=np.float32)


def median_ms(step, warmup, n):
    for _ in range(warmup):
        step()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        step()
        samples.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(samples))


def init_params():
    r = np.random.default_rng(1)
    params = []
    for n_in, n_out in ((784, 200), (200, 100), (100, 10)):
        params.append(((r.random((n_out, n_in), dtype=np.float32) - 0.5) * 0.1))
        params.append(((r.random(n_out, dtype=np.float32) - 0.5) * 0.1))
    return params


def numpy_ops(p_drop=0.2):
    W1, b1, W2, b2, W3, b3 = params = init_params()
    vel = [np.zeros_like(p) for p in params]
    x = X.reshape(B_MLP, -1)
    idx = np.arange(B_MLP)
    r = np.random.default_rng(2)

    def step():
        h1 = x @ W1.T + b1
        r1 = np.maximum(h1, 0)
        mask = (r.random(r1.shape, dtype=np.float32) >= p_drop) / (1 - p_drop)
        d1 = r1 * mask
        h2 = d1 @ W2.T + b2
        r2 = np.maximum(h2, 0)
        h3 = r2 @ W3.T + b3
        shifted = h3 - h3.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        total = exp.sum(axis=1, keepdims=True)
        logp = shifted - np.log(total)
        loss = -logp[idx, Y].mean()

        g = np.zeros_like(logp)
        g[idx, Y] = -1.0 / B_MLP
        g3 = g - (exp / total) * g.sum(axis=1, keepdims=True)
        grads = [None] * 6
        grads[4], grads[5] = g3.T @ r2, g3.sum(axis=0)
        g2 = (g3 @ W3) * (h2 > 0)
        grads[2], grads[3] = g2.T @ d1, g2.sum(axis=0)
        g1 = (g2 @ W2) * mask * (h1 > 0)
        grads[0], grads[1] = g1.T @ x, g1.sum(axis=0)
        for p, v, grad in zip(params, vel, grads):
            v *= 0.9
            v += grad
            p -= 0.01 * v
        return loss, grads

    return step


def torch_ops(p_drop=0.2, params=None):
    import torch

    params = params if params is not None else [torch.from_numpy(p) for p in init_params()]
    W1, b1, W2, b2, W3, b3 = params
    vel = [torch.zeros_like(p) for p in params]
    x = torch.from_numpy(X.reshape(B_MLP, -1))
    y = torch.from_numpy(Y)
    idx = torch.arange(B_MLP)

    @torch.no_grad()
    def step():
        h1 = x @ W1.T + b1
        r1 = h1.relu()
        mask = (torch.rand_like(r1) >= p_drop).float() / (1 - p_drop)
        d1 = r1 * mask
        h2 = d1 @ W2.T + b2
        r2 = h2.relu()
        h3 = r2 @ W3.T + b3
        logp = h3.log_softmax(dim=1)
        loss = -logp[idx, y].mean()

        g = torch.zeros_like(logp)
        g[idx, y] = -1.0 / B_MLP
        g3 = g - logp.exp() * g.sum(dim=1, keepdim=True)
        grads = [None] * 6
        grads[4], grads[5] = g3.T @ r2, g3.sum(dim=0)
        g2 = (g3 @ W3) * (h2 > 0)
        grads[2], grads[3] = g2.T @ d1, g2.sum(dim=0)
        g1 = (g2 @ W2) * mask * (h1 > 0)
        grads[0], grads[1] = g1.T @ x, g1.sum(dim=0)
        for p, v, grad in zip(params, vel, grads):
            v.mul_(0.9).add_(grad)
            p.sub_(0.01 * v)
        return loss, grads

    return step


def check_manual_backward():
    """The hand-written backward has to match torch's autograd."""
    import torch
    import torch.nn.functional as F

    leaves = [torch.from_numpy(p).requires_grad_() for p in init_params()]
    W1, b1, W2, b2, W3, b3 = leaves
    x = torch.from_numpy(X.reshape(B_MLP, -1))
    y = torch.from_numpy(Y)
    h = (x @ W1.T + b1).relu()
    h = (h @ W2.T + b2).relu()
    F.nll_loss((h @ W3.T + b3).log_softmax(dim=1), y).backward()
    _, grads_t = torch_ops(0.0, [p.detach().clone() for p in leaves])()
    _, grads_n = numpy_ops(0.0)()
    for leaf, gt, gn in zip(leaves, grads_t, grads_n):
        assert np.allclose(leaf.grad.numpy(), gt.numpy(), atol=1e-5), "torch manual"
        assert np.allclose(leaf.grad.numpy(), gn, atol=1e-5), "numpy manual"
    print("manual backward matches autograd")


def full_step(name, kind):
    if name == "torch":
        import torch as lib
        tensor = lambda a, long=False: lib.tensor(a, dtype=lib.long) if long else lib.tensor(a)
    elif name == "raxis":
        import raxis as lib
        tensor = lambda a, long=False: lib.tensor(a, dtype=lib.long) if long else lib.tensor(a)
    elif name == "synapx":
        import synapx as lib
        import torch
        tensor = lambda a, long=False: lib.tensor(a, dtype=torch.long) if long else lib.tensor(a)
    else:
        import synapgrad as lib
        tensor = lambda a, long=False: lib.Tensor(a.astype(np.int64) if long else a)
    nn = lib.nn
    if kind == "tiny":
        model = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = lib.optim.SGD(list(model.parameters()), lr=0.1)
        loss_fn, x, y = nn.MSELoss(), tensor(X_T), tensor(Y_T)
    elif kind == "mlp":
        model = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 200), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(200, 100), nn.ReLU(), nn.Linear(100, 10), nn.LogSoftmax(dim=1))
        opt = lib.optim.SGD(list(model.parameters()), lr=0.01, momentum=0.9)
        loss_fn, x, y = nn.NLLLoss(), tensor(X), tensor(Y, True)
    else:
        def block(c_in, c_out):
            return [nn.Conv2d(c_in, c_out, 3, padding="same", bias=False), nn.BatchNorm2d(c_out),
                    nn.ReLU(), nn.Dropout(0.3), nn.MaxPool2d(2)]
        model = nn.Sequential(*block(1, 16), *block(16, 32), nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
        opt = lib.optim.Adam(list(model.parameters()), lr=0.01)
        loss_fn, x, y = nn.CrossEntropyLoss(), tensor(X_C), tensor(Y_C, True)

    def step():
        loss = loss_fn(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return step


what = sys.argv[1]
if what == "check":
    check_manual_backward()
elif what == "numpy_ops":
    print(f"{what}: {median_ms(numpy_ops(), 20, 500):.3f} ms")
elif what == "torch_ops":
    print(f"{what}: {median_ms(torch_ops(), 20, 500):.3f} ms")
elif what == "profile":
    step = full_step("synapgrad", "cnn")
    step()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(5):
        step()
    profiler.disable()
    stats = pstats.Stats(profiler)
    groups = {}
    for (path, _, func), (_, _, tottime, _, _) in stats.stats.items():
        path = path.replace("\\", "/")
        if path == "~" or "site-packages/numpy" in path:
            group = "numpy"
        elif path.endswith(("cpu_ops.py", "conv_tools.py")):
            group = "synapgrad kernels (cpu_ops, conv_tools)"
        elif "synapgrad" in path:
            group = "synapgrad graph, nn, optim"
        else:
            group = "other"
        groups[group] = groups.get(group, 0.0) + tottime
    total = sum(groups.values())
    for group, seconds in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"{group}: {1e3 * seconds / 5:.1f} ms per step ({100 * seconds / total:.0f}%)")
    print(f"primitive calls per step: {stats.prim_calls // 5}")
else:
    name, kind = what.split(":")
    warmup, n = (3, 30) if kind == "cnn" else (20, 500)
    print(f"{what}: {median_ms(full_step(name, kind), warmup, n):.3f} ms")
