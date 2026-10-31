"""Conv algorithms on NumPy and libtorch, and synapgrad's CNN step without its graph.

conv: forward + backward (input and weight grads) of the two conv layers of the
MNIST CNN, batch 128, with each algorithm.
ops: synapgrad's CNN step calling its kernels by hand, without Tensor, graph or
nn.Module.
"""
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import synapgrad
from synapgrad import cpu_ops, conv_tools

LAYERS = [((128, 1, 28, 28), 16), ((128, 16, 14, 14), 32)]
K, P = 3, 1
rng = np.random.default_rng(0)


def median_ms(step, warmup, n):
    for _ in range(warmup):
        step()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        step()
        samples.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(samples))


def layer_data(shape, c_out):
    x = rng.random(shape, dtype=np.float32)
    w = (rng.random((c_out, shape[1], K, K), dtype=np.float32) - 0.5) * 0.2
    grad = rng.random((shape[0], c_out, shape[2], shape[3]), dtype=np.float32)
    return x, w, grad


def numpy_windows():
    data = [layer_data(*layer) for layer in LAYERS]

    def step():
        for x, w, grad in data:
            _, windows = cpu_ops.conv2d_forward(x, w, None, 1, P, 1)
            cpu_ops.conv2d_backward(grad, x.shape, w, None, 1, P, 1, windows)

    return step


def numpy_im2col():
    data = [layer_data(*layer) for layer in LAYERS]

    def step():
        for x, w, grad in data:
            N, c_out, H, W = grad.shape
            cols = conv_tools.im2col_fast(x, (K, K), (1, 1), (1, 1), (P, P), as_unfold=True)
            w_flat = w.reshape(c_out, -1)
            (w_flat @ cols).reshape(N, c_out, H, W)
            g_flat = grad.reshape(N, c_out, -1)
            conv_tools.col2im_fast(w_flat.T @ g_flat, x.shape, (K, K), (1, 1), (1, 1), (P, P))
            (g_flat @ cols.transpose(0, 2, 1)).sum(axis=0).reshape(w.shape)

    return step


def libtorch_windows():
    import synapx

    data = []
    for layer in LAYERS:
        x, w, grad = layer_data(*layer)
        data.append((synapx.tensor(torch.from_numpy(x), requires_grad=True),
                     synapx.tensor(torch.from_numpy(w), requires_grad=True),
                     synapx.tensor(torch.from_numpy(grad))))

    def step():
        for x, w, grad in data:
            synapx.nn.functional.conv2d(x, w, None, 1, P, 1).backward(grad)

    return step


def libtorch_im2col():
    data = [tuple(torch.from_numpy(a) for a in layer_data(*layer)) for layer in LAYERS]

    @torch.no_grad()
    def step():
        for x, w, grad in data:
            N, c_out, H, W = grad.shape
            cols = F.unfold(x, K, padding=P)
            w_flat = w.reshape(c_out, -1)
            (w_flat @ cols).reshape(N, c_out, H, W)
            g_flat = grad.reshape(N, c_out, -1)
            F.fold(w_flat.T @ g_flat, (x.shape[2], x.shape[3]), K, padding=P)
            (g_flat @ cols.transpose(1, 2)).sum(dim=0).reshape(w.shape)

    return step


def libtorch_native():
    data = []
    for layer in LAYERS:
        x, w, grad = [torch.from_numpy(a) for a in layer_data(*layer)]
        data.append((x.requires_grad_(), w.requires_grad_(), grad))

    def step():
        for x, w, grad in data:
            F.conv2d(x, w, padding=P).backward(grad)
            x.grad = None
            w.grad = None

    return step


def check_im2col_matches_windows():
    x, w, grad = layer_data(*LAYERS[1])
    N, c_out, H, W = grad.shape
    out_w, windows = cpu_ops.conv2d_forward(x, w, None, 1, P, 1)
    x_g, w_g, _ = cpu_ops.conv2d_backward(grad, x.shape, w, None, 1, P, 1, windows)
    cols = conv_tools.im2col_fast(x, (K, K), (1, 1), (1, 1), (P, P), as_unfold=True)
    w_flat = w.reshape(c_out, -1)
    g_flat = grad.reshape(N, c_out, -1)
    assert np.allclose((w_flat @ cols).reshape(N, c_out, H, W), out_w, atol=1e-4)
    assert np.allclose(conv_tools.col2im_fast(w_flat.T @ g_flat, x.shape, (K, K), (1, 1), (1, 1), (P, P)), x_g, atol=1e-3)
    assert np.allclose((g_flat @ cols.transpose(0, 2, 1)).sum(axis=0).reshape(w.shape), w_g, atol=1e-2)
    print("im2col matches windows")


def numpy_cnn_ops():
    """synapgrad's CNN step: same kernels, dropout and Adam, no graph."""
    X = rng.random((128, 1, 28, 28), dtype=np.float32)
    y = rng.integers(0, 10, size=128)
    r = np.random.default_rng(1)

    def uniform(*shape):
        return (r.random(shape, dtype=np.float32) - 0.5) * 0.5

    w1, w2 = uniform(16, 1, 3, 3), uniform(32, 16, 3, 3)
    g1, b1, g2, b2 = np.ones(16, np.float32), np.zeros(16, np.float32), np.ones(32, np.float32), np.zeros(32, np.float32)
    rm1, rv1, rm2, rv2 = np.zeros(16, np.float32), np.ones(16, np.float32), np.zeros(32, np.float32), np.ones(32, np.float32)
    w3, b3 = uniform(10, 32 * 7 * 7), uniform(10)
    params = [w1, g1, b1, w2, g2, b2, w3, b3]
    m1, m2 = [0] * len(params), [0] * len(params)
    t = [0]

    def dropout(x):
        mask = np.where(np.random.rand(*x.shape) <= 0.3, 0, 1) / 0.7
        return x * mask.astype(np.float32), mask.astype(np.float32)

    def block_fw(x, w, g, b, rm, rv):
        c, windows = cpu_ops.conv2d_forward(x, w, None, 1, P, 1)
        n, rm, rv, mean, var = cpu_ops.batch_norm_forward(c, g, b, rm, rv, True, 0.1, 1e-5)
        a = cpu_ops.relu_forward(n)
        d, mask = dropout(a)
        p, a_shape, p_windows = cpu_ops.max_pool2d_forward(d, (2, 2), (2, 2), 0, 1)
        return p, (x, c, n, windows, mean, var, mask, a_shape, p_windows)

    def block_bw(grad, w, g, b, state):
        x, c, n, windows, mean, var, mask, a_shape, p_windows = state
        d_g = cpu_ops.max_pool2d_backward(grad, (2, 2), (2, 2), 0, 1, a_shape, p_windows)
        a_g = d_g * mask
        n_g = cpu_ops.relu_backward(a_g, n)
        c_g, g_g, b_g = cpu_ops.batch_norm_backward(n_g, c, g, b, True, True, 1e-5, mean, var)
        x_g, w_g, _ = cpu_ops.conv2d_backward(c_g, x.shape, w, None, 1, P, 1, windows)
        return x_g, w_g, g_g, b_g

    def step():
        p1, s1 = block_fw(X, w1, g1, b1, rm1, rv1)
        p2, s2 = block_fw(p1, w2, g2, b2, rm2, rv2)
        flat = p2.reshape(128, -1)
        logits = flat @ w3.T + b3
        cpu_ops.cross_entropy_loss_forward(logits, y).mean()

        l_g = cpu_ops.cross_entropy_loss_backward(np.full((128, 1), 1 / 128, np.float32), logits, y)
        w3_g, b3_g, flat_g = l_g.T @ flat, l_g.sum(axis=0), l_g @ w3
        p1_g, w2_g, g2_g, b2_g = block_bw(flat_g.reshape(p2.shape), w2, g2, b2, s2)
        _, w1_g, g1_g, b1_g = block_bw(p1_g, w1, g1, b1, s1)

        t[0] += 1
        for i, (p, grad) in enumerate(zip(params, (w1_g, g1_g, b1_g, w2_g, g2_g, b2_g, w3_g, b3_g))):
            m1[i] = 0.9 * m1[i] + 0.1 * grad
            m2[i] = 0.999 * m2[i] + 0.001 * grad**2.0
            c1 = m1[i] / (1.0 - 0.9 ** t[0])
            c2 = m2[i] / (1.0 - 0.999 ** t[0])
            p -= (0.01 * c1 / (np.sqrt(c2) + 1e-8)).astype(p.dtype)

    return step


what = sys.argv[1]
if what == "check":
    check_im2col_matches_windows()
elif what == "ops":
    print(f"numpy_cnn_ops: {median_ms(numpy_cnn_ops(), 3, 30):.2f} ms")
else:
    print(f"{what}: {median_ms(globals()[what](), 3, 30):.2f} ms")
