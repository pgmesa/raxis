"""synapgrad's windows + tensordot vs im2col + matmul on NumPy, forward and backward apart."""
import sys, time
import numpy as np
from synapgrad import cpu_ops, conv_tools
rng = np.random.default_rng(0)
K, P = (3, 3), (1, 1)
data = []
for shape, c_out in (((128, 1, 28, 28), 16), ((128, 16, 14, 14), 32)):
    x = rng.random(shape, dtype=np.float32)
    w = (rng.random((c_out, shape[1], 3, 3), dtype=np.float32) - 0.5) * 0.2
    g = rng.random((shape[0], c_out, shape[2], shape[3]), dtype=np.float32)
    data.append((x, w, g))
def med(f, n=30):
    for _ in range(3): f()
    s = []
    for _ in range(n):
        t = time.perf_counter(); f(); s.append(time.perf_counter() - t)
    return 1e3 * float(np.median(s))
def win_fw():
    for x, w, g in data: cpu_ops.conv2d_forward(x, w, None, 1, 1, 1)
def col_fw():
    for x, w, g in data:
        N, c, H, W = g.shape
        (w.reshape(c, -1) @ conv_tools.im2col_fast(x, K, (1, 1), (1, 1), P, as_unfold=True)).reshape(N, c, H, W)
win_saved = [cpu_ops.conv2d_forward(x, w, None, 1, 1, 1)[1] for x, w, g in data]
col_saved = [conv_tools.im2col_fast(x, K, (1, 1), (1, 1), P, as_unfold=True) for x, w, g in data]
def win_bw():
    for (x, w, g), wd in zip(data, win_saved): cpu_ops.conv2d_backward(g, x.shape, w, None, 1, 1, 1, wd)
def col_bw():
    for (x, w, g), cols in zip(data, col_saved):
        N, c, H, W = g.shape
        wf, gf = w.reshape(c, -1), g.reshape(N, c, -1)
        conv_tools.col2im_fast(wf.T @ gf, x.shape, K, (1, 1), (1, 1), P)
        (gf @ cols.transpose(0, 2, 1)).sum(axis=0)
print(f"windows fw {med(win_fw):.1f} bw {med(win_bw):.1f} | im2col fw {med(col_fw):.1f} bw {med(col_bw):.1f}")
