"""Where the time of each conv algorithm goes, step by step, on NumPy and on torch in Python.

The second conv layer of the MNIST CNN (the expensive one), batch 128, forward and
backward. Every number is the median of the step on its own.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from synapgrad import conv_tools

SHAPE, C_OUT, K, P = (128, 16, 14, 14), 32, 3, 1
rng = np.random.default_rng(0)
x = rng.random(SHAPE, dtype=np.float32)
w = (rng.random((C_OUT, SHAPE[1], K, K), dtype=np.float32) - 0.5) * 0.2
grad = rng.random((SHAPE[0], C_OUT, SHAPE[2], SHAPE[3]), dtype=np.float32)
tx, tw, tgrad = torch.from_numpy(x), torch.from_numpy(w), torch.from_numpy(grad)
N, H, W = SHAPE[0], SHAPE[2], SHAPE[3]
kk, pp, oo = (K, K), (P, P), (1, 1)


def ms(step, n=41):
    for _ in range(3):
        step()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        step()
        samples.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(samples))


def report(title, rows):
    print(title)
    total = 0.0
    for name, step in rows:
        t = ms(step)
        total += t
        print(f"  {name:44s}{t:8.3f} ms")
    print(f"  {'total':44s}{total:8.3f} ms")


# Sliding windows + tensordot
win_n = conv_tools.extract_windows(x, kk, oo, pp, oo)
gw_n = np.moveaxis(np.tensordot(grad, w, axes=[[1], [0]]), 0, 2)
report("numpy windows", [
    ("extract windows (pad, strided view)", lambda: conv_tools.extract_windows(x, kk, oo, pp, oo)),
    ("forward tensordot", lambda: np.tensordot(w, win_n, axes=[(1, 2, 3), (3, 4, 5)])),
    ("backward tensordot, input", lambda: np.tensordot(grad, w, axes=[[1], [0]])),
    ("place_windows, a loop per window", lambda: conv_tools.place_windows(gw_n, SHAPE, kk, oo, pp, oo)),
    ("backward tensordot, weight", lambda: np.tensordot(grad, win_n, axes=[(2, 3, 0), (0, 1, 2)])),
])


def t_extract():
    return F.pad(tx, (P, P, P, P)).unfold(2, K, 1).unfold(3, K, 1).permute(2, 3, 0, 1, 4, 5)


def t_place(windows):
    output = torch.zeros(N, SHAPE[1], H + 2 * P, W + 2 * P)
    windows = windows.movedim(2, 0)
    for i in range(windows.shape[1]):
        for j in range(windows.shape[2]):
            output[..., i:i + K, j:j + K] += windows[:, i, j]
    return output[..., P:H + P, P:W + P]


win_t = t_extract()
gw_t = torch.tensordot(tgrad, tw, dims=([1], [0])).movedim(0, 2)
report("torch windows, in Python", [
    ("extract windows (pad, unfold views)", t_extract),
    ("forward tensordot", lambda: torch.tensordot(tw, win_t, dims=([1, 2, 3], [3, 4, 5]))),
    ("backward tensordot, input", lambda: torch.tensordot(tgrad, tw, dims=([1], [0]))),
    ("place_windows, a loop per window", lambda: t_place(gw_t)),
    ("backward tensordot, weight", lambda: torch.tensordot(tgrad, win_t, dims=([2, 3, 0], [0, 1, 2]))),
])

# im2col + matmul
cols_n = conv_tools.im2col_fast(x, kk, oo, oo, pp, as_unfold=True)
wf_n = w.reshape(C_OUT, -1)
gf_n = grad.reshape(N, C_OUT, -1)
back_n = wf_n.T @ gf_n
cols_tr_n = cols_n.transpose(0, 2, 1)
report("numpy im2col", [
    ("im2col (strided view + reshape copy)", lambda: conv_tools.im2col_fast(x, kk, oo, oo, pp, as_unfold=True)),
    ("forward: one matrix @ batch", lambda: wf_n @ cols_n),
    ("backward input: one matrix @ batch", lambda: wf_n.T @ gf_n),
    ("col2im", lambda: conv_tools.col2im_fast(back_n, SHAPE, kk, oo, oo, pp)),
    ("backward weight: batch @ batch + sum", lambda: (gf_n @ cols_tr_n).sum(axis=0)),
])

cols_t = F.unfold(tx, K, padding=P)
wf_t = tw.reshape(C_OUT, -1)
gf_t = tgrad.reshape(N, C_OUT, -1)
back_t = wf_t.T @ gf_t
cols_tr_t = cols_t.transpose(1, 2)
report("torch im2col, in Python", [
    ("F.unfold", lambda: F.unfold(tx, K, padding=P)),
    ("forward: one matrix @ batch", lambda: wf_t @ cols_t),
    ("backward input: one matrix @ batch", lambda: wf_t.T @ gf_t),
    ("F.fold", lambda: F.fold(back_t, (H, W), K, padding=P)),
    ("backward weight: batch @ batch + sum", lambda: (gf_t @ cols_tr_t).sum(dim=0)),
])
