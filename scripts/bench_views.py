"""
Cost of a view against the cost of a copy, at several tensor sizes: raxis vs PyTorch.

A view only writes a new shape, strides and offset, so its time should not depend
on the size of the tensor. contiguous() on a transposed tensor copies every element.

Usage:
    python bench_views.py
"""

import time

import numpy as np
import torch
import raxis

SIDES = (32, 1000, 10000)
OPS = {
    "x[1:3]": lambda x: x[1:3],
    "x[:, 1]": lambda x: x[:, 1],
    "x.transpose(0, 1)": lambda x: x.transpose(0, 1),
    "x[:, :1].expand(n, n)": lambda x: x[:, :1].expand(x.shape[0], x.shape[0]),
    "x.reshape(-1)": lambda x: x.reshape(-1),
    "x.contiguous() (no copy)": lambda x: x.contiguous(),
    "x.clone()": lambda x: x.clone(),
}


def median_us(fn, x, budget=0.5):
    fn(x)
    samples = []
    start = time.perf_counter()
    while len(samples) < 5 or (time.perf_counter() - start < budget and len(samples) < 20000):
        t0 = time.perf_counter()
        fn(x)
        samples.append(time.perf_counter() - t0)
    return 1e6 * float(np.median(samples))


def show(us):
    return f"{us:10.2f} us" if us < 1000 else f"{us / 1000:10.2f} ms"


print(f"{'op':28} {'side':>6} {'raxis':>13} {'torch':>13}")
for side in SIDES:
    a = np.random.default_rng(0).random((side, side), dtype=np.float32)
    x_r, x_t = raxis.tensor(a), torch.from_numpy(a.copy())
    ops = dict(OPS)
    for name, op in ops.items():
        print(f"{name:28} {side:6} {show(median_us(op, x_r)):>13} {show(median_us(op, x_t)):>13}")
    # A transposed square tensor: the copy has to walk the elements out of order
    t_r, t_t = x_r.transpose(0, 1), x_t.transpose(0, 1)
    name = "x.transpose(0, 1).contiguous()"
    print(f"{name:28} {side:6} {show(median_us(lambda t: t.contiguous(), t_r)):>13} {show(median_us(lambda t: t.contiguous(), t_t)):>13}")
    print()
