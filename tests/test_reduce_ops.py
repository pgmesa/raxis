import pytest
import raxis
import torch
import numpy as np

from test_utils import check_match, make_pair, make_grad_pair, randf


VAR_SHAPES = [(7,), (5, 4), (3, 4, 5), (300, 500)]


def offset_data(shape, dtype, seed=0):
    # A large mean: E[x^2] - E[x]^2 would lose all precision here
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * 3 + 1e4).astype(dtype)


def all_dims(shape):
    dims = [None]
    for d in range(-len(shape), len(shape)):
        dims.append(d)
    if len(shape) > 1:
        dims += [(0, len(shape) - 1), (len(shape) - 1, 0)]
    return dims


@pytest.mark.parametrize("shape", VAR_SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("name", ["var", "std"])
def test_var_std(name, dtype, shape):
    r, t = make_pair(offset_data(shape, dtype))
    for dim in all_dims(shape):
        for keepdim in (False, True):
            r_out = getattr(r, name)(dim, keepdim=keepdim)
            t_out = getattr(t, name)(dim, keepdim=keepdim)
            check_match(r_out, t_out, atol=1e-4)


@pytest.mark.parametrize("name", ["var", "std"])
def test_var_std_correction(name):
    r, t = make_pair(randf(6, 9))
    for correction in (0, 1, 2.5):
        check_match(getattr(r, name)(1, correction=correction), getattr(t, name)(1, correction=correction))
        check_match(getattr(raxis, name)(r, correction=correction), getattr(torch, name)(t, correction=correction))
    check_match(getattr(r, name)(0, unbiased=False), getattr(t, name)(0, unbiased=False))
    with pytest.raises(RuntimeError):
        getattr(r, name)(0, unbiased=False, correction=0)


@pytest.mark.parametrize("name", ["var", "std"])
def test_var_std_degenerate(name):
    # n <= correction: division by zero gives inf, or NaN when M2 is 0
    for data in ([1.0], [1.0, 2.0], [3.0, 3.0]):
        r, t = make_pair(np.array(data, dtype=np.float32))
        check_match(getattr(r, name)(correction=len(data)), getattr(t, name)(correction=len(data)))
    r, t = make_pair(np.zeros((0, 3), dtype=np.float32))
    check_match(getattr(r, name)(0), getattr(t, name)(0))


def test_var_requires_float():
    with pytest.raises(RuntimeError):
        raxis.tensor([1, 2, 3]).var()


def test_var_non_contiguous():
    r, t = make_pair(randf(40, 60))
    check_match(r.transpose(0, 1).var(1), t.t().var(1))
    check_match(r[:, ::3].std(0), t[:, ::3].std(0))


@pytest.mark.parametrize("name", ["var", "std"])
@pytest.mark.parametrize("dim,keepdim", [(None, False), (1, False), (0, True), ((0, 2), False), ((2, 0), True)])
def test_var_std_backward(name, dim, keepdim):
    data = randf(4, 5, 6)
    r, t = make_grad_pair(data)
    w = np.random.default_rng(1).standard_normal(tuple(t.var(dim, keepdim=keepdim).shape)).astype(np.float32)
    (getattr(r, name)(dim, keepdim=keepdim) * raxis.from_numpy(w)).sum().backward()
    (getattr(t, name)(dim, keepdim=keepdim) * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad, atol=1e-4)
    assert getattr(r, name)(dim).grad_fn.name() == getattr(t, name)(dim).grad_fn.name()


def test_std_backward_constant_input():
    # std = 0: the gradient is 0 instead of 0 / 0
    r, t = make_grad_pair(np.full((3, 4), 2.0, dtype=np.float32))
    r.std(1).sum().backward()
    t.std(1).sum().backward()
    check_match(r.grad, t.grad)
