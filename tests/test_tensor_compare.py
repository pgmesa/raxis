import pytest
import raxis
import torch
import numpy as np

from test_utils import check_match, make_pair, make_grad_pair, randf, randf_centered


DTYPES = [np.float32, np.float64, np.int32, np.int64, np.uint8]
# 7 elements only take the scalar loop, 1003 the vectorized one plus a tail
SIZES = [7, 1003]


def rand_of(dtype, n, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.floating):
        return (rng.standard_normal(n) * 5).astype(dtype)
    if dtype == np.uint8:
        return rng.integers(0, 255, n).astype(dtype)
    return rng.integers(-50, 50, n).astype(dtype)


# ===========================================================================
# sign
# ===========================================================================

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
def test_sign(dtype, n):
    data = rand_of(dtype, n)
    data[:3] = 0
    r, t = make_pair(data)
    check_match(r.sign(), t.sign())
    check_match(raxis.sign(r), torch.sign(t))


def test_sign_special_values():
    r, t = make_pair(np.array([np.nan, -0.0, 0.0, -np.inf, np.inf, -2.5], dtype=np.float32))
    check_match(r.sign(), t.sign())
    r, t = make_pair(np.array([True, False]))
    check_match(r.sign(), t.sign())


def test_sign_backward_is_zero():
    r, t = make_grad_pair(randf_centered(10))
    (r.sign() * r).sum().backward()
    (t.sign() * t).sum().backward()
    check_match(r.grad, t.grad)


def test_abs_backward():
    data = randf_centered(20)
    data[:2] = 0
    r, t = make_grad_pair(data)
    r.abs().sum().backward()
    t.abs().sum().backward()
    check_match(r.grad, t.grad)


# ===========================================================================
# where
# ===========================================================================

WHERE_SHAPES = [
    ((1003,), (1003,), (1003,)),
    ((5, 4),  (5, 4),  (1, 4)),
    ((5, 1),  (1, 4),  (5, 4)),
    ((3, 4),  (4,),    ()),
]


@pytest.mark.parametrize("cond_shape,x_shape,y_shape", WHERE_SHAPES)
def test_where(cond_shape, x_shape, y_shape):
    rng = np.random.default_rng(0)
    cond = rng.random(cond_shape) > 0.5
    rc, tc = make_pair(cond)
    rx, tx = make_pair(randf(*x_shape, seed=1))
    ry, ty = make_pair(randf(*y_shape, seed=2))
    check_match(raxis.where(rc, rx, ry), torch.where(tc, tx, ty))
    check_match(rx.where(rc, ry), tx.where(tc, ty))


@pytest.mark.parametrize("x,y", [(np.int32, np.float64), (np.int64, np.int32), (np.uint8, np.float32)],
                         ids=["int32-float64", "int64-int32", "uint8-float32"])
def test_where_promotes(x, y):
    cond = np.array([True, False, True])
    rc, tc = make_pair(cond)
    rx, tx = make_pair(np.array([1, 2, 3], dtype=x))
    ry, ty = make_pair(np.array([4, 5, 6], dtype=y))
    check_match(raxis.where(rc, rx, ry), torch.where(tc, tx, ty))


@pytest.mark.parametrize("x,y", [(0.5, None), (None, 2), (1.0, 0.0), (1, 0), (True, False)])
def test_where_scalars(x, y):
    cond = np.array([[True, False], [False, True]])
    rc, tc = make_pair(cond)
    rt, tt = make_pair(np.array([[1, 2], [3, 4]], dtype=np.int64))
    r_x, t_x = (rt, tt) if x is None else (x, x)
    r_y, t_y = (rt, tt) if y is None else (y, y)
    check_match(raxis.where(rc, r_x, r_y), torch.where(tc, t_x, t_y))


def test_where_requires_bool_condition():
    r, _ = make_pair(randf(3))
    with pytest.raises(RuntimeError, match="boolean"):
        raxis.where(r, r, r)


@pytest.mark.parametrize("x_shape,y_shape", [((5, 4), (5, 4)), ((5, 4), (4,)), ((1, 4), (5, 1))])
def test_where_backward(x_shape, y_shape):
    rng = np.random.default_rng(0)
    rc, tc = make_pair(rng.random((5, 4)) > 0.5)
    rx, tx = make_grad_pair(randf(*x_shape, seed=1))
    ry, ty = make_grad_pair(randf(*y_shape, seed=2))
    (raxis.where(rc, rx, ry) * 3.0).sum().backward()
    (torch.where(tc, tx, ty) * 3.0).sum().backward()
    check_match(rx.grad, tx.grad)
    check_match(ry.grad, ty.grad)


def test_where_backward_scalar_other():
    rng = np.random.default_rng(0)
    rc, tc = make_pair(rng.random(10) > 0.5)
    rx, tx = make_grad_pair(randf(10))
    raxis.where(rc, rx, 0.0).exp().sum().backward()
    torch.where(tc, tx, 0.0).exp().sum().backward()
    check_match(rx.grad, tx.grad)


# ===========================================================================
# clamp
# ===========================================================================

@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("lo,hi", [(-3, 4), (None, 2), (1, None), (5, 2)])
def test_clamp(dtype, n, lo, hi):
    if dtype == np.uint8 and lo is not None and lo < 0:
        lo = 0
    r, t = make_pair(rand_of(dtype, n))
    check_match(r.clamp(lo, hi), t.clamp(lo, hi))
    check_match(raxis.clamp(r, min=lo, max=hi), torch.clamp(t, min=lo, max=hi))
    check_match(r.clip(lo, hi), t.clip(lo, hi))


def test_clamp_min_max_functions():
    r, t = make_pair(randf_centered(1003))
    check_match(r.clamp_min(0.2), t.clamp_min(0.2))
    check_match(r.clamp_max(-0.1), t.clamp_max(-0.1))
    check_match(raxis.clamp_min(r, 0.2), torch.clamp_min(t, 0.2))


@pytest.mark.parametrize("lo,hi", [(2.5, 7), (2, 7), (2, 7.5)])
def test_clamp_promotes_integer_tensors(lo, hi):
    r, t = make_pair(np.array([1, 5, 9], dtype=np.int64))
    check_match(r.clamp(lo, hi), t.clamp(lo, hi))


@pytest.mark.parametrize("lo,hi", [(float("nan"), 1.0), (None, float("nan")), (float("nan"), None)])
def test_clamp_nan_bound(lo, hi):
    r, t = make_pair(randf(1003))
    check_match(r.clamp(lo, hi), t.clamp(lo, hi))


@pytest.mark.parametrize("n", SIZES)
def test_clamp_keeps_nan(n):
    data = randf_centered(n)
    data[::5] = np.nan
    r, t = make_pair(data)
    check_match(r.clamp(-0.5, 0.5), t.clamp(-0.5, 0.5))
    check_match(r.clamp_min(0.0), t.clamp_min(0.0))
    check_match(r.clamp_max(0.0), t.clamp_max(0.0))


def test_clamp_inplace():
    r, t = make_pair(randf_centered(1003))
    r.clamp_(-0.3, 0.3)
    t.clamp_(-0.3, 0.3)
    check_match(r, t)


def test_clamp_inplace_cannot_promote():
    r = raxis.tensor([1, 5, 9])
    with pytest.raises(RuntimeError, match="can't be cast"):
        r.clamp_(2.5, 7)


def test_clamp_requires_a_bound():
    r, t = make_pair(randf(3))
    with pytest.raises(RuntimeError):
        r.clamp()


@pytest.mark.parametrize("lo,hi", [(-0.3, 0.4), (None, 0.1), (-0.2, None)])
def test_clamp_backward(lo, hi):
    data = randf_centered(50)
    data[:2] = [-0.3, 0.4]  # on the bounds
    r, t = make_grad_pair(data)
    (r.clamp(lo, hi) * 2.0).sum().backward()
    (t.clamp(lo, hi) * 2.0).sum().backward()
    check_match(r.grad, t.grad)


# ===========================================================================
# max / min over a dim, argmax / argmin
# ===========================================================================

ARG_SHAPES = [(7,), (5, 4), (3, 4, 5), (300, 500)]


def with_ties_and_nans(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    data = rng.integers(-3, 4, shape).astype(dtype)  # small range: many ties
    if np.issubdtype(dtype, np.floating) and data.size > 4:
        data.flat[::7] = np.nan
    return data


def check_indices(r, t):
    np.testing.assert_array_equal(np.array(r.tolist()), t.numpy())


@pytest.mark.parametrize("shape", ARG_SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64, np.uint8],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("name", ["argmax", "argmin"])
def test_arg_reductions(name, dtype, shape):
    if dtype == np.uint8:
        data = np.random.default_rng(0).integers(0, 5, shape).astype(dtype)
    else:
        data = with_ties_and_nans(shape, dtype)
    r, t = make_pair(data)
    check_indices(getattr(raxis, name)(r), getattr(torch, name)(t))
    for dim in range(-len(shape), len(shape)):
        for keepdim in (False, True):
            check_indices(getattr(r, name)(dim, keepdim), getattr(t, name)(dim, keepdim))


@pytest.mark.parametrize("name", ["argmax", "argmin"])
def test_arg_reductions_special_cases(name):
    r, t = make_pair(np.ones((2, 1, 3), dtype=np.float32))
    check_indices(getattr(r, name)(1), getattr(t, name)(1))
    assert tuple(getattr(r, name)(keepdim=True).shape) == tuple(getattr(t, name)(keepdim=True).shape)
    check_indices(getattr(raxis.tensor(3.0), name)(0), getattr(torch.tensor(3.0), name)(0))
    r, t = make_pair(np.zeros((0, 3), dtype=np.float32))
    assert tuple(getattr(r, name)(1).shape) == tuple(getattr(t, name)(1).shape)
    with pytest.raises(RuntimeError):
        getattr(r, name)()
    with pytest.raises(RuntimeError):
        getattr(raxis.tensor([True, False]), name)()


def test_argmax_large_parallel():
    data = np.random.default_rng(1).standard_normal(200_000)
    data[123_456] = 50.0
    r, t = make_pair(data)
    assert r.argmax().item() == t.argmax().item()
    assert r.argmin().item() == t.argmin().item()


def test_argmax_non_contiguous():
    r, t = make_pair(randf(40, 60))
    check_indices(r.transpose(0, 1).argmax(1), t.t().argmax(1))
    check_indices(r[:, ::3].argmin(0), t[:, ::3].argmin(0))


@pytest.mark.parametrize("shape", ARG_SHAPES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64], ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("name", ["max", "min"])
def test_minmax_dim(name, dtype, shape):
    r, t = make_pair(with_ties_and_nans(shape, dtype))
    for dim in range(-len(shape), len(shape)):
        for keepdim in (False, True):
            r_out = getattr(r, name)(dim, keepdim)
            t_out = getattr(t, name)(dim, keepdim)
            check_match(r_out.values, t_out.values)
            check_indices(r_out.indices, t_out.indices)
            r_values, r_indices = getattr(raxis, name)(r, dim, keepdim=keepdim)
            check_match(r_values, t_out.values)


def test_minmax_dim_bool_and_0d():
    r, t = make_pair(np.array([[True, False], [False, False]]))
    check_match(r.max(1).values, t.max(1).values)
    check_indices(r.min(0).indices, t.min(0).indices)
    out = raxis.tensor(3.0).max(0)
    assert out.values.item() == 3.0 and out.indices.item() == 0


def test_minmax_dim_empty_dim_raises():
    with pytest.raises(RuntimeError):
        raxis.zeros((0, 3)).max(0)


@pytest.mark.parametrize("name", ["max", "min"])
@pytest.mark.parametrize("keepdim", [False, True])
def test_minmax_dim_backward(name, keepdim):
    data = np.array([[1.0, 3.0, 3.0], [4.0, 0.0, 2.0]], dtype=np.float32)
    r, t = make_grad_pair(data)
    (getattr(r, name)(1, keepdim).values.exp() * 2.0).sum().backward()
    (getattr(t, name)(1, keepdim).values.exp() * 2.0).sum().backward()
    check_match(r.grad, t.grad)
    assert getattr(r, name)(0).values.grad_fn.name() == getattr(t, name)(0).values.grad_fn.name()
