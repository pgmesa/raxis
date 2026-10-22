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


@pytest.mark.parametrize("name", ["sum", "mean", "prod"])
def test_reduce_empty_dim(name):
    # Reducing a size-0 dim gives a non-empty result: the identity (0 for
    # sum, 1 for prod) or NaN (mean)
    cases = [((0, 3), 0), ((2, 0, 3), 1)]
    if name != "prod":  # torch.prod takes a single dim
        cases.append(((2, 0, 3), (0, 1)))
    for shape, dim in cases:
        r, t = make_pair(np.zeros(shape, dtype=np.float32))
        check_match(getattr(r, name)(dim), getattr(t, name)(dim))


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


# ===========================================================================
# all / any
# ===========================================================================

ALLANY_DTYPES = [np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_]


def sparse_data(shape, dtype, density, seed=0):
    # Mostly zeros or mostly ones, so all() and any() are not trivially decided
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < density).astype(dtype) * (3 if dtype != np.bool_ else 1)


@pytest.mark.parametrize("dtype", ALLANY_DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("name", ["all", "any"])
@pytest.mark.parametrize("shape", [(7,), (5, 4), (3, 40, 50), (300, 500)])
def test_all_any(name, shape, dtype):
    for density in (0.002, 0.5, 0.998, 1.0):
        r, t = make_pair(sparse_data(shape, dtype, density))
        for dim in all_dims(shape):
            for keepdim in (False, True):
                r_out = getattr(r, name)(dim, keepdim=keepdim)
                t_out = getattr(t, name)(dim, keepdim=keepdim)
                check_match(r_out, t_out)
                # bool, except uint8 -> uint8 as in torch
                assert np.array(r_out).dtype == t_out.numpy().dtype
        check_match(getattr(raxis, name)(r), getattr(torch, name)(t))


@pytest.mark.parametrize("name", ["all", "any"])
def test_all_any_empty_and_nan(name):
    r, t = make_pair(np.zeros((0, 3), dtype=np.float32))
    check_match(getattr(r, name)(), getattr(t, name)())
    check_match(getattr(r, name)(0), getattr(t, name)(0))
    # NaN is nonzero
    r, t = make_pair(np.array([np.nan, 0.0, 1.0], dtype=np.float32))
    check_match(getattr(r, name)(), getattr(t, name)())


def test_all_any_non_contiguous():
    data = sparse_data((40, 60), np.float32, 0.99)
    r, t = make_pair(data)
    check_match(r[:, ::3].all(0), t[:, ::3].all(0))
    check_match(r.transpose(0, 1).any(1), t.transpose(0, 1).any(1))


# ===========================================================================
# cumsum
# ===========================================================================

@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_],
                         ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("shape", [(7,), (5, 4), (3, 40, 50), (300, 500)])
def test_cumsum(shape, dtype):
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(shape) * 10).astype(dtype)
    r, t = make_pair(data)
    for dim in range(-len(shape), len(shape)):
        r_out, t_out = r.cumsum(dim), t.cumsum(dim)
        # Accumulated in double like torch, so float results match closely
        check_match(r_out, t_out, atol=1e-4)
        # Integer and bool inputs give int64
        assert np.array(r_out).dtype == t_out.numpy().dtype
    check_match(raxis.cumsum(r, 0, dtype=raxis.float64), torch.cumsum(t, 0, dtype=torch.float64), atol=1e-4)


def test_cumsum_edge_cases():
    r, t = make_pair(np.array(2.5, dtype=np.float32))
    check_match(r.cumsum(0), t.cumsum(0))
    check_match(r.cumsum(-1), t.cumsum(-1))
    r, t = make_pair(np.zeros((0, 3), dtype=np.float32))
    check_match(r.cumsum(1), t.cumsum(1))
    r, t = make_pair(np.arange(60, dtype=np.float32).reshape(6, 10))
    check_match(r[:, ::3].cumsum(1), t[:, ::3].cumsum(1))
    check_match(r.transpose(0, 1).cumsum(0), t.transpose(0, 1).cumsum(0))


def test_cumsum_long_float_accumulates_in_double():
    # 1e6 values of 0.1: a float accumulator drifts by ~1e3, a double one
    # stays within float rounding of the true sums
    r, t = make_pair(np.full(1_000_000, 0.1, dtype=np.float32))
    np.testing.assert_array_equal(np.array(r.cumsum(0)), t.cumsum(0).numpy())


@pytest.mark.parametrize("dim", [0, 1, -1, 2])
def test_cumsum_backward(dim):
    data = randf(4, 5, 6)
    r, t = make_grad_pair(data)
    w = np.random.default_rng(1).standard_normal((4, 5, 6)).astype(np.float32)
    (r.cumsum(dim) * raxis.from_numpy(w)).sum().backward()
    (t.cumsum(dim) * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad, atol=1e-4)
    assert r.cumsum(dim).grad_fn.name() == t.cumsum(dim).grad_fn.name()


def test_cumsum_backward_dtype_and_size_one():
    r, t = make_grad_pair(randf(3, 1))
    (r.cumsum(1, dtype=raxis.float64) * 2).sum().backward()
    (t.cumsum(1, dtype=torch.float64) * 2).sum().backward()
    check_match(r.grad, t.grad)
    assert np.array(r.grad).dtype == np.float32
