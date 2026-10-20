import pytest
import raxis
import torch
import numpy as np

from test_utils import check_match, make_pair, make_grad_pair, randf


def index_pair(data):
    return raxis.from_numpy(data), torch.from_numpy(data.copy())


def rand_index(shape, high, seed=0):
    return np.random.default_rng(seed).integers(0, high, shape).astype(np.int64)


# self shape, dim, index shape (index can be smaller than self apart from dim)
GATHER_CASES = [
    ((5, 4),       0, (3, 4)),
    ((5, 4),       1, (5, 7)),
    ((5, 4),      -1, (2, 3)),
    ((3, 4, 6),    1, (3, 2, 6)),
    ((3, 4, 6),    2, (2, 4, 9)),
    ((300, 200),   0, (100, 200)),
    ((10,),        0, (25,)),
]


@pytest.mark.parametrize("shape,dim,index_shape", GATHER_CASES)
def test_gather(shape, dim, index_shape):
    r, t = make_pair(randf(*shape))
    ri, ti = index_pair(rand_index(index_shape, shape[dim]))
    check_match(r.gather(dim, ri), t.gather(dim, ti))
    check_match(raxis.gather(r, dim, ri), torch.gather(t, dim, ti))


@pytest.mark.parametrize("shape,dim,index_shape", GATHER_CASES)
def test_gather_backward(shape, dim, index_shape):
    r, t = make_grad_pair(randf(*shape))
    ri, ti = index_pair(rand_index(index_shape, shape[dim]))
    w = randf(*index_shape, seed=3)
    (r.gather(dim, ri) * raxis.from_numpy(w)).sum().backward()
    (t.gather(dim, ti) * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad)


def test_gather_non_contiguous():
    r, t = make_pair(randf(6, 8))
    ri, ti = index_pair(rand_index((8, 3), 6))
    check_match(r.transpose(0, 1).gather(1, ri), t.t().gather(1, ti))


def test_gather_out_of_bounds():
    r, t = make_pair(randf(4, 5))
    ri, ti = index_pair(np.full((2, 5), 4, dtype=np.int64))
    with pytest.raises(RuntimeError, match="out of bounds"):
        r.gather(0, ri)
    with pytest.raises(RuntimeError):
        t.gather(0, ti)


# Scatter targets: without repeated indices along dim (where the last write
# wins in an unspecified order), so the results are deterministic
SCATTER_CASES = [
    ((5, 4),    0, (5, 4)),
    ((5, 4),    1, (3, 4)),
    ((4, 6, 3), 1, (2, 6, 3)),
    ((300, 64), 0, (300, 64)),
]


def permutation_index(index_shape, dim, high, seed=0):
    """Distinct positions along dim for every other position."""
    rng = np.random.default_rng(seed)
    moved = np.moveaxis(np.empty(index_shape, dtype=np.int64), dim, -1)
    flat = moved.reshape(-1, moved.shape[-1])
    for row in flat:
        row[:] = rng.permutation(high)[:len(row)]
    return np.ascontiguousarray(np.moveaxis(flat.reshape(moved.shape), -1, dim))


@pytest.mark.parametrize("shape,dim,index_shape", SCATTER_CASES)
def test_scatter(shape, dim, index_shape):
    r, t = make_pair(randf(*shape))
    ri, ti = index_pair(permutation_index(index_shape, dim, shape[dim]))
    rs, ts = make_pair(randf(*index_shape, seed=1))
    check_match(r.scatter(dim, ri, rs), t.scatter(dim, ti, ts))
    check_match(r.scatter(dim, ri, 7.5), t.scatter(dim, ti, 7.5))
    r.scatter_(dim, ri, rs)
    t.scatter_(dim, ti, ts)
    check_match(r, t)


@pytest.mark.parametrize("shape,dim,index_shape", SCATTER_CASES + [((5, 4), 0, (12, 4))])
def test_scatter_add(shape, dim, index_shape):
    # Repeated indices accumulate
    r, t = make_pair(randf(*shape))
    ri, ti = index_pair(rand_index(index_shape, shape[dim]))
    rs, ts = make_pair(randf(*index_shape, seed=1))
    check_match(r.scatter_add(dim, ri, rs), t.scatter_add(dim, ti, ts))
    check_match(raxis.scatter_add(r, dim, ri, rs), torch.scatter_add(t, dim, ti, ts))


@pytest.mark.parametrize("shape,dim,index_shape", SCATTER_CASES)
def test_scatter_backward(shape, dim, index_shape):
    r, t = make_grad_pair(randf(*shape))
    rs, ts = make_grad_pair(randf(*index_shape, seed=1))
    ri, ti = index_pair(permutation_index(index_shape, dim, shape[dim]))
    w = randf(*shape, seed=2)
    (r.scatter(dim, ri, rs) * raxis.from_numpy(w)).sum().backward()
    (t.scatter(dim, ti, ts) * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad)
    check_match(rs.grad, ts.grad)


def test_scatter_add_backward():
    r, t = make_grad_pair(randf(5, 4))
    rs, ts = make_grad_pair(randf(9, 4, seed=1))
    ri, ti = index_pair(rand_index((9, 4), 5))
    w = randf(5, 4, seed=2)
    (r.scatter_add(0, ri, rs) * raxis.from_numpy(w)).sum().backward()
    (t.scatter_add(0, ti, ts) * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad)
    check_match(rs.grad, ts.grad)


def test_scatter_requires_int64_index():
    r = raxis.zeros((3, 3))
    with pytest.raises(RuntimeError, match="int64"):
        r.scatter(0, raxis.tensor([[0, 1, 2]], dtype=raxis.int32), 1.0)


# ===========================================================================
# index_select / index_add
# ===========================================================================

INDEX_SELECT_CASES = [
    ((10,),         0, [3, 0, 3, 9]),
    ((6, 5),        0, [5, 0, 5, 2]),
    ((6, 5),        1, [4, 4, 0]),
    ((4, 5, 6),     2, [1, 0, 5, 5, 3]),
    ((2000, 64),    0, None),  # small slices: parallel memcpy path
    ((4, 40000),    0, [3, 1, 3]),  # large slices: parallel copy kernel
]


def select_index(index, size):
    if index is None:
        return rand_index((5000,), size)
    return np.array(index, dtype=np.int64)


@pytest.mark.parametrize("shape,dim,index", INDEX_SELECT_CASES)
def test_index_select(shape, dim, index):
    r, t = make_pair(randf(*shape))
    ri, ti = index_pair(select_index(index, shape[dim]))
    check_match(r.index_select(dim, ri), t.index_select(dim, ti))
    check_match(raxis.index_select(r, dim, ri), torch.index_select(t, dim, ti))


def test_index_select_non_contiguous_and_int32():
    r, t = make_pair(randf(6, 8))
    ri, ti = index_pair(np.array([7, 0, 3], dtype=np.int32))
    check_match(r.transpose(0, 1).index_select(0, ri), t.t().index_select(0, ti))


@pytest.mark.parametrize("shape,dim,index", INDEX_SELECT_CASES[:4])
def test_index_select_backward(shape, dim, index):
    r, t = make_grad_pair(randf(*shape))
    ri, ti = index_pair(select_index(index, shape[dim]))
    r.index_select(dim, ri).exp().sum().backward()
    t.index_select(dim, ti).exp().sum().backward()
    check_match(r.grad, t.grad)


def test_index_select_out_of_range():
    r = raxis.zeros((3, 2))
    with pytest.raises(RuntimeError, match="out of range"):
        r.index_select(0, raxis.tensor([3]))


@pytest.mark.parametrize("shape,dim,index", INDEX_SELECT_CASES[:4])
def test_index_add(shape, dim, index):
    r, t = make_pair(randf(*shape))
    idx = select_index(index, shape[dim])
    ri, ti = index_pair(idx)
    source_shape = list(shape)
    source_shape[dim] = len(idx)
    rs, ts = make_pair(randf(*source_shape, seed=1))
    check_match(r.index_add(dim, ri, rs), t.index_add(dim, ti, ts))
    r.index_add_(dim, ri, rs)
    t.index_add_(dim, ti, ts)
    check_match(r, t)


def test_index_add_backward():
    r, t = make_grad_pair(randf(6, 5))
    rs, ts = make_grad_pair(randf(4, 5, seed=1))
    ri, ti = index_pair(np.array([5, 0, 5, 2], dtype=np.int64))
    w = randf(6, 5, seed=2)
    (r.index_add(0, ri, rs) * raxis.from_numpy(w)).sum().backward()
    (t.index_add(0, ti, ts) * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad)
    check_match(rs.grad, ts.grad)


def test_add_on_bool_is_or():
    for lib in (raxis, torch):
        idx = lib.tensor([0, 0, 2])
        src = lib.tensor([True, True, False])
        assert lib.zeros(3, dtype=lib.bool).scatter_add(0, idx, src).tolist() == [True, False, False]
        assert lib.zeros(3, dtype=lib.bool).index_add(0, idx, src).tolist() == [True, False, False]
        out = lib.zeros((3, 2), dtype=lib.bool).index_add(0, idx, lib.ones((3, 2), dtype=lib.bool))
        assert out.tolist() == [[True, True], [False, False], [True, True]]
