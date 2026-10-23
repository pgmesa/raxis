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
    with pytest.raises(IndexError, match="out of range"):
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


# ---------------------------------------------------------------------------
# Advanced indexing: x[idx], x[mask], x[idx] = v
# ---------------------------------------------------------------------------

MASK = randf(4, 5, 6, seed=7) > 0.5

# Each case builds the same index with either library: lib is raxis or torch
GETITEM_CASES = {
    "long":             lambda lib: lib.tensor([0, 2, 3]),
    "repeated":         lambda lib: lib.tensor([3, 0, 3, 3]),
    "negative":         lambda lib: lib.tensor([-1, 0, -4]),
    "int32":            lambda lib: lib.tensor([1, 2], dtype=lib.int32),
    "2d index":         lambda lib: lib.tensor([[0, 1], [2, 3]]),
    "after slice":      lambda lib: (slice(None), lib.tensor([1, 4])),
    "adjacent":         lambda lib: (slice(None), lib.tensor([0, 1]), lib.tensor([2, 3])),
    "separated":        lambda lib: (lib.tensor([0, 1]), slice(None), lib.tensor([2, 3])),
    "broadcast":        lambda lib: (lib.tensor([[0], [3]]), lib.tensor([1, 2, 4])),
    "with int":         lambda lib: (lib.tensor([[0, 1], [2, 3]]), 1),
    "with ellipsis":    lambda lib: (Ellipsis, lib.tensor([5, 0])),
    "with none":        lambda lib: (None, lib.tensor([1]), None, slice(1, 3)),
    "with step slice":  lambda lib: (slice(0, 4, 2), lib.tensor([4, 0])),
    "list":             lambda lib: [0, 2],
    "empty list":       lambda lib: [],
    "list of bools":    lambda lib: [True, False, True, False],
    "list in tuple":    lambda lib: (0, [1, 2], [3, 4]),
    "0-d long":         lambda lib: lib.tensor(1),
    "0-d long and idx": lambda lib: (lib.tensor(-1), lib.tensor([-2])),
    "0-d bool":         lambda lib: lib.tensor(True),
    "true":             lambda lib: True,
    "false":            lambda lib: False,
    "mask":             lambda lib: lib.from_numpy(MASK.copy()),
    "mask 2d":          lambda lib: (slice(None), lib.from_numpy(MASK[0].copy())),
    "mask 1d":          lambda lib: (lib.from_numpy(MASK[:, 0, 0].copy()), 2),
    "mask and index":   lambda lib: (lib.from_numpy(MASK[:, :, 0].copy()), lib.tensor([0])),
    "empty mask":       lambda lib: lib.zeros(4, dtype=lib.bool),
    "numpy":            lambda lib: np.array([1, 3]),
    "numpy int":        lambda lib: (np.int64(2), lib.tensor([0, 1])),
}


@pytest.mark.parametrize("make_index", GETITEM_CASES.values(), ids=GETITEM_CASES.keys())
def test_getitem(make_index):
    r, t = make_pair(randf(4, 5, 6))
    check_match(r[make_index(raxis)], t[make_index(torch)])


def test_getitem_non_contiguous():
    r, t = make_pair(randf(6, 5, 4))
    rr, tt = r.permute(2, 0, 1), t.permute(2, 0, 1)
    check_match(rr[raxis.tensor([3, 0]), 1:4], tt[torch.tensor([3, 0]), 1:4])
    check_match(rr[:, raxis.tensor([5, 5, 0])], tt[:, torch.tensor([5, 5, 0])])


@pytest.mark.parametrize("dtype", ["int64", "int32", "uint8", "bool", "float64"])
def test_getitem_dtypes(dtype):
    data = (randf(5, 3) * 10).astype(dtype)
    r, t = raxis.from_numpy(data.copy()), torch.from_numpy(data.copy())
    check_match(r[raxis.tensor([4, 0, 4])], t[torch.tensor([4, 0, 4])])
    check_match(r[r > 3], t[t > 3])


def test_advanced_getitem_copies_basic_getitem_views():
    x = raxis.zeros(4, 3)
    x[raxis.tensor([1])][0, 0] = 5.0
    assert x.sum().item() == 0.0
    x[1:2][0, 0] = 5.0
    assert x[1, 0].item() == 5.0


def test_legacy_list_as_tuple_warns():
    r, t = make_pair(randf(3, 4))
    with pytest.warns(UserWarning, match="non-tuple sequence"):
        out = r[[[0, 1], [1, 0]]]
    with pytest.warns(UserWarning):
        expected = t[[[0, 1], [1, 0]]]
    check_match(out, expected)


GETITEM_ERRORS = {
    "out of bounds":    (lambda lib: lib.tensor([4]), "index 4 is out of bounds for dimension 0 with size 4"),
    "float index":      (lambda lib: lib.tensor([0.5]), "tensors used as indices must be long, int, byte or bool tensors"),
    "mask shape":       (lambda lib: lib.ones(2, 2, dtype=lib.bool), "The shape of the mask \\[2, 2\\] at index 0"),
    "broadcast":        (lambda lib: (lib.tensor([0, 1]), lib.tensor([0, 1, 2])), "shape mismatch"),
    "too many":         (lambda lib: (0, 0, 0, lib.tensor([0])), "too many indices for tensor of dimension 3"),
    "int out of range": (lambda lib: (0, 7), "index 7 is out of bounds for dimension 1 with size 5"),
}


@pytest.mark.parametrize("make_index,message", GETITEM_ERRORS.values(), ids=GETITEM_ERRORS.keys())
def test_getitem_errors(make_index, message):
    r, t = make_pair(randf(4, 5, 6))
    with pytest.raises(IndexError, match=message):
        r[make_index(raxis)]
    with pytest.raises(IndexError):
        t[make_index(torch)]


def test_iteration_stops_at_index_error():
    r = raxis.tensor([[0, 1], [2, 3], [4, 5]])
    assert [row.tolist() for row in r] == [[0, 1], [2, 3], [4, 5]]
    with pytest.raises(IndexError, match="select\\(\\): index 3 out of range"):
        r.select(0, 3)


SETITEM_CASES = {
    "long scalar":        (lambda lib: lib.tensor([0, 2]), 3.0),
    "after slice":        (lambda lib: (slice(None), lib.tensor([1, 4])), -1.0),
    "mask scalar":        (lambda lib: lib.from_numpy(MASK.copy()), 0.0),
    "mask 2d scalar":     (lambda lib: (slice(None), lib.from_numpy(MASK[0].copy())), 7.0),
    "mask 1d with int":   (lambda lib: (lib.from_numpy(MASK[:, 0, 0].copy()), 1), 2.0),
    "separated tensor":   (lambda lib: (lib.tensor([0, 1]), slice(None), lib.tensor([2, 3])), (2, 5)),
    "broadcast tensor":   (lambda lib: lib.tensor([3, 1]), (6,)),
    "leading ones value": (lambda lib: lib.tensor([3, 1]), (1, 1, 2, 5, 6)),
    "mask tensor":        (lambda lib: lib.from_numpy(MASK.copy()), (int(MASK.sum()),)),
    "0-d tensor value":   (lambda lib: lib.tensor([1]), ()),
    "basic slice":        (lambda lib: (slice(1, 3), 0), (6,)),
    "true":               (lambda lib: True, 4.0),
    "false":              (lambda lib: False, 4.0),
}


@pytest.mark.parametrize("make_index,value", SETITEM_CASES.values(), ids=SETITEM_CASES.keys())
def test_setitem(make_index, value):
    r, t = make_pair(randf(4, 5, 6))
    if isinstance(value, tuple):
        v = randf(*value, seed=3)
        r[make_index(raxis)] = raxis.from_numpy(v.copy())
        t[make_index(torch)] = torch.from_numpy(v.copy())
    else:
        r[make_index(raxis)] = value
        t[make_index(torch)] = value
    check_match(r, t)


def test_setitem_dtype_mismatch_raises():
    for lib in (raxis, torch):
        x = lib.zeros(3)
        with pytest.raises(RuntimeError, match="Index put requires the source and destination dtypes match"):
            x[lib.tensor([0, 1])] = lib.tensor([1, 2])


def test_index_put_accumulate():
    r, t = make_pair(randf(5, 3))
    v = randf(4, 3, seed=1)
    idx = np.array([4, 0, 4, 4], dtype=np.int64)
    r.index_put_((raxis.from_numpy(idx),), raxis.from_numpy(v.copy()), accumulate=True)
    t.index_put_((torch.from_numpy(idx),), torch.from_numpy(v.copy()), accumulate=True)
    check_match(r, t)
    # None keeps a dim whole (torch takes it only through aten), values broadcast
    r2, t2 = make_pair(randf(5, 3))
    out_r = r2.index_put((None, raxis.tensor([2, 2])), raxis.tensor(1.0), accumulate=True)
    out_t = torch.ops.aten.index_put(t2, [None, torch.tensor([2, 2])], torch.tensor(1.0), True)
    check_match(out_r, out_t)
    check_match(r2, t2)  # out of place


def test_index_put_accumulate_bool_is_or():
    for lib in (raxis, torch):
        x = lib.zeros(3, dtype=lib.bool)
        x.index_put_((lib.tensor([0, 0, 2]),), lib.tensor([True, True, False]), accumulate=True)
        assert x.tolist() == [True, False, False]


@pytest.mark.parametrize("name", ["repeated", "separated", "broadcast", "mask", "mask and index", "list in tuple"])
def test_getitem_backward(name):
    r, t = make_grad_pair(randf(4, 5, 6))
    out_r, out_t = r[GETITEM_CASES[name](raxis)], t[GETITEM_CASES[name](torch)]
    w = randf(*out_t.shape, seed=5)
    (out_r * raxis.from_numpy(w)).sum().backward()
    (out_t * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad)
    assert out_r.grad_fn.name() == "IndexBackward0"


@pytest.mark.parametrize("accumulate", [False, True])
def test_index_put_backward(accumulate):
    r, t = make_grad_pair(randf(5, 4))
    rv, tv = make_grad_pair(randf(4, seed=1))  # broadcast over the indexed rows
    ri, ti = index_pair(np.array([3, 0, 3], dtype=np.int64))
    w = randf(5, 4, seed=2)
    (r.index_put((ri,), rv, accumulate=accumulate) * raxis.from_numpy(w)).sum().backward()
    (t.index_put((ti,), tv, accumulate=accumulate) * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad)
    check_match(rv.grad, tv.grad)


def test_inplace_index_put_on_grad_tensor_raises():
    r = raxis.zeros(3, requires_grad=True)
    with pytest.raises(RuntimeError, match="in-place"):
        r[raxis.tensor([0])] = 1.0


# ---------------------------------------------------------------------------
# masked_fill, nonzero
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mask_shape", [(4, 5, 6), (5, 6), (6,), (4, 1, 6), ()])
def test_masked_fill(mask_shape):
    r, t = make_pair(randf(4, 5, 6))
    m = randf(*mask_shape, seed=1) > 0.5 if mask_shape else np.array(True)
    rm, tm = raxis.from_numpy(m.copy()), torch.from_numpy(m.copy())
    check_match(r.masked_fill(rm, -1.5), t.masked_fill(tm, -1.5))
    check_match(raxis.masked_fill(r, rm, 2), torch.masked_fill(t, tm, 2))
    r.masked_fill_(rm, 9.0)
    t.masked_fill_(tm, 9.0)
    check_match(r, t)


def test_masked_fill_broadcasts_self_and_backward():
    r, t = make_grad_pair(randf(5, 1))
    m = randf(5, 3, seed=1) > 0.5
    out_r = r.masked_fill(raxis.from_numpy(m.copy()), 0.0)
    out_t = t.masked_fill(torch.from_numpy(m.copy()), 0.0)
    check_match(out_r, out_t)
    w = randf(5, 3, seed=2)
    (out_r * raxis.from_numpy(w)).sum().backward()
    (out_t * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad)


def test_masked_fill_errors():
    x = raxis.zeros(3)
    with pytest.raises(RuntimeError, match="only supports boolean masks"):
        x.masked_fill(raxis.tensor([1, 0, 1]), 1.0)
    with pytest.raises(RuntimeError, match="in-place"):
        raxis.zeros(3, requires_grad=True).masked_fill_(raxis.tensor([True, False, True]), 1.0)


@pytest.mark.parametrize("shape", [(1000, 1000), (7, 3, 5), (17,), (), (0, 3)])
def test_nonzero(shape):
    data = (randf(*shape) > 0.7) * randf(*shape, seed=1) if shape else np.float32(0.5)
    data = np.asarray(data, dtype=np.float32)
    r, t = make_pair(data)
    check_match(r.nonzero(), t.nonzero())
    check_match(raxis.nonzero(r), torch.nonzero(t))
    assert r.nonzero().dtype == raxis.int64
    for rc, tc in zip(r.nonzero(as_tuple=True), t.nonzero(as_tuple=True)):
        check_match(rc, tc)


def test_nonzero_values():
    for lib in (raxis, torch):
        x = lib.tensor([[0.0, -0.0, float("nan")], [1.0, 0.0, -2.0]])
        assert x.nonzero().tolist() == [[0, 2], [1, 0], [1, 2]]
