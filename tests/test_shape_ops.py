import pytest
import raxis
import torch

from test_utils import check_match, device, make_pair, make_grad_pair, randf  # noqa: F401 (fixture)


def check_list_match(r_list, t_list):
    assert len(r_list) == len(t_list)
    for r, t in zip(r_list, t_list):
        check_match(r, t)


# ===========================================================================
# expand
# ===========================================================================

EXPAND_CASES = [
    ((1, 4),    (3, 4)),
    ((3, 1),    (3, 5)),
    ((4,),      (2, 3, 4)),
    ((1, 4),    (2, -1, 4)),
    ((2, 1, 3), (-1, 5, -1)),
    ((3,),      (1, 3)),
]


@pytest.mark.parametrize("shape,size", EXPAND_CASES)
def test_expand(shape, size):
    r, t = make_pair(randf(*shape))
    r_out, t_out = r.expand(size), t.expand(size)
    check_match(r_out, t_out)
    assert tuple(r_out.strides) == tuple(t_out.stride())


@pytest.mark.parametrize("shape,size", EXPAND_CASES)
def test_expand_backward(shape, size):
    r, t = make_grad_pair(randf(*shape))
    (r.expand(size) * 2.0).sum().backward()
    (t.expand(size) * 2.0).sum().backward()
    check_match(r.grad, t.grad)


@pytest.mark.parametrize("shape,size", [((3, 4), (3, 5)), ((4,), (-1, 4))])
def test_expand_invalid(shape, size):
    r, t = make_pair(randf(*shape))
    with pytest.raises(RuntimeError):
        t.expand(size)
    with pytest.raises(RuntimeError):
        r.expand(size)


# ===========================================================================
# split / chunk / unbind
# ===========================================================================

SPLIT_CASES = [
    ((10, 4), 3,         0),
    ((10, 4), 5,         0),
    ((10, 4), 20,        0),
    ((4, 10), 4,        -1),
    ((10, 4), [2, 3, 5], 0),
    ((4, 10), [10],      1),
    ((0, 4),  0,         0),
]


@pytest.mark.parametrize("shape,split,dim", SPLIT_CASES)
def test_split(shape, split, dim):
    r, t = make_pair(randf(*shape))
    check_list_match(r.split(split, dim), t.split(split, dim))
    check_list_match(raxis.split(r, split, dim), torch.split(t, split, dim))


@pytest.mark.parametrize("shape,chunks,dim", [((10, 4), 3, 0), ((5, 4), 4, 0), ((4, 6), 6, 1),
                                              ((4, 2), 5, 1), ((0, 3), 3, 0)])
def test_chunk(shape, chunks, dim):
    r, t = make_pair(randf(*shape))
    check_list_match(r.chunk(chunks, dim), t.chunk(chunks, dim))
    check_list_match(raxis.chunk(r, chunks, dim), torch.chunk(t, chunks, dim))


def test_split_returns_views():
    r = raxis.zeros((6, 2))
    r.split(2)[1].fill_(1.0)
    assert r.tolist() == [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 0]]


@pytest.mark.parametrize("used", [[0], [1, 2], [0, 1, 2]])
def test_split_backward_with_unused_pieces(used):
    r, t = make_grad_pair(randf(9, 4))
    r_pieces, t_pieces = r.split([2, 3, 4]), t.split([2, 3, 4])
    sum(r_pieces[i].exp().sum() for i in used).backward()
    sum(t_pieces[i].exp().sum() for i in used).backward()
    check_match(r.grad, t.grad)


def test_chunk_backward():
    r, t = make_grad_pair(randf(4, 6))
    a, b, c = r.chunk(3, 1)
    (a * b + c).sum().backward()
    a, b, c = t.chunk(3, 1)
    (a * b + c).sum().backward()
    check_match(r.grad, t.grad)


@pytest.mark.parametrize("used", [[0], [2], [0, 1, 2]])
def test_unbind_backward_with_unused_outputs(used):
    r, t = make_grad_pair(randf(3, 4))
    r_rows, t_rows = r.unbind(0), t.unbind(0)
    sum((r_rows[i] * (i + 1)).sum() for i in used).backward()
    sum((t_rows[i] * (i + 1)).sum() for i in used).backward()
    check_match(r.grad, t.grad)


def test_split_invalid():
    r, t = make_pair(randf(5, 2))
    for x in (r, t):
        with pytest.raises(RuntimeError):
            x.split([2, 2])
    for x in (raxis.zeros(()), torch.zeros(())):
        with pytest.raises(RuntimeError):
            x.split(1)


# ===========================================================================
# __iter__
# ===========================================================================

def test_iter():
    r, t = make_pair(randf(3, 4))
    check_list_match(list(r), list(t))


def test_iter_backward():
    r, t = make_grad_pair(randf(3, 4))
    a, b, c = r
    (a * c).sum().backward()
    a, b, c = t
    (a * c).sum().backward()
    check_match(r.grad, t.grad)



# ===========================================================================
# Sizes as separate ints
# ===========================================================================

@pytest.mark.parametrize("name", ["empty", "zeros", "ones", "rand", "randn"])
def test_factory_varargs(name):
    assert tuple(getattr(raxis, name)(2, 3, 4).shape) == (2, 3, 4)
    assert tuple(getattr(raxis, name)((2, 3)).shape) == (2, 3)
    assert tuple(getattr(raxis, name)(5).shape) == (5,)


def test_method_varargs():
    r, t = make_pair(randf(2, 3, 4))
    check_match(r.reshape(6, -1), t.reshape(6, -1))
    check_match(r.view(4, 6), t.view(4, 6))
    check_match(r.permute(2, 0, 1), t.permute(2, 0, 1))
    check_match(r[:1].expand(3, -1, -1), t[:1].expand(3, -1, -1))
    check_match(r.reshape(-1), t.reshape(-1))
    check_match(r.view(-1), t.view(-1))


# ===========================================================================
# flip
# ===========================================================================

FLIP_CASES = [
    ((7,),       (0,)),
    ((3, 4),     (0,)),
    ((3, 4),     (1,)),
    ((3, 4),     (0, 1)),
    ((2, 3, 4),  (0, 2)),
    ((2, 3, 4),  (-1, 1)),
    ((4, 1, 3),  (1, 2)),
    ((64, 300),  (1,)),
    ((300, 64),  (0,)),
    ((3, 4),     ()),
]


@pytest.mark.parametrize("shape,dims", FLIP_CASES)
def test_flip(shape, dims, device):
    r, t = make_pair(randf(*shape), device)
    check_match(r.flip(dims), t.flip(dims))
    check_match(raxis.flip(r, dims), torch.flip(t, dims))


def test_flip_layouts_and_dtypes(device):
    r, t = make_pair(randf(2, 3, 4), device)
    check_match(r.permute(2, 0, 1).flip(0, 1), t.permute(2, 0, 1).flip(0, 1))
    check_match(r[:, ::2].flip(1), t[:, ::2].flip(1))
    check_match(r[:1].expand(3, -1, -1).flip(0, 2), t[:1].expand(3, -1, -1).flip(0, 2))
    check_match(r.flip(1).flip(1), t)
    for dtype in ["int64", "int32", "bool", "float64"]:
        rx, tx = r.to(getattr(raxis, dtype)), t.to(getattr(torch, dtype))
        check_match(rx.flip(0, 2), tx.flip(0, 2))
    with pytest.raises(RuntimeError):
        r.flip(0, 0)


def test_flip_preserves_layout(device):
    r, t = make_pair(randf(4, 6), device)
    r_out, t_out = r.transpose(0, 1).flip(0), t.t().flip(0)
    check_match(r_out, t_out)
    assert tuple(r_out.strides) == tuple(t_out.stride())


@pytest.mark.parametrize("shape,dims", FLIP_CASES[:6])
def test_flip_backward(shape, dims, device):
    r, t = make_grad_pair(randf(*shape), device)
    w = randf(*shape, seed=1)
    (r.flip(dims) * raxis.from_numpy(w).to(device)).sum().backward()
    (t.flip(dims) * torch.from_numpy(w).to(device)).sum().backward()
    check_match(r.grad, t.grad)
    assert r.flip(dims).grad_fn.name() == t.flip(dims).grad_fn.name()


# ===========================================================================
# tril / triu
# ===========================================================================

@pytest.mark.parametrize("fn", ["tril", "triu"])
@pytest.mark.parametrize("diagonal", [-7, -2, 0, 1, 3, 9])
@pytest.mark.parametrize("shape", [(5, 6), (6, 5), (3, 5, 6), (2, 1, 4, 4), (300, 300)])
def test_tril_triu(fn, diagonal, shape):
    r, t = make_pair(randf(*shape))
    check_match(getattr(r, fn)(diagonal), getattr(t, fn)(diagonal))
    check_match(getattr(raxis, fn)(r, diagonal), getattr(torch, fn)(t, diagonal))


@pytest.mark.parametrize("fn", ["tril", "triu"])
def test_tril_triu_layouts_and_dtypes(fn):
    r, t = make_pair(randf(3, 5, 6))
    # A transposed matrix is read in place; a permuted batch is copied first
    check_match(getattr(r[0].transpose(0, 1), fn)(1), getattr(t[0].transpose(0, 1), fn)(1))
    check_match(getattr(r.transpose(0, 2), fn)(-1), getattr(t.transpose(0, 2), fn)(-1))
    check_match(getattr(r[:, ::2], fn)(), getattr(t[:, ::2], fn)())
    for dtype in ["int64", "int32", "bool", "float64"]:
        check_match(getattr(r.to(getattr(raxis, dtype)), fn)(), getattr(t.to(getattr(torch, dtype)), fn)())
    check_match(getattr(raxis.ones(0, 3), fn)(), getattr(torch.ones(0, 3), fn)())
    with pytest.raises(RuntimeError, match="input tensor must have at least 2 dimensions"):
        getattr(raxis.ones(3), fn)()


@pytest.mark.parametrize("fn", ["tril", "triu"])
def test_tril_triu_backward(fn):
    r, t = make_grad_pair(randf(2, 4, 5))
    w = randf(2, 4, 5, seed=1)
    r_out, t_out = getattr(r, fn)(1), getattr(t, fn)(1)
    (r_out * raxis.from_numpy(w)).sum().backward()
    (t_out * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad)
    assert r_out.grad_fn.name() == t_out.grad_fn.name()


# ===========================================================================
# clone(memory_format=...)
# ===========================================================================

@pytest.mark.parametrize("fmt", ["preserve_format", "contiguous_format"])
def test_clone_memory_format(fmt):
    r, t = make_pair(randf(4, 6))
    r_out = r.transpose(0, 1).clone(memory_format=getattr(raxis, fmt))
    t_out = t.t().clone(memory_format=getattr(torch, fmt))
    check_match(r_out, t_out)
    assert tuple(r_out.strides) == tuple(t_out.stride())


@pytest.mark.parametrize("name", ["empty_like", "zeros_like", "ones_like", "rand_like", "randn_like"])
def test_like_memory_format(name):
    r, t = make_pair(randf(4, 6))
    r_out = getattr(raxis, name)(r.transpose(0, 1), memory_format=raxis.contiguous_format)
    t_out = getattr(torch, name)(t.t(), memory_format=torch.contiguous_format)
    assert tuple(r_out.strides) == tuple(t_out.stride())
