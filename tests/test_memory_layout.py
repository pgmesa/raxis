import pytest
import raxis
import torch
import numpy as np

from test_utils import check_match, make_pair, randf


def strides_of(r, t):
    return tuple(r.strides), tuple(t.stride())


# ===========================================================================
# Overlap checks on written-to tensors
# ===========================================================================

INPLACE_ON_BROADCAST = [
    ("add_",  lambda x, y: x.add_(y)),
    ("mul_",  lambda x, y: x.mul_(y)),
    ("copy_", lambda x, y: x.copy_(y)),
]


@pytest.mark.parametrize("name,op", INPLACE_ON_BROADCAST, ids=[c[0] for c in INPLACE_ON_BROADCAST])
def test_inplace_into_broadcast_view_raises(name, op):
    r, t = make_pair(randf(1, 4))
    r_other, t_other = make_pair(randf(3, 4, seed=1))
    with pytest.raises(RuntimeError, match="more than one element"):
        op(t.broadcast_to((3, 4)), t_other)
    with pytest.raises(RuntimeError, match="more than one element"):
        op(r.broadcast_to((3, 4)), r_other)


@pytest.mark.parametrize("op", [lambda x: x.fill_(2.0), lambda x: x.zero_()], ids=["fill_", "zero_"])
def test_idempotent_fill_into_broadcast_view(op):
    r, t = make_pair(randf(1, 4))
    op(r.broadcast_to((3, 4)))
    op(t.broadcast_to((3, 4)))
    check_match(r, t)


def test_partial_overlap_with_input_raises():
    r, t = make_pair(randf(10))
    with pytest.raises(RuntimeError, match="some elements of the input"):
        t[1:].add_(t[:-1])
    with pytest.raises(RuntimeError, match="some elements of the input"):
        r[1:].add_(r[:-1])


def test_same_block_other_layout_raises():
    r, t = make_pair(randf(4, 4))
    with pytest.raises(RuntimeError, match="some elements of the input"):
        t.copy_(t.t())
    with pytest.raises(RuntimeError, match="some elements of the input"):
        r.copy_(r.transpose(0, 1))


def test_full_overlap_is_allowed():
    r, t = make_pair(randf(4, 5))
    r.add_(r)
    t.add_(t)
    r[:].mul_(r)
    t[:].mul_(t)
    check_match(r, t)


INTO_BROADCAST_OPERAND = [
    ("add_",    lambda L, a, b: b.add_(a)),
    ("add out", lambda L, a, b: L.add(a, b, out=b)),
    ("mul_",    lambda L, a, b: b.mul_(a)),
]


@pytest.mark.parametrize("name,op", INTO_BROADCAST_OPERAND, ids=[c[0] for c in INTO_BROADCAST_OPERAND])
def test_output_cannot_be_broadcast(name, op):
    # b is broadcast to a's shape as an input; as the output it would need a
    # stride of 0, and every row's sums would land in one element
    r_a, t_a = make_pair(randf(3, 4))
    r_b, t_b = make_pair(randf(3, 1, seed=1))
    with pytest.raises(RuntimeError, match=r"output with shape \[3, 1\] doesn't match the broadcast shape \[3, 4\]"):
        op(torch, t_a, t_b)
    with pytest.raises(RuntimeError, match=r"output with shape \[3, 1\] doesn't match the broadcast shape \[3, 4\]"):
        op(raxis, r_a, r_b)


def test_too_hard_overlap_is_not_detected():
    # Two strided views of one storage that do overlap: the check gives up
    # (neither is dense), as in torch, and the loop reads values it already
    # wrote. Pinned here because the TensorIterator article shows it
    r = raxis.arange(10.0)
    t = torch.arange(10.0)
    raxis.add(r[:-2:2], 0, out=r[2::2])
    torch.add(t[:-2:2], 0, out=t[2::2])
    assert r[2::2].tolist() == t[2::2].tolist() == [0.0, 0.0, 0.0, 0.0]


COALESCE_CASES = [
    # (label, a, b, loop shape, strides in bytes of out, a, b; inner dim first)
    ("same shape", lambda: (raxis.ones(1000, 1000), raxis.ones(1000, 1000)),
     [1000000], [[4], [4], [4]]),
    ("column", lambda: (raxis.ones(1000, 1000), raxis.ones(1000, 1)),
     [1000, 1000], [[4, 4000], [4, 4000], [0, 4]]),
    ("expanded scalar", lambda: (raxis.ones(1000, 1000), raxis.ones(1, 1).expand(1000, 1000)),
     [1000000], [[4], [4], [0]]),
    ("3-D middle", lambda: (raxis.ones(100, 100, 100), raxis.ones(100, 1, 100)),
     [100, 100, 100], [[4, 400, 40000], [4, 400, 40000], [4, 0, 400]]),
    ("3-D first", lambda: (raxis.ones(100, 100, 100), raxis.ones(1, 100, 100)),
     [10000, 100], [[4, 40000], [4, 40000], [4, 0]]),
    ("every other column, M even", lambda: (raxis.ones(6, 10)[:, ::2], raxis.ones(6, 10)[:, ::2]),
     [30], [[4], [8], [8]]),
    ("every other column, M odd", lambda: (raxis.ones(6, 5)[:, ::2], raxis.ones(6, 5)[:, ::2]),
     [3, 6], [[4, 12], [8, 20], [8, 20]]),
    ("both transposed", lambda: (raxis.ones(1000, 1000).t(), raxis.ones(1000, 1000).t()),
     [1000000], [[4], [4], [4]]),
    ("one transposed", lambda: (raxis.ones(1000, 1000), raxis.ones(1000, 1000).t()),
     [1000, 1000], [[4, 4000], [4, 4000], [4000, 4]]),
]


@pytest.mark.parametrize("label,make,shape,strides", COALESCE_CASES, ids=[c[0] for c in COALESCE_CASES])
def test_binary_iterator_coalesces(label, make, shape, strides):
    a, b = make()
    state = raxis._C._binary_iterator_state(a, b)
    assert state["shape"] == shape
    assert state["strides"] == strides


def test_disjoint_slices_are_allowed():
    r, t = make_pair(randf(10))
    r[:5].add_(r[5:])
    t[:5].add_(t[5:])
    check_match(r, t)


@pytest.mark.parametrize("shape", [(4, 4), (64, 64)])
def test_zero_on_strided_slice(shape):
    r, t = make_pair(randf(*shape))
    r[:, 0].zero_()
    t[:, 0].zero_()
    check_match(r, t)


# ===========================================================================
# Layout preserved by *_like and clone
# ===========================================================================

def transposed(x):
    return x.transpose(0, 1)


def permuted(x):
    return x.permute((2, 0, 1)) if isinstance(x, raxis.Tensor) else x.permute(2, 0, 1)


def sliced(x):
    return x[:, ::2]


def broadcast(x):
    return x[:1].broadcast_to((6, 8))


LAYOUTS = [
    ("transposed", (6, 8),    transposed),
    ("permuted",   (4, 5, 6), permuted),
    ("sliced",     (6, 8),    sliced),
    ("broadcast",  (6, 8),    broadcast),
]

LIKE_OPS = [
    ("empty_like", raxis.empty_like, torch.empty_like),
    ("zeros_like", raxis.zeros_like, torch.zeros_like),
    ("ones_like",  raxis.ones_like,  torch.ones_like),
    ("rand_like",  raxis.rand_like,  torch.rand_like),
    ("randn_like", raxis.randn_like, torch.randn_like),
    ("clone",      lambda x: x.clone(), lambda x: x.clone()),
]


@pytest.mark.parametrize("layout,shape,view", LAYOUTS, ids=[c[0] for c in LAYOUTS])
@pytest.mark.parametrize("name,r_op,t_op", LIKE_OPS, ids=[c[0] for c in LIKE_OPS])
def test_like_preserves_layout(name, r_op, t_op, layout, shape, view):
    r, t = make_pair(randf(*shape))
    r_out, t_out = r_op(view(r)), t_op(view(t))
    assert tuple(r_out.shape) == tuple(t_out.shape)
    r_strides, t_strides = strides_of(r_out, t_out)
    assert r_strides == t_strides


@pytest.mark.parametrize("layout,shape,view", LAYOUTS, ids=[c[0] for c in LAYOUTS])
def test_clone_and_contiguous_values(layout, shape, view):
    r, t = make_pair(randf(*shape))
    check_match(view(r).clone(), view(t).clone())
    r_out, t_out = view(r).contiguous(), view(t).contiguous()
    check_match(r_out, t_out)
    assert r_out.is_contiguous()
    r_strides, t_strides = strides_of(r_out, t_out)
    assert r_strides == t_strides


@pytest.mark.parametrize("shape,perm", [((0, 3), (1, 0)), ((3, 1, 4), (1, 2, 0)), ((2, 3), (0, 1))])
def test_is_contiguous(shape, perm):
    r, t = make_pair(np.zeros(shape, dtype=np.float32))
    r_view, t_view = r.permute(perm), t.permute(perm)
    assert r_view.is_contiguous() == t_view.is_contiguous()


# ===========================================================================
# Gradient layout contract
# ===========================================================================

GRAD_CASES = [
    ("mul",    lambda w: (w * 2.0).sum()),
    ("exp",    lambda w: w.exp().sum()),
    ("matmul", lambda w: (w @ w.transpose(0, 1)).sum()),
]


@pytest.mark.parametrize("name,fn", GRAD_CASES, ids=[c[0] for c in GRAD_CASES])
def test_grad_of_transposed_leaf_keeps_its_layout(name, fn):
    data = randf(5, 3)
    r = raxis.from_numpy(data).transpose(0, 1).requires_grad_(True)
    t = torch.from_numpy(data.copy()).t().requires_grad_(True)
    fn(r).backward()
    fn(t).backward()
    check_match(r.grad, t.grad)
    r_strides, t_strides = strides_of(r.grad, t.grad)
    assert r_strides == t_strides == tuple(r.strides)
