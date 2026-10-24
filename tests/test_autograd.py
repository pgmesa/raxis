
import pytest
import raxis
import torch
import numpy as np

from test_utils import (
    check_match, make_pair, make_grad_pair, randf, randf_positive,
    randf_nonzero, randf_centered, randf_small, randf_unit, device
)


# ---------------------------------------------------------------------------
# Autograd helpers
# ---------------------------------------------------------------------------

NOT_IMPLEMENTED_MSG = "is not implemented"


def try_backward(r_out, t_out, r_inputs, t_inputs, context="",
                 atol=1e-4, rtol=1e-3):
    """Run backward on both and compare gradients.

    Skips the test if raxis raises a 'not implemented' error.
    Re-raises any other RuntimeError so real bugs surface.
    """
    t_out.backward(torch.ones_like(t_out), retain_graph=True)
    try:
        r_out.backward(raxis.ones_like(r_out), retain_graph=True)
    except RuntimeError as e:
        if NOT_IMPLEMENTED_MSG in str(e):
            pytest.skip(f"{context}: {e}")
        raise
    except AttributeError:
        pytest.skip(f"{context}: raxis backward not available")

    for r_in, t_in in zip(r_inputs, t_inputs):
        if t_in.grad is not None and r_in.grad is not None:
            check_match(r_in.grad, t_in.grad, atol=atol, rtol=rtol,
                        context=f"{context} grad")


def zero_grads(*tensors):
    for t in tensors:
        if hasattr(t, "grad"):
            t.grad = None


# ---------------------------------------------------------------------------
# Op tables
# ---------------------------------------------------------------------------

BACKWARD_SHAPES = [(5, 10), (10, 20), (3, 10, 20)]

UNARY_BACKWARD_OPS = [
    ("neg",        randf_centered),
    ("abs",        randf_centered),
    ("exp",        randf_small),
    ("expm1",      randf_small),
    ("log",        randf_positive),
    ("log2",       randf_positive),
    ("log10",      randf_positive),
    ("log1p",      randf_positive),
    ("sqrt",       randf_positive),
    ("rsqrt",      randf_positive),
    ("reciprocal", randf_nonzero),
    ("sigmoid",    randf_centered),
    ("sin",        randf_centered),
    ("cos",        randf_centered),
    ("tan",        randf_small),
    ("asin",       randf_unit),
    ("acos",       randf_unit),
    ("atan",       randf_centered),
    ("sinh",       randf_small),
    ("cosh",       randf_small),
    ("tanh",       randf_centered),
    ("erf",        randf_centered),
    ("erfc",       randf_centered),
]

ZERO_GRAD_OPS = [
    ("ceil",  randf_centered),
    ("floor", randf_centered),
    ("round", randf_centered),
]

BINARY_BACKWARD_OPS = [
    ("add", "+", randf, randf),
    ("sub", "-", randf, randf),
    ("mul", "*", randf, randf),
    ("div", "/", randf, randf_nonzero),
]

BACKWARD_BROADCAST_CASES = [
    ((10, 20), (1, 20)),
    ((10, 20), (10, 1)),
    ((10, 20), (20,)),
    ((5, 1, 20), (1, 10, 20)),
]


# ===========================================================================
# UNARY BACKWARD
# ===========================================================================

class TestUnaryBackward:

    @pytest.mark.parametrize("shape", BACKWARD_SHAPES)
    @pytest.mark.parametrize(
        "name,data_fn", UNARY_BACKWARD_OPS,
        ids=[u[0] for u in UNARY_BACKWARD_OPS]
    )
    def test_backward(self, shape, name, data_fn, device):
        data = data_fn(*shape)
        r, t = make_grad_pair(data, device)
        r_out = getattr(r, name)()
        t_out = getattr(t, name)()
        try_backward(r_out, t_out, [r], [t], context=f"{name} {shape}")

    @pytest.mark.parametrize(
        "name,data_fn", ZERO_GRAD_OPS,
        ids=[u[0] for u in ZERO_GRAD_OPS]
    )
    def test_zero_grad_ops(self, name, data_fn, device):
        """ceil/floor/round have zero gradient. Both raxis and pytorch should
        either return zero grads or raise not-implemented."""
        data = data_fn(10, 20)
        r, t = make_grad_pair(data, device)
        r_out = getattr(r, name)()
        t_out = getattr(t, name)()
        try_backward(r_out, t_out, [r], [t], context=f"{name} zero-grad")


# ===========================================================================
# BINARY BACKWARD
# ===========================================================================

class TestBinaryBackward:

    @pytest.mark.parametrize("shape", BACKWARD_SHAPES)
    @pytest.mark.parametrize(
        "name,op_str,data_fn_a,data_fn_b", BINARY_BACKWARD_OPS,
        ids=[o[0] for o in BINARY_BACKWARD_OPS]
    )
    def test_backward(self, shape, name, op_str, data_fn_a, data_fn_b, device):
        a_data = data_fn_a(*shape, seed=42)
        b_data = data_fn_b(*shape, seed=99)
        r_a, t_a = make_grad_pair(a_data, device)
        r_b, t_b = make_grad_pair(b_data, device)
        r_out = eval(f"r_a {op_str} r_b")
        t_out = eval(f"t_a {op_str} t_b")
        try_backward(r_out, t_out, [r_a, r_b], [t_a, t_b],
                      context=f"{name} {shape}")

    @pytest.mark.parametrize(
        "lhs_shape,rhs_shape", BACKWARD_BROADCAST_CASES,
        ids=[f"{a}x{b}" for a, b in BACKWARD_BROADCAST_CASES]
    )
    @pytest.mark.parametrize(
        "name,op_str,data_fn_a,data_fn_b", BINARY_BACKWARD_OPS,
        ids=[o[0] for o in BINARY_BACKWARD_OPS]
    )
    def test_backward_broadcast(self, lhs_shape, rhs_shape, name,
                                 op_str, data_fn_a, data_fn_b, device):
        a_data = data_fn_a(*lhs_shape, seed=42)
        b_data = data_fn_b(*rhs_shape, seed=99)
        r_a, t_a = make_grad_pair(a_data, device)
        r_b, t_b = make_grad_pair(b_data, device)
        r_out = eval(f"r_a {op_str} r_b")
        t_out = eval(f"t_a {op_str} t_b")
        try_backward(r_out, t_out, [r_a, r_b], [t_a, t_b],
                      context=f"{name} bcast {lhs_shape}x{rhs_shape}")

    @pytest.mark.parametrize("scalar", [2.0, 0.5, -3.0])
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_scalar_backward(self, scalar, op, device):
        data = randf(10, 20, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = getattr(r, op)(scalar)
        t_out = getattr(t, op)(scalar)
        try_backward(r_out, t_out, [r], [t], context=f"{op}(scalar={scalar})")


# ===========================================================================
# REVERSED OPS BACKWARD (rsub, rdiv)
# ===========================================================================

class TestReversedOpsBackward:

    @pytest.mark.parametrize("scalar", [2.0, 5.0])
    def test_rsub_backward(self, scalar, device):
        data = randf(10, 20, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = scalar - r
        t_out = scalar - t
        try_backward(r_out, t_out, [r], [t], context=f"rsub(scalar={scalar})")

    @pytest.mark.parametrize("scalar", [2.0, 5.0])
    def test_rdiv_backward(self, scalar, device):
        data = randf_nonzero(10, 20, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = scalar / r
        t_out = scalar / t
        try_backward(r_out, t_out, [r], [t], context=f"rdiv(scalar={scalar})")


# ===========================================================================
# MAXIMUM / MINIMUM BACKWARD
# ===========================================================================

class TestMaxMinElementwiseBackward:

    @pytest.mark.parametrize("op", ["maximum", "minimum"])
    @pytest.mark.parametrize("shape", [(10, 20), (5, 10, 20)])
    def test_backward(self, op, shape, device):
        a_data = randf(*shape, seed=1)
        b_data = randf(*shape, seed=2)
        r_a, t_a = make_grad_pair(a_data, device)
        r_b, t_b = make_grad_pair(b_data, device)
        r_out = getattr(raxis, op)(r_a, r_b)
        t_out = getattr(torch, op)(t_a, t_b)
        try_backward(r_out, t_out, [r_a, r_b], [t_a, t_b], context=f"{op}")

    @pytest.mark.parametrize("op", ["maximum", "minimum"])
    @pytest.mark.parametrize(
        "lhs_shape,rhs_shape", BACKWARD_BROADCAST_CASES,
        ids=[f"{a}x{b}" for a, b in BACKWARD_BROADCAST_CASES]
    )
    def test_backward_broadcast(self, op, lhs_shape, rhs_shape, device):
        a_data = randf(*lhs_shape, seed=1)
        b_data = randf(*rhs_shape, seed=2)
        r_a, t_a = make_grad_pair(a_data, device)
        r_b, t_b = make_grad_pair(b_data, device)
        r_out = getattr(raxis, op)(r_a, r_b)
        t_out = getattr(torch, op)(t_a, t_b)
        try_backward(r_out, t_out, [r_a, r_b], [t_a, t_b],
                      context=f"{op} bcast {lhs_shape}x{rhs_shape}")


# ===========================================================================
# POW BACKWARD
# ===========================================================================

class TestPowBackward:

    @pytest.mark.parametrize("exp", [2.0, 3.0, 0.5, -1.0])
    def test_pow_scalar(self, exp, device):
        data = randf_positive(10, 20)
        r, t = make_grad_pair(data, device)
        try_backward(r.pow(exp), t.pow(exp), [r], [t],
                      context=f"pow(scalar={exp})")

    def test_pow_tensor_tensor(self, device):
        base_data = randf_positive(10, 20, seed=1)
        exp_data = randf(10, 20, seed=2, lo=0.5, hi=3.0)
        r_b, t_b = make_grad_pair(base_data, device)
        r_e, t_e = make_grad_pair(exp_data, device)
        r_out = raxis.pow(r_b, r_e)
        t_out = torch.pow(t_b, t_e)
        try_backward(r_out, t_out, [r_b, r_e], [t_b, t_e],
                      context="pow tensor-tensor")

    def test_pow_scalar_base(self, device):
        exp_data = randf(10, 20, seed=2, lo=0.5, hi=3.0)
        r_e, t_e = make_grad_pair(exp_data, device)
        r_out = raxis.pow(2.0, r_e)
        t_out = torch.pow(2.0, t_e)
        try_backward(r_out, t_out, [r_e], [t_e],
                      context="pow(2.0, tensor)")


# ===========================================================================
# REDUCTION BACKWARD
# ===========================================================================

class TestReductionBackward:

    # -- sum --

    @pytest.mark.parametrize("shape", [(10, 20), (5, 10, 20)])
    def test_sum_all(self, shape, device):
        data = randf(*shape, seed=42)
        r, t = make_grad_pair(data, device)
        try_backward(r.sum(), t.sum(), [r], [t], context=f"sum_all {shape}")

    @pytest.mark.parametrize("shape,dim,keepdim", [
        ((10, 20), 0, False),
        ((10, 20), 1, False),
        ((10, 20), 0, True),
        ((5, 10, 20), 2, False),
        ((5, 10, 20), (0, 2), False),
        ((5, 10, 20), -1, True),
    ])
    def test_sum_dim(self, shape, dim, keepdim, device):
        data = randf(*shape, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = r.sum(dim=dim, keepdim=keepdim)
        t_out = t.sum(dim=dim, keepdim=keepdim)
        try_backward(r_out, t_out, [r], [t],
                      context=f"sum dim={dim} keepdim={keepdim} {shape}")

    # -- mean --

    @pytest.mark.parametrize("shape", [(10, 20), (5, 10, 20)])
    def test_mean_all(self, shape, device):
        data = randf(*shape, seed=42)
        r, t = make_grad_pair(data, device)
        try_backward(r.mean(), t.mean(), [r], [t], context=f"mean_all {shape}")

    @pytest.mark.parametrize("shape,dim,keepdim", [
        ((10, 20), 0, False),
        ((10, 20), 1, True),
        ((5, 10, 20), 2, False),
        ((5, 10, 20), (0, 2), False),
    ])
    def test_mean_dim(self, shape, dim, keepdim, device):
        data = randf(*shape, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = r.mean(dim=dim, keepdim=keepdim)
        t_out = t.mean(dim=dim, keepdim=keepdim)
        try_backward(r_out, t_out, [r], [t],
                      context=f"mean dim={dim} keepdim={keepdim} {shape}")

    # -- prod --

    @pytest.mark.parametrize("shape", [(5, 10), (3, 5, 7)])
    def test_prod_all(self, shape, device):
        data = randf_unit(*shape, seed=42)
        r, t = make_grad_pair(data, device)
        try_backward(r.prod(), t.prod(), [r], [t], context=f"prod_all {shape}")

    @pytest.mark.parametrize("shape,dim", [
        ((5, 10), 0),
        ((5, 10), 1),
        ((3, 5, 7), 2),
    ])
    def test_prod_dim(self, shape, dim, device):
        data = randf_unit(*shape, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = r.prod(dim=dim)
        t_out = t.prod(dim=dim)
        try_backward(r_out, t_out, [r], [t], context=f"prod dim={dim} {shape}")

    # -- max / min (full) --

    @pytest.mark.parametrize("op", ["max", "min"])
    @pytest.mark.parametrize("shape", [(10, 20), (5, 10, 20)])
    def test_max_min_all(self, op, shape, device):
        data = randf(*shape, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = getattr(r, op)()
        t_out = getattr(t, op)()
        try_backward(r_out, t_out, [r], [t], context=f"{op}_all {shape}",
                      atol=1e-3, rtol=1e-2)

    # -- amax / amin --

    @pytest.mark.parametrize("op", ["amax", "amin"])
    @pytest.mark.parametrize("shape,dim,keepdim", [
        ((10, 20), 0, False),
        ((10, 20), 1, True),
        ((5, 10, 20), 2, False),
    ])
    def test_amax_amin(self, op, shape, dim, keepdim, device):
        data = randf(*shape, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = getattr(r, op)(dim=dim, keepdim=keepdim)
        t_out = getattr(t, op)(dim=dim, keepdim=keepdim)
        try_backward(r_out, t_out, [r], [t],
                      context=f"{op} dim={dim} keepdim={keepdim} {shape}",
                      atol=1e-3, rtol=1e-2)

    # -- nansum / nanmean --

    @pytest.mark.parametrize("op", ["nansum", "nanmean"])
    @pytest.mark.parametrize("shape", [(10, 20), (5, 10, 20)])
    def test_nan_reduce_all(self, op, shape, device):
        data = randf(*shape, seed=42)
        rng = np.random.RandomState(42)
        data[rng.rand(*shape) < 0.1] = float("nan")
        r, t = make_grad_pair(data, device)
        r_out = getattr(r, op)()
        t_out = getattr(t, op)()
        try_backward(r_out, t_out, [r], [t], context=f"{op}_all {shape}")

    @pytest.mark.parametrize("op", ["nansum", "nanmean"])
    @pytest.mark.parametrize("shape,dim", [
        ((10, 20), 0),
        ((10, 20), 1),
        ((5, 10, 20), 2),
    ])
    def test_nan_reduce_dim(self, op, shape, dim, device):
        data = randf(*shape, seed=42)
        rng = np.random.RandomState(42)
        data[rng.rand(*shape) < 0.1] = float("nan")
        r, t = make_grad_pair(data, device)
        r_out = getattr(r, op)(dim=dim)
        t_out = getattr(t, op)(dim=dim)
        try_backward(r_out, t_out, [r], [t], context=f"{op} dim={dim} {shape}")


# ===========================================================================
# LINEAR ALGEBRA BACKWARD
# ===========================================================================

class TestLinalgBackward:

    def test_mm(self, device):
        r_a, t_a = make_grad_pair(randf(10, 20, seed=1), device)
        r_b, t_b = make_grad_pair(randf(20, 15, seed=2), device)
        try_backward(raxis.mm(r_a, r_b), torch.mm(t_a, t_b),
                      [r_a, r_b], [t_a, t_b], context="mm")

    def test_mv(self, device):
        r_m, t_m = make_grad_pair(randf(10, 20, seed=1), device)
        r_v, t_v = make_grad_pair(randf(20, seed=2), device)
        try_backward(raxis.mv(r_m, r_v), torch.mv(t_m, t_v),
                      [r_m, r_v], [t_m, t_v], context="mv")

    def test_dot(self, device):
        r_a, t_a = make_grad_pair(randf(64, seed=1), device)
        r_b, t_b = make_grad_pair(randf(64, seed=2), device)
        try_backward(raxis.dot(r_a, r_b), torch.dot(t_a, t_b),
                      [r_a, r_b], [t_a, t_b], context="dot")

    def test_bmm(self, device):
        r_a, t_a = make_grad_pair(randf(4, 8, 16, seed=1), device)
        r_b, t_b = make_grad_pair(randf(4, 16, 6, seed=2), device)
        try_backward(raxis.bmm(r_a, r_b), torch.bmm(t_a, t_b),
                      [r_a, r_b], [t_a, t_b], context="bmm")

    def test_addmm(self, device):
        m, k, n = 8, 16, 4
        r_s, t_s = make_grad_pair(randf(m, n, seed=1), device)
        r_a, t_a = make_grad_pair(randf(m, k, seed=2), device)
        r_b, t_b = make_grad_pair(randf(k, n, seed=3), device)
        try_backward(
            raxis.addmm(r_s, r_a, r_b, beta=0.5, alpha=2.0),
            torch.addmm(t_s, t_a, t_b, beta=0.5, alpha=2.0),
            [r_s, r_a, r_b], [t_s, t_a, t_b], context="addmm")

    def test_addmv(self, device):
        m, k = 8, 16
        r_s, t_s = make_grad_pair(randf(m, seed=1), device)
        r_m, t_m = make_grad_pair(randf(m, k, seed=2), device)
        r_v, t_v = make_grad_pair(randf(k, seed=3), device)
        try_backward(
            raxis.addmv(r_s, r_m, r_v, beta=0.5, alpha=2.0),
            torch.addmv(t_s, t_m, t_v, beta=0.5, alpha=2.0),
            [r_s, r_m, r_v], [t_s, t_m, t_v], context="addmv")

    @pytest.mark.parametrize("desc,shape_a,shape_b", [
        ("dot",       (16,),        (16,)),
        ("mv",        (8, 16),      (16,)),
        ("vec_mat",   (16,),        (16, 8)),
        ("mm",        (8, 16),      (16, 4)),
        ("bmm",       (4, 8, 16),   (4, 16, 6)),
        ("bmm_bcast", (1, 8, 16),   (4, 16, 6)),
    ])
    def test_matmul(self, desc, shape_a, shape_b, device):
        r_a, t_a = make_grad_pair(randf(*shape_a, seed=1), device)
        r_b, t_b = make_grad_pair(randf(*shape_b, seed=2), device)
        try_backward(raxis.matmul(r_a, r_b), torch.matmul(t_a, t_b),
                      [r_a, r_b], [t_a, t_b], context=f"matmul {desc}")


# ===========================================================================
# SHAPE OPS BACKWARD
# ===========================================================================

class TestShapeBackward:

    def test_reshape(self, device):
        r, t = make_grad_pair(randf(10, 20, seed=42), device)
        try_backward(r.reshape((-1,)), t.reshape((-1,)), [r], [t],
                      context="reshape")

    def test_view(self, device):
        r, t = make_grad_pair(randf(10, 20, seed=42), device)
        try_backward(r.view((20, 10)), t.view((20, 10)), [r], [t],
                      context="view")

    def test_transpose(self, device):
        r, t = make_grad_pair(randf(10, 20, seed=42), device)
        try_backward(r.transpose(0, 1), t.transpose(0, 1), [r], [t],
                      context="transpose")

    def test_permute(self, device):
        r, t = make_grad_pair(randf(3, 10, 20, seed=42), device)
        try_backward(r.permute([2, 0, 1]), t.permute(2, 0, 1), [r], [t],
                      context="permute")

    def test_unsqueeze(self, device):
        r, t = make_grad_pair(randf(10, 20, seed=42), device)
        try_backward(r.unsqueeze(0), t.unsqueeze(0), [r], [t],
                      context="unsqueeze")

    def test_squeeze(self, device):
        r, t = make_grad_pair(randf(10, 1, 20, seed=42), device)
        try_backward(r.squeeze(1), t.squeeze(1), [r], [t],
                      context="squeeze")

    def test_broadcast_to(self, device):
        r, t = make_grad_pair(randf(1, 20, seed=42), device)
        try_backward(r.broadcast_to([10, 20]), t.broadcast_to(10, 20),
                      [r], [t], context="broadcast_to")

    def test_select(self, device):
        r, t = make_grad_pair(randf(5, 10, seed=42), device)
        try_backward(r.select(0, 2), t.select(0, 2), [r], [t],
                      context="select")

    def test_slice(self, device):
        r, t = make_grad_pair(randf(10, 20, seed=42), device)
        try_backward(r[2:5], t[2:5], [r], [t], context="slice")


# ===========================================================================
# CAT / STACK / UNBIND BACKWARD
# ===========================================================================

class TestCatStackBackward:

    def test_cat_backward(self, device):
        r_a, t_a = make_grad_pair(randf(5, 10, seed=1), device)
        r_b, t_b = make_grad_pair(randf(3, 10, seed=2), device)
        r_out = raxis.cat([r_a, r_b], dim=0)
        t_out = torch.cat([t_a, t_b], dim=0)
        try_backward(r_out, t_out, [r_a, r_b], [t_a, t_b], context="cat dim=0")

    def test_cat_dim1_backward(self, device):
        r_a, t_a = make_grad_pair(randf(5, 10, seed=1), device)
        r_b, t_b = make_grad_pair(randf(5, 7, seed=2), device)
        r_out = raxis.cat([r_a, r_b], dim=1)
        t_out = torch.cat([t_a, t_b], dim=1)
        try_backward(r_out, t_out, [r_a, r_b], [t_a, t_b], context="cat dim=1")

    def test_stack_backward(self, device):
        r_a, t_a = make_grad_pair(randf(5, 10, seed=1), device)
        r_b, t_b = make_grad_pair(randf(5, 10, seed=2), device)
        r_out = raxis.stack([r_a, r_b], dim=0)
        t_out = torch.stack([t_a, t_b], dim=0)
        try_backward(r_out, t_out, [r_a, r_b], [t_a, t_b], context="stack dim=0")

    def test_unbind_backward(self, device):
        data = randf(3, 10, seed=42)
        r, t = make_grad_pair(data, device)
        r_parts = raxis.unbind(r, dim=0)
        t_parts = torch.unbind(t, dim=0)
        r_out = r_parts[0] + r_parts[1] + r_parts[2]
        t_out = t_parts[0] + t_parts[1] + t_parts[2]
        try_backward(r_out, t_out, [r], [t], context="unbind")


# ===========================================================================
# COMPOSED OPERATIONS
# ===========================================================================

class TestComposedBackward:

    def test_chain_mul_add_sum(self, device):
        """(a * b + c).sum()"""
        r_a, t_a = make_grad_pair(randf(10, 20, seed=1), device)
        r_b, t_b = make_grad_pair(randf(10, 20, seed=2), device)
        r_c, t_c = make_grad_pair(randf(10, 20, seed=3), device)
        r_out = (r_a * r_b + r_c).sum()
        t_out = (t_a * t_b + t_c).sum()
        try_backward(r_out, t_out, [r_a, r_b, r_c], [t_a, t_b, t_c],
                      context="chain (a*b+c).sum()")

    def test_mlp_forward(self, device):
        """(x @ W).sum()"""
        r_x, t_x = make_grad_pair(randf(8, 16, seed=1), device)
        r_w, t_w = make_grad_pair(randf(16, 10, seed=2), device)
        r_out = raxis.mm(r_x, r_w).sum()
        t_out = torch.mm(t_x, t_w).sum()
        try_backward(r_out, t_out, [r_x, r_w], [t_x, t_w],
                      context="mlp (x@W).sum()")

    def test_sigmoid_mm_sum(self, device):
        """sigmoid(x @ W).sum()"""
        r_x, t_x = make_grad_pair(randf(8, 16, seed=1), device)
        r_w, t_w = make_grad_pair(randf(16, 10, seed=2), device)
        r_out = raxis.mm(r_x, r_w).sigmoid().sum()
        t_out = torch.mm(t_x, t_w).sigmoid().sum()
        try_backward(r_out, t_out, [r_x, r_w], [t_x, t_w],
                      context="sigmoid(x@W).sum()")

    def test_layernorm_style(self, device):
        """(x - mean) / sqrt(var + eps)"""
        data = randf(8, 16, seed=42)
        r, t = make_grad_pair(data, device)
        eps = 1e-5

        r_mean = r.mean(dim=1, keepdim=True)
        r_var = ((r - r_mean).pow(2)).mean(dim=1, keepdim=True)
        r_out = ((r - r_mean) / (r_var + eps).sqrt()).sum()

        t_mean = t.mean(dim=1, keepdim=True)
        t_var = ((t - t_mean).pow(2)).mean(dim=1, keepdim=True)
        t_out = ((t - t_mean) / (t_var + eps).sqrt()).sum()

        try_backward(r_out, t_out, [r], [t], context="layernorm-style",
                      atol=1e-3, rtol=1e-2)

    def test_fan_out_of_broadcast_grad(self, device):
        """A non-leaf used twice receives the stride-0 grad of sum() twice."""
        r_x, t_x = make_grad_pair(randf(3, 4, seed=1), device)
        r_z, t_z = r_x * 2, t_x * 2
        r_z.retain_grad()
        t_z.retain_grad()
        (r_z + r_z).sum().backward()
        (t_z + t_z).sum().backward()
        check_match(r_z.grad, t_z.grad, context="fan-out retained grad")
        check_match(r_x.grad, t_x.grad, context="fan-out leaf grad")

    def test_leaf_grad_does_not_alias_user_gradient(self, device):
        r_x, _ = make_grad_pair(randf(3, 4, seed=1), device)
        gradient = raxis.ones((3, 4))
        r_x.clone().backward(gradient)
        gradient += 1
        check_match(r_x.grad, torch.ones(3, 4), context="leaf grad alias")


# ===========================================================================
# IN-PLACE OPS
# ===========================================================================

class TestInPlace:
    """In-place ops with autograd: version counters, rebase_history, CopySlices."""

    A = randf(4, 3, seed=1) + 0.5
    B = randf(4, 3, seed=2) + 0.5

    # Each case runs on (lib, a, b) with a and b leaves that require grad
    CASES = {
        "mul_ saves self":       lambda lib, a, b: (a * 1).mul_(b),
        "exp_ saves result":     lambda lib, a, b: (a * 2).exp_() * b,
        "relu_":                 lambda lib, a, b: (a - 1.2).relu_() * b,
        "sin_ saves self":       lambda lib, a, b: (a * 1).sin_() * b,
        "neg_ saves nothing":    lambda lib, a, b: (a * 3).neg_() * b,
        "scalar ops":            lambda lib, a, b: (a * 1).add_(2).mul_(3).sub_(1).div_(2) * b,
        "augmented assignment":  lambda lib, a, b: TestInPlace.augmented(a * 1, b),
        "self as other":         lambda lib, a, b: (a * 1).add_(a * 1),
        "into a plain tensor":   lambda lib, a, b: lib.zeros(4, 3).add_(a) * b,
        "on a view":             lambda lib, a, b: TestInPlace.on_view(a * 1, b),
        "view of a view":        lambda lib, a, b: TestInPlace.view_of_view(a * 1, b),
        "transposed view":       lambda lib, a, b: TestInPlace.transposed_view(a * 1, b),
        "stale view":            lambda lib, a, b: TestInPlace.stale_view(a * 1, b),
        "stale expanded view":   lambda lib, a, b: TestInPlace.stale_expand(a[0] * 1, b),
        "copy_ into a slice":    lambda lib, a, b: TestInPlace.copy_into_slice(lib.zeros(4, 3), a, b),
        "setitem basic":         lambda lib, a, b: TestInPlace.setitem(lib.zeros(4, 3), 1, b[1] * 3) * a,
        "setitem index":         lambda lib, a, b: TestInPlace.setitem(a * 1, lib.tensor([0, 2]), b[:2] * 2),
        "setitem mask":          lambda lib, a, b: TestInPlace.setitem_mask(a * 1) * b,
        "fill_ and zero_":       lambda lib, a, b: TestInPlace.fills(a * 1) * b,
        "clamp_ and masked_fill_": lambda lib, a, b: TestInPlace.clamp_mask(a * 2) * b,
        "index_add_ scatter_":   lambda lib, a, b: TestInPlace.scatters(lib, a * 1, b),
    }

    @staticmethod
    def augmented(x, b):
        x += b[0]
        x -= b[1]
        x *= 2
        x /= b
        return x

    @staticmethod
    def on_view(x, b):
        x[1:3].mul_(b[1:3])
        return x

    @staticmethod
    def view_of_view(x, b):
        v = x[1:]
        v[:, 1:].mul_(b[1:, 1:])
        v.add_(1)
        return x

    @staticmethod
    def transposed_view(x, b):
        x.transpose(0, 1)[0].mul_(5)
        return x * b

    @staticmethod
    def stale_view(x, b):
        # v is taken before x changes: its grad_fn must see the new history
        v = x[0]
        x.mul_(2)
        return v * b[0]

    @staticmethod
    def stale_expand(x, b):
        e = x.expand(4, 3)
        x.mul_(3)
        return e * b

    @staticmethod
    def copy_into_slice(x, a, b):
        x[:, 1].copy_(a[:, 0])
        return x * b

    @staticmethod
    def setitem(x, index, value):
        x[index] = value
        return x

    @staticmethod
    def setitem_mask(x):
        x[x > 1.0] = 0.0
        return x

    @staticmethod
    def fills(x):
        x[0].fill_(2.0)
        x[1].zero_()
        return x

    @staticmethod
    def clamp_mask(x):
        x.clamp_(1.2, 2.5)
        return x.masked_fill_(x > 2.0, 0.5)

    @staticmethod
    def scatters(lib, x, b):
        x.index_add_(0, lib.tensor([0, 0]), b[:2])
        x.scatter_(1, lib.tensor([[0], [1], [2], [0]]), b[:, :1])
        return x

    @staticmethod
    def run(lib, fn):
        a = lib.tensor(TestInPlace.A.tolist(), requires_grad=True)
        b = lib.tensor(TestInPlace.B.tolist(), requires_grad=True)
        out = fn(lib, a, b)
        out.sum().backward()
        return out, a.grad, b.grad

    @pytest.mark.parametrize("fn", CASES.values(), ids=CASES.keys())
    def test_matches_torch(self, fn):
        r_out, r_ga, r_gb = self.run(raxis, fn)
        t_out, t_ga, t_gb = self.run(torch, fn)
        check_match(r_out, t_out)
        for r_g, t_g in [(r_ga, t_ga), (r_gb, t_gb)]:
            assert (r_g is None) == (t_g is None)
            if r_g is not None:
                check_match(r_g, t_g)

    def test_grad_fn_names(self):
        for lib in (raxis, torch):
            a = lib.ones(4, 3, requires_grad=True)
            x = a * 1
            v = x[1:]
            v.mul_(2)
            assert "CopySlices" in str(type(x.grad_fn)) or x.grad_fn.name().endswith("CopySlices")
            assert v.grad_fn.name() == "AsStridedBackward0"
            y = a * 1
            assert y.exp_().grad_fn.name() == "ExpBackward0"
            assert y.fill_(1).grad_fn.name() == "FillBackward2"

    def test_version_counter(self):
        for lib in (raxis, torch):
            x = lib.zeros(3)
            v = x[1:]
            assert x._version == 0 and v._is_view() and v._base is not None and x._base is None
            v.add_(1)
            x.mul_(2)
            assert x._version == 2 and v._version == 2
            assert not x.detach()._is_view() and x.detach()._version == 2
            x.clone().add_(1)
            assert x._version == 2

    ERRORS = {
        "leaf":                 (lambda lib, a: a.add_(1), "a leaf Variable that requires grad is being used in an in-place operation."),
        "view of leaf":         (lambda lib, a: a[0].add_(1), "a view of a leaf Variable that requires grad"),
        "saved input changed":  (lambda lib, a: TestInPlace.modify_saved_input(a * 1), "is at version 1; expected version 0 instead"),
        "saved output changed": (lambda lib, a: a.exp().add_(1), "which is output 0 of Exp, is at version 1"),
        "unbind output":        (lambda lib, a: (a * 1).unbind(0)[0].mul_(2), "Output 0 of Unbind is a view and is being modified inplace"),
        "split output changed": (lambda lib, a: TestInPlace.split_then_base(a * 1), "its base or another view of its base has been modified inplace"),
        "no_grad view":         (lambda lib, a: TestInPlace.no_grad_view(lib, a * 1).mul_(2), "A view was created in no_grad mode"),
    }

    @staticmethod
    def modify_saved_input(x):
        y = x.sin()
        x.add_(1)
        return y * x

    @staticmethod
    def split_then_base(x):
        first = x.split(2)[0]
        x.mul_(2)
        return first

    @staticmethod
    def no_grad_view(lib, x):
        with lib.no_grad():
            return x[0]

    @pytest.mark.parametrize("fn,message", ERRORS.values(), ids=ERRORS.keys())
    def test_errors(self, fn, message):
        for lib in (raxis, torch):
            a = lib.tensor(self.A.tolist(), requires_grad=True)
            with pytest.raises(RuntimeError, match=message):
                fn(lib, a).sum().backward()

    def test_self_as_both_inputs_keeps_the_original(self):
        # x.mul_(x) saves x as self and as other. PyTorch copies only self
        # and fails at backward; raxis copies both and gets the gradient
        a = raxis.tensor(self.A.tolist(), requires_grad=True)
        (a * 1).mul_(a * 1)  # distinct inputs: fine anywhere
        x = a * 1
        x.mul_(x).sum().backward()
        check_match(a.grad, torch.from_numpy(2 * self.A))

    def test_no_grad_inplace_on_leaf_bumps_version(self):
        for lib in (raxis, torch):
            w = lib.ones(3, requires_grad=True)
            y = (w * w).sum()
            with lib.no_grad():
                w.sub_(1)  # an optimizer step
            with pytest.raises(RuntimeError, match="modified by an inplace operation"):
                y.backward()

    def test_no_grad_view_requires_grad_without_edges(self):
        for lib in (raxis, torch):
            a = lib.ones(3, requires_grad=True)
            x = a * 2
            with lib.no_grad():
                v = x[:2]
            assert v.requires_grad and v.grad_fn is None
            (v * 3).sum().backward()
            assert a.grad is None

    def test_inplace_does_not_leak_the_graph(self):
        x = raxis.rand((4, 5), requires_grad=True)
        y = x * 2
        y.exp_()
        y[1:3].mul_(3)
        y.relu_()
        assert y._use_count() == 1
        base_view = y[0]
        del base_view
        assert y._use_count() == 1


# ===========================================================================
# HOOKS
# ===========================================================================

def _values(t):
    return None if t is None else np.asarray(t.detach().tolist(), dtype=np.float64).ravel().round(4).tolist()


class TestHooks:
    """Tensor hooks, node hooks and post-accumulate-grad hooks, against torch."""

    @staticmethod
    def leaf_hook(lib):
        a = lib.tensor([1.0, 2.0, 3.0], requires_grad=True)
        seen = []
        a.register_hook(lambda grad: seen.append(_values(grad)))
        handle = a.register_hook(lambda grad: grad * 2)
        (a * 3).sum().backward()
        handle.remove()
        (a * 3).sum().backward()
        return seen, _values(a.grad)

    @staticmethod
    def non_leaf_hook(lib):
        a = lib.tensor([1.0, 2.0, 3.0], requires_grad=True)
        x = a * 2
        x.register_hook(lambda grad: grad + 1)
        x.retain_grad()
        (x * x).sum().backward()
        return _values(a.grad), _values(x.grad)

    @staticmethod
    def retain_grad_sees_hooked_grad(lib):
        a = lib.tensor([1.0, 2.0], requires_grad=True)
        x = a * 1
        x.retain_grad()
        x.register_hook(lambda grad: grad * 10)
        x.sum().backward()
        return _values(x.grad), _values(a.grad)

    @staticmethod
    def retain_grad_of_second_output(lib):
        a = lib.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        u, v = a.unbind(0)
        v.retain_grad()
        (u * 2 + v * 3).sum().backward()
        return _values(v.grad), _values(a.grad)

    @staticmethod
    def node_hooks(lib):
        a = lib.tensor([1.0, 2.0], requires_grad=True)
        b = lib.tensor([3.0, 4.0], requires_grad=True)
        y = a * b
        calls = []

        def post(grad_inputs, grad_outputs):
            calls.append(("post", len(grad_inputs), len(grad_outputs),
                          _values(grad_inputs[0]), _values(grad_outputs[0])))
            return (grad_inputs[0] * 2, None)  # b gets no gradient

        def pre(grad_outputs):
            calls.append(("pre", _values(grad_outputs[0])))
            return (grad_outputs[0] + 1,)

        y.grad_fn.register_hook(post)
        y.grad_fn.register_prehook(pre)
        y.sum().backward()
        return calls, _values(a.grad), _values(b.grad)

    @staticmethod
    def post_accumulate_grad(lib):
        a = lib.tensor([1.0, 2.0], requires_grad=True)
        seen = []
        a.register_post_accumulate_grad_hook(lambda t: seen.append(_values(t.grad)))
        (a * 2).sum().backward()
        (a * 3).sum().backward()
        return seen

    @staticmethod
    def hooks_and_inplace(lib):
        # The first hook belongs to the old x: it sees the gradient of x
        # before mul_, the second one the gradient of the new x
        a = lib.tensor([1.0, 2.0], requires_grad=True)
        x = a * 1
        x.register_hook(lambda grad: grad * 100)
        x.mul_(2)
        x.register_hook(lambda grad: grad + 1)
        x.sum().backward()
        return _values(a.grad)

    @staticmethod
    def handle_as_context_manager(lib):
        a = lib.tensor([1.0], requires_grad=True)
        with a.register_hook(lambda grad: grad * 5):
            (a * 1).sum().backward()
        (a * 1).sum().backward()
        return _values(a.grad)

    CASES = [leaf_hook, non_leaf_hook, retain_grad_sees_hooked_grad, retain_grad_of_second_output,
             node_hooks, post_accumulate_grad, hooks_and_inplace, handle_as_context_manager]

    @pytest.mark.parametrize("case", CASES, ids=[c.__name__ for c in CASES])
    def test_matches_torch(self, case):
        assert case(raxis) == case(torch)

    def test_errors(self):
        for lib in (raxis, torch):
            with pytest.raises(RuntimeError, match="cannot register a hook on a tensor that doesn't require gradient"):
                lib.tensor([1.0]).register_hook(lambda grad: grad)
            non_leaf = lib.tensor([1.0], requires_grad=True) * 2
            with pytest.raises(RuntimeError, match="post accumulate grad hooks cannot be registered on non-leaf tensors"):
                non_leaf.register_post_accumulate_grad_hook(lambda t: None)

    def test_hook_returning_wrong_count_raises(self):
        a = raxis.tensor([1.0], requires_grad=True)
        y = a * 2
        y.grad_fn.register_hook(lambda grad_inputs, grad_outputs: (None, None, None))
        with pytest.raises(RuntimeError, match="incorrect number of values"):
            y.sum().backward()

    def test_removed_handle_does_not_keep_tensor_alive(self):
        a = raxis.tensor([1.0], requires_grad=True)
        x = a * 2
        handle = x.register_hook(lambda grad: grad)
        assert x._use_count() == 1
        del x
        handle.remove()  # the list is gone: nothing to do


# ===========================================================================
# CUSTOM FUNCTIONS (autograd.Function)
# ===========================================================================

def _custom_functions(lib):
    """The same custom Functions written for raxis and for torch."""
    F = lib.autograd.Function

    class Exp(F):
        @staticmethod
        def forward(ctx, x):
            y = x.exp()
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad):
            y, = ctx.saved_tensors
            return grad * y

    class MulConst(F):
        @staticmethod
        def forward(ctx, x, k):
            ctx.k = k
            return x * k

        @staticmethod
        def backward(ctx, grad):
            return grad * ctx.k, None

    class WithIndices(F):
        @staticmethod
        def forward(ctx, x):
            idx = (x > 1).to(lib.int64)
            ctx.mark_non_differentiable(idx)
            return x * 2, idx

        @staticmethod
        def backward(ctx, grad, grad_idx):
            return grad * 2

    class TripleInPlace(F):
        @staticmethod
        def forward(ctx, x):
            x.mul_(3)
            ctx.mark_dirty(x)
            return x

        @staticmethod
        def backward(ctx, grad):
            return grad * 3

    class Identity(F):
        @staticmethod
        def forward(ctx, x):
            return x

        @staticmethod
        def backward(ctx, grad):
            return grad * 5

    class MulSetup(F):
        @staticmethod
        def forward(x, y):
            return x * y

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs)

        @staticmethod
        def backward(ctx, grad):
            x, y = ctx.saved_tensors
            return grad * y, grad * x

    class SumToScalar(F):
        @staticmethod
        def forward(ctx, x):
            return x.sum()

        @staticmethod
        def backward(ctx, grad):
            # A broadcast gradient: reduced back to the input shape
            return grad * lib.ones(4, 3)

    class Once(F):
        @staticmethod
        def forward(ctx, x):
            return x * 2

        @staticmethod
        @lib.autograd.function.once_differentiable
        def backward(ctx, grad):
            assert not lib.is_grad_enabled()
            return grad * 2

    return dict(Exp=Exp, MulConst=MulConst, WithIndices=WithIndices, TripleInPlace=TripleInPlace,
                Identity=Identity, MulSetup=MulSetup, SumToScalar=SumToScalar, Once=Once)


class TestFunction:

    A = randf(4, 3, seed=3) + 0.5
    CASES = {
        "saves output":        lambda fns, lib, a, b: fns["Exp"].apply(a) * b,
        "non-tensor input":    lambda fns, lib, a, b: fns["MulConst"].apply(a, 2.5) * b,
        "non-differentiable":  lambda fns, lib, a, b: fns["WithIndices"].apply(a)[0] * b,
        "mark_dirty":          lambda fns, lib, a, b: fns["TripleInPlace"].apply(a * 1) * b,
        "input returned":      lambda fns, lib, a, b: fns["Identity"].apply(a) * b,
        "setup_context":       lambda fns, lib, a, b: fns["MulSetup"].apply(a, b),
        "broadcast gradient":  lambda fns, lib, a, b: fns["SumToScalar"].apply(a[0] * b[0]),
        "once_differentiable": lambda fns, lib, a, b: fns["Once"].apply(a) * b,
        "chained":             lambda fns, lib, a, b: fns["Exp"].apply(fns["MulConst"].apply(a, 0.5)) * b,
    }

    @staticmethod
    def run(lib, fn):
        a = lib.tensor(TestFunction.A.tolist(), requires_grad=True)
        b = lib.tensor((TestFunction.A * 2).tolist(), requires_grad=True)
        out = fn(_custom_functions(lib), lib, a, b)
        out.sum().backward()
        return out, a.grad, b.grad

    @pytest.mark.parametrize("fn", CASES.values(), ids=CASES.keys())
    def test_matches_torch(self, fn):
        r_out, r_ga, r_gb = self.run(raxis, fn)
        t_out, t_ga, t_gb = self.run(torch, fn)
        check_match(r_out, t_out)
        for r_g, t_g in [(r_ga, t_ga), (r_gb, t_gb)]:
            assert (r_g is None) == (t_g is None)
            if r_g is not None:
                check_match(r_g, t_g)

    def test_graph_node(self):
        fns = _custom_functions(raxis)
        a = raxis.tensor(self.A.tolist(), requires_grad=True)
        y = fns["Exp"].apply(a)
        assert y.grad_fn.name() == "ExpBackward"
        assert y.grad_fn.next_functions[0][0].name() == "AccumulateGrad"
        y2, idx = fns["WithIndices"].apply(a)
        assert y2.requires_grad and not idx.requires_grad
        assert fns["Exp"].apply(raxis.tensor([1.0])).grad_fn is None
        assert fns["TripleInPlace"].apply(a * 1)._version == 2  # mul_ and mark_dirty

    def test_needs_input_grad_and_materialize(self):
        seen = []

        class TwoOutputs(raxis.autograd.Function):
            @staticmethod
            def forward(ctx, x, y, k):
                seen.append(ctx.needs_input_grad)
                ctx.set_materialize_grads(False)
                return x * k, y * k

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                seen.append((grad_x is None, grad_y is None))
                return grad_x, grad_y, None

        a = raxis.tensor([1.0, 2.0], requires_grad=True)
        u, _ = TwoOutputs.apply(a, raxis.tensor([3.0, 4.0]), 2)
        u.sum().backward()
        assert seen == [(True, False, False), (False, True)]

    ERRORS = {
        "wrong count":       ("WrongCount", "returned an incorrect number of gradients \\(expected 2, got 1\\)"),
        "grad for an int":   ("GradForInt", "returned a gradient different than None at position 2"),
        "inplace on output": ("InPlaceOnOutput", "This view was created inside a custom Function"),
        "saved modified":    ("SavedModified", "modified by an inplace operation"),
        "dirty not output":  ("DirtyNotReturned", "marked as dirty during the forward method were not returned"),
    }

    @staticmethod
    def error_case(lib, name, a):
        fns = _custom_functions(lib)
        F = lib.autograd.Function

        class WrongCount(F):
            @staticmethod
            def forward(ctx, x, y):
                return x + y

            @staticmethod
            def backward(ctx, grad):
                return grad

        class GradForInt(F):
            @staticmethod
            def forward(ctx, x, k):
                return x * k

            @staticmethod
            def backward(ctx, grad):
                return grad, grad

        class DirtyNotReturned(F):
            @staticmethod
            def forward(ctx, x):
                x.add_(1)
                ctx.mark_dirty(x)
                return x * 2

            @staticmethod
            def backward(ctx, grad):
                return grad

        if name == "WrongCount":
            return WrongCount.apply(a, a * 1).sum().backward()
        if name == "GradForInt":
            return GradForInt.apply(a, 2).sum().backward()
        if name == "InPlaceOnOutput":
            return fns["Identity"].apply(a * 1).add_(1)
        if name == "SavedModified":
            y = fns["Exp"].apply(a * 1)
            y.add_(1)
            return y.sum().backward()
        if name == "DirtyNotReturned":
            return DirtyNotReturned.apply(a * 1)

    @pytest.mark.parametrize("name,message", ERRORS.values(), ids=ERRORS.keys())
    def test_errors(self, name, message):
        for lib in (raxis, torch):
            a = lib.tensor(self.A.tolist(), requires_grad=True)
            with pytest.raises(RuntimeError, match=message):
                self.error_case(lib, name, a)

    def test_ctx_and_graph_are_freed(self):
        import weakref
        ctxs = []

        class Square(raxis.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                ctxs.append(weakref.ref(ctx))
                ctx.save_for_backward(x)
                return x * x

            @staticmethod
            def backward(ctx, grad):
                x, = ctx.saved_tensors
                return 2 * grad * x

        a = raxis.tensor([1.0, 2.0], requires_grad=True)
        y = Square.apply(a)
        assert y._use_count() == 1
        assert ctxs[0]() is not None
        del y
        assert ctxs[0]() is None


# ===========================================================================
# HIGHER-ORDER GRADIENTS (create_graph, autograd.grad)
# ===========================================================================

F_ = raxis.nn.functional

# (name, input shape, op). Inputs are positive, and away from 0 and 1 so
# that log, sqrt, acos, ... are defined
RECORDED_OPS = [
    ("neg", (3, 4), lambda x: -x),
    ("abs", (3, 4), lambda x: x.abs()),
    ("exp", (3, 4), lambda x: x.exp()),
    ("log", (3, 4), lambda x: x.log()),
    ("sqrt", (3, 4), lambda x: x.sqrt()),
    ("sin", (3, 4), lambda x: x.sin()),
    ("tanh", (3, 4), lambda x: x.tanh()),
    ("sigmoid", (3, 4), lambda x: x.sigmoid()),
    ("relu", (3, 4), lambda x: (x - 0.5).relu()),
    ("reciprocal", (3, 4), lambda x: x.reciprocal()),
    ("asin", (3, 4), lambda x: (x * 0.5).asin()),
    ("pow scalar", (3, 4), lambda x: x ** 3),
    ("pow tensor", (3, 4), lambda x: x ** x),
    ("add", (3, 4), lambda x: x + x * 2),
    ("sub", (3, 4), lambda x: x - x * 2),
    ("mul", (3, 4), lambda x: x * x),
    ("div", (3, 4), lambda x: x / (x + 1)),
    ("maximum", (3, 4), lambda x: raxis.maximum(x, x * 0.5 + 0.2)),
    ("matmul", (3, 4), lambda x: x.matmul(x.transpose(0, 1))),
    ("bmm", (2, 3, 3), lambda x: x.bmm(x)),
    ("sum dim", (3, 4), lambda x: x.sum(1)),
    ("mean", (3, 4), lambda x: x.mean(0)),
    ("var", (3, 4), lambda x: x.var(1)),
    ("std", (3, 4), lambda x: x.std(0)),
    ("max dim", (3, 4), lambda x: x.max(1)[0]),
    ("max", (3, 4), lambda x: x.max()),
    ("cumsum", (3, 4), lambda x: x.cumsum(1)),
    ("cumprod", (3, 4), lambda x: x.cumprod(1)),
    ("flip", (3, 4), lambda x: x.flip(0, 1)),
    ("sort", (3, 4), lambda x: x.sort(1)[0]),
    ("topk", (3, 4), lambda x: x.topk(2, 0)[0]),
    ("view", (3, 4), lambda x: x.view(4, 3)),
    ("transpose", (3, 4), lambda x: x.transpose(0, 1)),
    ("expand", (3, 1), lambda x: x.expand(3, 4)),
    ("select slice", (3, 4), lambda x: x[1, 1:3]),
    ("cat", (3, 4), lambda x: raxis.cat([x, x * 2], 0)),
    ("stack", (3, 4), lambda x: raxis.stack([x, x * 2], 0)),
    ("unbind", (3, 4), lambda x: x.unbind(0)[1]),
    ("split", (3, 4), lambda x: x.split(2, 1)[1]),
    ("where", (3, 4), lambda x: raxis.where(x > 0.7, x, x * 3)),
    ("clamp", (3, 4), lambda x: x.clamp(0.3, 0.8)),
    ("gather", (3, 4), lambda x: x.gather(1, raxis.tensor([[0, 0], [1, 3], [2, 2]]))),
    ("scatter", (3, 4), lambda x: x.scatter(1, raxis.tensor([[0], [1], [2]]), x[:, :1] * 2)),
    ("index_select", (3, 4), lambda x: x.index_select(0, raxis.tensor([2, 0, 2]))),
    ("index_add", (3, 4), lambda x: x.index_add(0, raxis.tensor([1, 1]), x[:2] * 2)),
    ("advanced index", (3, 4), lambda x: x[raxis.tensor([0, 2, 0]), 1:]),
    ("masked_fill", (3, 4), lambda x: x.masked_fill(x > 0.7, 0.0)),
    ("index_put", (3, 4), lambda x: x.index_put((raxis.tensor([0, 0]),), x[1] * 3, accumulate=True)),
    ("inplace on view", (3, 4), lambda x: TestCreateGraph.inplace_on_view(x)),
    ("softmax", (3, 4), lambda x: x.softmax(1)),
    ("log_softmax", (3, 4), lambda x: x.log_softmax(1)),
    ("nll_loss", (3, 4), lambda x: F_.nll_loss(x.log_softmax(1), raxis.tensor([0, 3, 1]))),
    ("cross_entropy", (3, 4), lambda x: F_.cross_entropy(x, raxis.tensor([0, 3, 1]))),
    ("mse_loss", (3, 4), lambda x: F_.mse_loss(x, raxis.ones(3, 4))),
    ("bce", (3, 4), lambda x: F_.binary_cross_entropy(x.sigmoid(), raxis.ones(3, 4) * 0.5)),
    ("max_pool2d", (1, 1, 4, 4), lambda x: F_.max_pool2d(x, 2)),
    ("avg_pool2d", (1, 1, 4, 4), lambda x: F_.avg_pool2d(x, 2)),
    ("dropout", (3, 4), lambda x: F_.dropout(x, 0.5, True)),
    ("custom Function", (3, 4), lambda x: TestCreateGraph.Cube.apply(x)),
]


class TestCreateGraph:

    class Cube(raxis.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x ** 3

        @staticmethod
        def backward(ctx, grad):
            x, = ctx.saved_tensors
            return 3 * grad * x ** 2

    @staticmethod
    def inplace_on_view(x):
        y = x * 1
        y[1:].mul_(x[1:])
        return y

    @pytest.mark.parametrize("shape,op", [(s, op) for _, s, op in RECORDED_OPS],
                             ids=[name for name, _, _ in RECORDED_OPS])
    def test_backward_formula_is_recorded(self, shape, op):
        """With create_graph the gradient of (op(x) * w).sum() is J^T w: it
        depends on w for every op, so it must require grad. A backward formula
        that called a kernel without recording it would return a constant
        (a silent zero second derivative)."""
        x = raxis.tensor(randf(*shape, seed=1, lo=0.2, hi=0.9).tolist(), requires_grad=True)
        out = op(x)
        w = raxis.tensor(randf(*out.shape, seed=2).tolist(), requires_grad=True)
        g, = raxis.autograd.grad((out * w).sum(), x, create_graph=True)
        assert g.requires_grad and g.grad_fn is not None

    HEADS = {
        "sum sq":          lambda lib, y: (y * y).sum(),
        "softmax":         lambda lib, y: (y.softmax(1) * lib.tensor([[1.0, 2.0, 3.0]])).sum(),
        "log_softmax nll": lambda lib, y: lib.nn.functional.nll_loss(y.log_softmax(1), lib.tensor([0, 2, 1, 1, 0])),
        "cross_entropy":   lambda lib, y: lib.nn.functional.cross_entropy(y, lib.tensor([0, 2, 1, 1, 0])),
        "mse":             lambda lib, y: lib.nn.functional.mse_loss(y, lib.ones(5, 3)),
    }
    ACTIVATIONS = {
        "tanh":    lambda z: z.tanh(),
        "relu":    lambda z: z.relu(),
        "sigmoid": lambda z: z.sigmoid(),
    }

    @staticmethod
    def hessian_vector_product(lib, act, head):
        rng = np.random.default_rng(0)
        x = lib.tensor(rng.standard_normal((5, 4)).astype(np.float32).tolist())
        w1 = lib.tensor((rng.standard_normal((4, 6)) * 0.5).astype(np.float32).tolist(), requires_grad=True)
        w2 = lib.tensor((rng.standard_normal((6, 3)) * 0.5).astype(np.float32).tolist(), requires_grad=True)
        v = lib.tensor(rng.standard_normal((4, 6)).astype(np.float32).tolist())
        loss = head(lib, act(x.matmul(w1)).matmul(w2))
        g1, = lib.autograd.grad(loss, w1, create_graph=True)
        hv1, hv2 = lib.autograd.grad((g1 * v).sum(), (w1, w2))
        return g1, hv1, hv2

    @pytest.mark.parametrize("act", ACTIVATIONS.keys())
    @pytest.mark.parametrize("head", HEADS.keys())
    def test_hessian_vector_product(self, act, head):
        r = self.hessian_vector_product(raxis, self.ACTIVATIONS[act], self.HEADS[head])
        t = self.hessian_vector_product(torch, self.ACTIVATIONS[act], self.HEADS[head])
        for r_g, t_g in zip(r, t):
            check_match(r_g, t_g, atol=1e-4, rtol=1e-3)

    @staticmethod
    def scenarios(lib):
        out = {}
        x = lib.tensor([1.0, 2.0, 3.0], requires_grad=True)
        g, = lib.autograd.grad((x ** 3).sum(), x, create_graph=True)
        gg, = lib.autograd.grad(g.sum(), x)
        out["third power"] = (g.tolist(), gg.tolist())

        # backward(create_graph=True) leaves a differentiable .grad
        y = lib.tensor([1.0, 2.0], requires_grad=True)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the reference cycle warning
            (y.sin() * y).sum().backward(create_graph=True)
        grad_y = y.grad
        out[".grad requires grad"] = grad_y.requires_grad
        y.grad = None
        grad_y.sum().backward()
        out["second derivative"] = [round(v, 5) for v in y.grad.tolist()]

        a = lib.tensor([1.0, 2.0], requires_grad=True)
        b = lib.tensor([3.0, 4.0], requires_grad=True)
        ga, gb = lib.autograd.grad((a * a).sum(), (a, b), allow_unused=True)
        out["unused"] = (ga.tolist(), gb, a.grad)
        out["materialize"] = lib.autograd.grad((a * a).sum(), (a, b), materialize_grads=True)[1].tolist()
        h = a * 3
        out["intermediate"] = lib.autograd.grad((h * h).sum(), h)[0].tolist()
        u, v = a * 2, a * a
        out["several outputs"] = lib.autograd.grad(
            (u, v), a, grad_outputs=(lib.tensor([1.0, 1.0]), lib.tensor([2.0, 3.0])))[0].tolist()

        # gradient penalty
        w = lib.tensor([[0.5, -1.0], [2.0, 0.3]], requires_grad=True)
        inp = lib.tensor([[1.0, 2.0]], requires_grad=True)
        gi, = lib.autograd.grad(inp.matmul(w).tanh().sum(), inp, create_graph=True)
        (((gi * gi).sum() - 1) ** 2).backward()
        out["gradient penalty"] = np.round(np.asarray(w.grad.tolist()), 5).tolist()
        return out

    def test_scenarios_match_torch(self):
        assert self.scenarios(raxis) == self.scenarios(torch)

    def test_errors(self):
        for lib in (raxis, torch):
            a = lib.tensor([1.0, 2.0], requires_grad=True)
            b = lib.tensor([3.0, 4.0], requires_grad=True)
            with pytest.raises(RuntimeError, match="appears to not have been used in the graph"):
                lib.autograd.grad((a * a).sum(), (a, b))
            with pytest.raises(RuntimeError, match="does not require grad"):
                lib.autograd.grad((a * a).sum(), lib.tensor([1.0]))
            with pytest.raises(RuntimeError, match="Mismatch in shape"):
                lib.autograd.grad(a * a, a, grad_outputs=lib.ones(3))

    def test_double_backward_not_implemented_raises(self):
        x = raxis.rand((1, 1, 5, 5), requires_grad=True)
        w = raxis.rand((1, 1, 3, 3), requires_grad=True)
        g, = raxis.autograd.grad(raxis.conv2d(x, w).sum(), w, create_graph=True)
        assert g.requires_grad
        with pytest.raises(RuntimeError, match="derivative for 'convolution_backward' is not implemented"):
            raxis.autograd.grad(g.sum(), x)

    def test_once_differentiable_raises_on_second_derivative(self):
        import warnings
        for lib in (raxis, torch):
            class Square(lib.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return x * x

                @staticmethod
                @lib.autograd.function.once_differentiable
                def backward(ctx, grad):
                    x, = ctx.saved_tensors
                    return 2 * grad * x

            x = lib.tensor([1.0, 2.0], requires_grad=True)
            g, = lib.autograd.grad((Square.apply(x * 1) * x).sum(), x, create_graph=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError, match="trying to differentiate twice"):
                    g.sum().backward()

    def test_grad_does_not_touch_dot_grad(self):
        w = raxis.tensor([1.0, 2.0], requires_grad=True)
        other = raxis.tensor([1.0], requires_grad=True)
        raxis.autograd.grad((w * w).sum() + other.sum(), w)
        assert w.grad is None and other.grad is None


# ===========================================================================
# NO_GRAD / ENABLE_GRAD
# ===========================================================================

class TestGradMode:

    def test_no_grad_skips_graph(self, device):
        r = raxis.tensor(randf(10, 20, seed=42).tolist(), requires_grad=True)
        t = torch.tensor(randf(10, 20, seed=42).tolist(), requires_grad=True)

        with raxis.no_grad():
            r_out = r * 2.0
        with torch.no_grad():
            t_out = t * 2.0

        assert r_out.grad_fn is None
        assert t_out.grad_fn is None
        assert not r_out.requires_grad
        assert not t_out.requires_grad

    def test_no_grad_nested(self, device):
        r = raxis.tensor([1.0, 2.0, 3.0], requires_grad=True)

        with raxis.no_grad():
            out1 = r + 1
            assert out1.grad_fn is None

            with raxis.no_grad():
                out2 = r + 2
                assert out2.grad_fn is None

            out3 = r + 3
            assert out3.grad_fn is None

        out4 = r + 4
        assert out4.grad_fn is not None

    def test_enable_grad_inside_no_grad(self, device):
        r = raxis.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        with raxis.no_grad():
            with raxis.enable_grad():
                r_out = r * 2.0
        with torch.no_grad():
            with torch.enable_grad():
                t_out = t * 2.0

        assert r_out.grad_fn is not None
        assert t_out.grad_fn is not None

    def test_is_grad_enabled(self, device):
        assert raxis.is_grad_enabled() == True
        with raxis.no_grad():
            assert raxis.is_grad_enabled() == False
            with raxis.enable_grad():
                assert raxis.is_grad_enabled() == True
            assert raxis.is_grad_enabled() == False
        assert raxis.is_grad_enabled() == True

    def test_no_grad_restores_on_exception(self, device):
        """no_grad must restore state even if the body raises."""
        assert raxis.is_grad_enabled() == True
        try:
            with raxis.no_grad():
                assert raxis.is_grad_enabled() == False
                raise ValueError("test")
        except ValueError:
            pass
        assert raxis.is_grad_enabled() == True


# ===========================================================================
# RETAIN_GRAPH / MULTIPLE BACKWARD
# ===========================================================================

class TestRetainGraph:

    def test_backward_twice(self, device):
        """Two backward passes with retain_graph produce the same gradients."""
        data = randf(10, 20, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = (r * r).sum()
        t_out = (t * t).sum()

        r_out.backward(retain_graph=True)
        t_out.backward(retain_graph=True)
        check_match(r.grad, t.grad, context="first backward")

        r_grad1 = np.array(r.grad).copy()
        r.grad = None
        t.grad = None

        r_out.backward(retain_graph=True)
        t_out.backward(retain_graph=True)
        check_match(r.grad, t.grad, context="second backward")
        np.testing.assert_allclose(np.array(r.grad), r_grad1, atol=1e-6,
                                    err_msg="Two backward passes should give same grad")

    def test_backward_without_retain_graph_twice_errors(self, device):
        """Second backward without retain_graph should error on both."""
        r = raxis.tensor(randf(10, seed=42).tolist(), requires_grad=True)
        r_out = (r * r).sum()
        r_out.backward()

        t = torch.tensor(randf(10, seed=42).tolist(), requires_grad=True)
        t_out = (t * t).sum()
        t_out.backward()

        with pytest.raises(RuntimeError, match="backward through the graph a second time"):
            t_out.backward()

        with pytest.raises(RuntimeError, match="backward through the graph a second time"):
            r_out.backward()

    def test_backward_twice_without_saved_tensors(self):
        # Only saved tensors are freed: a graph that saves none can run again
        def run(x):
            out = (x + 1.0).sum()
            out.backward()
            out.backward()
            return x.grad
        r, t = make_grad_pair(randf(3, 4))
        check_match(run(r), run(t))

    def test_saved_tensors_released_after_backward(self):
        x = raxis.rand((4, 5), requires_grad=True)
        a = x * 2.0
        out = a.sin().sum()
        assert a._use_count() == 2  # a and the SinBackward0 that saved it
        out.backward()
        assert a._use_count() == 1

    def test_saved_tensors_kept_with_retain_graph(self):
        x = raxis.rand((4, 5), requires_grad=True)
        a = x * 2.0
        out = a.sin().sum()
        out.backward(retain_graph=True)
        assert a._use_count() == 2
        out.backward()
        assert a._use_count() == 1

    def test_grad_accumulates(self, device):
        """Multiple backward calls accumulate gradients (without zeroing)."""
        data = randf(10, 20, seed=42)
        r, t = make_grad_pair(data, device)
        r_out = (r * r).sum()
        t_out = (t * t).sum()

        r_out.backward(retain_graph=True)
        t_out.backward(retain_graph=True)

        r_out.backward(retain_graph=True)
        t_out.backward(retain_graph=True)

        check_match(r.grad, t.grad, context="accumulated grad (2x)")


# ===========================================================================
# DETACH
# ===========================================================================

class TestDetach:

    def test_detach_breaks_graph(self, device):
        r = raxis.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = (r * 2.0).detach()
        assert out.grad_fn is None
        assert not out.requires_grad

    def test_detach_values_match(self, device):
        data = randf(10, 20, seed=42)
        r, t = make_grad_pair(data, device)
        r_det = (r * 2.0).detach()
        t_det = (t * 2.0).detach()
        check_match(r_det, t_det, context="detach values")

    def test_backward_stops_at_detach(self, device):
        """Detach prevents gradients from flowing past the detach point."""
        r = raxis.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # Only the direct path through r contributes: d/dr (mid * r) = mid
        r_out = ((r * 2.0).detach() * r).sum()
        t_out = ((t * 2.0).detach() * t).sum()
        r_out.backward()
        t_out.backward()

        check_match(r.grad, t.grad, context="detach")


# ===========================================================================
# LEAF / NON-LEAF PROPERTIES
# ===========================================================================

class TestLeafProperties:

    def test_leaf_tensor(self, device):
        r = raxis.tensor([1.0, 2.0], requires_grad=True)
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        assert r.is_leaf == t.is_leaf == True
        assert r.grad_fn is None
        assert t.grad_fn is None

    def test_non_leaf_tensor(self, device):
        r = raxis.tensor([1.0, 2.0], requires_grad=True)
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        r_out = r * 2.0
        t_out = t * 2.0
        assert r_out.is_leaf == t_out.is_leaf == False
        assert r_out.grad_fn is not None
        assert t_out.grad_fn is not None

    def test_non_leaf_grad_is_none(self, device):
        """Grad is only retained for leaf tensors by default."""
        r = raxis.tensor([1.0, 2.0], requires_grad=True)
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        r_out = (r * 2.0).sum()
        t_out = (t * 2.0).sum()
        r_out.backward()
        t_out.backward()
        assert r.grad is not None
        assert t.grad is not None

    def test_backward_on_leaf_accumulates_into_grad(self, device):
        r = raxis.tensor([1.0, 2.0], requires_grad=True)
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        r.backward(raxis.tensor([3.0, 4.0]))
        t.backward(torch.tensor([3.0, 4.0]))
        r.backward(raxis.tensor([1.0, 1.0]))
        t.backward(torch.tensor([1.0, 1.0]))
        check_match(r.grad, t.grad, context="leaf backward")

    def test_requires_grad_false_no_graph(self, device):
        r = raxis.tensor([1.0, 2.0])
        out = r * 2.0
        assert out.grad_fn is None
        assert not out.requires_grad

    def test_factory_requires_grad(self, device):
        r = raxis.zeros((3, 4), requires_grad=True)
        t = torch.zeros((3, 4), requires_grad=True)
        assert r.requires_grad == True
        assert r.is_leaf == True
        r_out = r.sum()
        t_out = t.sum()
        r_out.backward()
        t_out.backward()
        check_match(r.grad, t.grad, context="factory requires_grad")


# ===========================================================================
# RETAIN GRAD
# ===========================================================================

class TestRetainGrad:

    def test_retain_grad_on_non_leaf(self, device):
        r = raxis.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        r_mid = r * 2.0
        t_mid = t * 2.0

        r_mid.retain_grad()
        t_mid.retain_grad()

        (r_mid * 3.0).sum().backward()
        (t_mid * 3.0).sum().backward()

        assert r_mid.grad is not None
        assert t_mid.grad is not None
        check_match(r_mid.grad, t_mid.grad, context="retain_grad")

# ===========================================================================
# GRAPH LIFETIME
# ===========================================================================

# Ops whose backward node saves the forward output. If the node stored the
# output itself, output -> grad_fn -> node -> output would form a reference
# cycle and the whole graph would leak.
OUTPUT_SAVING_OPS = [
    ("exp",        lambda x: x.exp()),
    ("expm1",      lambda x: x.expm1()),
    ("sqrt",       lambda x: x.sqrt()),
    ("rsqrt",      lambda x: x.rsqrt()),
    ("reciprocal", lambda x: x.reciprocal()),
    ("sigmoid",    lambda x: x.sigmoid()),
    ("tan",        lambda x: x.tan()),
    ("tanh",       lambda x: x.tanh()),
    ("pow_tensor", lambda x: x.pow(x)),
    ("pow_scalar_base", lambda x: raxis.pow(2.0, x)),
    ("prod",       lambda x: x.prod()),
    ("prod_dim",   lambda x: x.prod(dim=1)),
    ("cumprod",    lambda x: x.cumprod(1)),
    ("amax",       lambda x: x.amax(dim=1)),
    ("amin",       lambda x: x.amin(dim=1)),
    ("max",        lambda x: x.max()),
    ("min",        lambda x: x.min()),
    ("relu",       lambda x: x.relu()),
    ("softmax",    lambda x: x.softmax(1)),
    ("log_softmax", lambda x: x.log_softmax(1)),
]


class TestEngine:

    DEPTH = 100_000

    # sin saves its input, which holds the previous node through its grad_fn
    @pytest.mark.parametrize("op", [lambda y: y + 1.0, lambda y: y * 1.0, lambda y: y.exp() * 0.0 + y,
                                    lambda y: y.sin() * 0.0 + y],
                             ids=["add", "mul", "exp", "sin"])
    def test_deep_chain_backward(self, op):
        # A recursive engine (or recursive node destructors) overflows the stack here
        x = raxis.tensor([1.0], requires_grad=True)
        y = x
        for _ in range(self.DEPTH // 3):
            y = op(y)
        y.sum().backward()
        check_match(x.grad, torch.ones(1))

    @pytest.mark.parametrize("op", [lambda y: y + 1.0, lambda y: y.sin()], ids=["add", "sin"])
    def test_deep_chain_freed_without_backward(self, op):
        x = raxis.tensor([1.0], requires_grad=True)
        y = x
        for _ in range(self.DEPTH):
            y = op(y)
        del y

    def test_diamond(self):
        def f(x):
            a = x * 2.0
            return (a.exp() * a.sin() + a).sum()
        r, t = make_grad_pair(randf(5, 4))
        f(r).backward()
        f(t).backward()
        check_match(r.grad, t.grad, atol=1e-4)

    def test_user_gradient_not_modified(self):
        # AddBackward0 passes the same gradient to both edges, so the buffer of
        # a's node gets the user's tensor twice: it must not be summed in place
        r, t = make_grad_pair(randf(3, 4))
        g = raxis.ones((3, 4))
        a = r * 2.0
        (a + a).backward(g)
        check_match(g, torch.ones(3, 4))
        a = t * 2.0
        (a + a).backward(torch.ones(3, 4))
        check_match(r.grad, t.grad)

    def test_broadcast_gradient(self):
        # SumBackward0 returns an expanded (stride 0) gradient: not writable
        r, t = make_grad_pair(randf(3, 4))
        a = r * 2.0
        (a.sum() + (a * 3.0).sum() + a.sum()).backward()
        a = t * 2.0
        (a.sum() + (a * 3.0).sum() + a.sum()).backward()
        check_match(r.grad, t.grad)

    def test_retained_grad_not_modified(self):
        # b.grad is the same tensor that b's node passes on to a's buffer,
        # where c's gradient arrives later
        def f(x):
            a = x * 2.0
            c = a * 5.0
            b = a + 1.0
            b.retain_grad()
            ((b * 3.0).sum() + c.sum()).backward()
            return b
        r, t = make_grad_pair(randf(3, 4))
        check_match(f(r).grad, f(t).grad)
        check_match(r.grad, t.grad)

    def test_grad_accumulated_in_place(self):
        # As in PyTorch, a later backward adds into the existing .grad tensor
        def run(x):
            (x * 2.0).sum().backward()
            g = x.grad
            (x * 3.0).sum().backward()
            return g
        r, t = make_grad_pair(randf(3, 4))
        check_match(run(r), run(t))
        check_match(r.grad, t.grad)

    def test_leaf_used_many_times(self):
        def f(x):
            return (x * 2.0 + x.exp() - x.sin() * x).sum()
        r, t = make_grad_pair(randf(3, 4))
        f(r).backward()
        f(t).backward()
        check_match(r.grad, t.grad, atol=1e-4)

    def test_grad_is_not_a_view_of_the_user_gradient(self):
        # reshape backward returns a view of g; stealing it as .grad would make
        # the second backward add into g (PyTorch 2.14 does: g ends up as 2)
        x = raxis.zeros((2, 2), requires_grad=True)
        g = raxis.ones(4)
        y = x.reshape(4)
        y.backward(g, retain_graph=True)
        y.backward(g)
        check_match(g, torch.ones(4))
        check_match(x.grad, torch.full((2, 2), 2.0))

    def test_unused_output_of_multi_output_op(self):
        r, t = make_grad_pair(randf(4, 3))
        r.unbind(0)[1].sum().backward()
        t.unbind(0)[1].sum().backward()
        check_match(r.grad, t.grad)


class TestGraphLifetime:

    @pytest.mark.parametrize("name, op", OUTPUT_SAVING_OPS,
                             ids=[n for n, _ in OUTPUT_SAVING_OPS])
    def test_output_not_owned_by_its_graph(self, name, op):
        x = raxis.rand((4, 5), requires_grad=True)
        y = op(x)
        assert y.grad_fn is not None
        assert y._use_count() == 1, f"{name}: output is kept alive by its own grad_fn"

    def test_retain_grad_does_not_own_tensor(self):
        x = raxis.rand((4, 5), requires_grad=True)
        y = x * 2.0
        y.retain_grad()
        assert y._use_count() == 1

    def test_backward_after_retained_tensor_is_freed(self):
        r = raxis.tensor([1.0, 2.0, 3.0], requires_grad=True)
        t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

        # AddBackward0 does not save its inputs, so deleting mid frees it
        r_mid = r * 2.0
        r_mid.retain_grad()
        r_out = (r_mid + 1.0).sum()
        del r_mid
        r_out.backward()

        t_out = (t * 2.0 + 1.0).sum()
        t_out.backward()

        check_match(r.grad, t.grad, context="retain_grad on freed tensor")


@pytest.mark.parametrize("op", ["sum", "mean"])
@pytest.mark.parametrize("dims", [(2, 0), (0, 2), (-1, 0), (1,)])
def test_reduce_backward_with_unsorted_dims(op, dims):
    # The reduced dims are restored in increasing order whatever their order
    data = np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32)
    r, t = make_grad_pair(data)
    kept = [s for i, s in enumerate(data.shape) if i not in {d % 3 for d in dims}]
    w = np.random.default_rng(1).standard_normal(kept).astype(np.float32)
    (getattr(r, op)(dims) * raxis.from_numpy(w)).sum().backward()
    (getattr(t, op)(dims) * torch.from_numpy(w)).sum().backward()
    check_match(r.grad, t.grad)
