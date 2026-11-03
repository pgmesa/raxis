import warnings

import numpy as np
import pytest
import raxis
import torch

from test_utils import check_match, device, make_pair, randf, randf_positive  # noqa: F401 (fixture)


UNARY = ["neg", "abs", "sign", "ceil", "floor", "round", "exp", "expm1", "log", "log2",
         "log10", "log1p", "sqrt", "rsqrt", "reciprocal", "sigmoid", "sin", "cos", "tan",
         "asin", "acos", "atan", "sinh", "cosh", "tanh", "erf", "erfc", "logical_not"]
BINARY = ["add", "sub", "mul", "div", "maximum", "minimum", "pow",
          "gt", "ge", "lt", "le", "eq", "ne", "logical_and", "logical_or"]
WITH_SCALAR = ["add", "sub", "mul", "div", "pow", "gt", "ge", "lt", "le", "eq", "ne"]
COMPARISONS = ["gt", "ge", "lt", "le", "eq", "ne", "logical_and", "logical_or"]


def empty_pair(shape, dtype, device):
    return (raxis.empty(shape, dtype=getattr(raxis, dtype), device=device),
            torch.empty(shape, dtype=getattr(torch, dtype), device=device))


def out_dtype(name):
    return "bool" if name in COMPARISONS or name == "logical_not" else "float32"


@pytest.mark.parametrize("name", UNARY)
def test_unary(name, device):
    # in [0.1, 0.9]: inside the domain of every op here (asin, log, ...)
    r, t = make_pair(randf(3, 4, lo=0.1, hi=0.9), device)
    r_out, t_out = empty_pair((3, 4), out_dtype(name), device)
    result = getattr(raxis, name)(r, out=r_out)
    getattr(torch, name)(t, out=t_out)
    assert result is r_out
    check_match(r_out, t_out, context=name)


@pytest.mark.parametrize("name", BINARY)
def test_binary(name, device):
    r_a, t_a = make_pair(randf_positive(3, 4), device)
    r_b, t_b = make_pair(randf_positive(1, 4, seed=7), device)
    r_out, t_out = empty_pair((3, 4), out_dtype(name), device)
    result = getattr(raxis, name)(r_a, r_b, out=r_out)
    getattr(torch, name)(t_a, t_b, out=t_out)
    assert result is r_out
    check_match(r_out, t_out, context=name)


@pytest.mark.parametrize("name", WITH_SCALAR)
def test_with_scalar(name, device):
    r, t = make_pair(randf_positive(3, 4), device)
    r_out, t_out = empty_pair((3, 4), out_dtype(name), device)
    getattr(raxis, name)(r, 0.5, out=r_out)
    getattr(torch, name)(t, 0.5, out=t_out)
    check_match(r_out, t_out, context=name)


def test_scalar_base_pow(device):
    r, t = make_pair(randf(5), device)
    r_out, t_out = empty_pair((5,), "float32", device)
    raxis.pow(2, r, out=r_out)
    torch.pow(2, t, out=t_out)
    check_match(r_out, t_out)


@pytest.mark.parametrize("name", ["eq", "logical_and"])
def test_comparison_into_float(name, device):
    r, t = make_pair(np.array([0.0, 1.0, 2.0], dtype=np.float32), device)
    r_out, t_out = empty_pair((3,), "float32", device)
    getattr(raxis, name)(r, r, out=r_out)
    getattr(torch, name)(t, t, out=t_out)
    check_match(r_out, t_out)


class TestResize:

    def test_empty_out_takes_the_inputs_layout(self, device):
        r_a, t_a = make_pair(randf(3, 4), device)
        r_out, t_out = empty_pair((0,), "float32", device)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            raxis.add(r_a.t(), r_a.t(), out=r_out)
            torch.add(t_a.t(), t_a.t(), out=t_out)
        assert r_out.shape == t_out.shape
        assert r_out.stride() == t_out.stride()
        check_match(r_out, t_out)

    def test_wrong_shape_warns(self, device):
        r_a, t_a = make_pair(randf(3, 4), device)
        r_out, t_out = empty_pair((5,), "float32", device)
        with pytest.warns(UserWarning, match="was resized since it had shape") as raxis_warnings:
            raxis.exp(r_a, out=r_out)
        with pytest.warns(UserWarning) as torch_warnings:
            torch.exp(t_a, out=t_out)
        # torch appends where in its source it was raised
        assert str(torch_warnings[0].message).startswith(str(raxis_warnings[0].message))
        check_match(r_out, t_out)

    def test_right_shape_keeps_its_strides(self, device):
        r_a, t_a = make_pair(randf(3, 4), device)
        r_out = raxis.empty(4, 3, device=device).t()
        t_out = torch.empty(4, 3, device=device).t()
        raxis.mul(r_a, 2, out=r_out)
        torch.mul(t_a, 2, out=t_out)
        assert r_out.stride() == t_out.stride()
        check_match(r_out, t_out)

    def test_resized_to_another_dtype(self, device):
        r_a, t_a = make_pair(randf(3, 4), device)
        r_out, t_out = empty_pair((0,), "float64", device)
        raxis.add(r_a, r_a, out=r_out)
        torch.add(t_a, t_a, out=t_out)
        assert r_out.dtype == raxis.float64 and r_out.shape == (3, 4)
        check_match(r_out, t_out)


class TestDtype:

    def test_wider_out(self, device):
        r, t = make_pair(np.arange(4, dtype=np.int64), device)
        r_out, t_out = empty_pair((4,), "float32", device)
        raxis.exp(r, out=r_out)
        torch.exp(t, out=t_out)
        check_match(r_out, t_out)

    def test_unsafe_cast_raises(self, device):
        r, t = make_pair(randf(4), device)
        r_out, t_out = empty_pair((4,), "int64", device)
        with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
            raxis.add(r, r, out=r_out)
        with pytest.raises(RuntimeError, match="can't be cast to the desired output type"):
            torch.add(t, t, out=t_out)

    def test_integer_rounding_copies(self, device):
        r, t = make_pair(np.arange(4, dtype=np.int64), device)
        r_out, t_out = empty_pair((4,), "int64", device)
        raxis.floor(r, out=r_out)
        torch.floor(t, out=t_out)
        check_match(r_out, t_out)


class TestAutograd:

    MESSAGE = "functions with out=... arguments don't support automatic differentiation"

    def test_input_requiring_grad_raises(self):
        x = raxis.randn(3, requires_grad=True)
        with pytest.raises(RuntimeError, match=r"exp\(\): " + self.MESSAGE):
            raxis.exp(x, out=raxis.empty(3))

    def test_out_requiring_grad_raises(self):
        with pytest.raises(RuntimeError, match=self.MESSAGE):
            raxis.add(raxis.ones(3), 1, out=raxis.empty(3, requires_grad=True))

    def test_allowed_without_grad_mode(self):
        x = raxis.randn(3, requires_grad=True)
        out = raxis.empty(3)
        with raxis.no_grad():
            raxis.exp(x, out=out)
        assert not out.requires_grad

    def test_comparisons_allowed(self):
        # Not differentiable, as in torch
        x = raxis.randn(3, requires_grad=True)
        out = raxis.empty(3, dtype=raxis.bool)
        raxis.gt(x, 0, out=out)

    @pytest.mark.parametrize("call", [
        lambda out: raxis.mul(raxis.ones(3), 2, out=out),
        lambda out: raxis.lt(raxis.ones(3), 2, out=out),
        lambda out: raxis.floor(raxis.ones(3), out=out),
    ])
    def test_version_moves_once(self, call):
        out = raxis.empty(3)
        call(out)
        assert out._version == 1


def test_overlapping_out_raises():
    a = raxis.randn(3, 4)
    with pytest.raises(RuntimeError, match="more than one element of the written-to tensor"):
        raxis.add(a, a, out=a[:, :1].expand(3, 4))


def test_inplace_through_out_still_works():
    a = raxis.ones(3, 4)
    raxis.add(a, 1, out=a)
    assert (a == 2).all().item()
