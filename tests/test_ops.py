
import json
import os
import subprocess
import sys

import pytest
import raxis
import torch
import numpy as np

from test_utils import (
    check_match, make_pair, randf, randf_positive, randf_nonzero,
    randf_centered, randf_small, randf_unit, device
)


# ===========================================================================
# Op tables
# ===========================================================================

UNARY_SHAPES = [(5,), (10, 20), (3, 10, 20), (500, 200), (100, 100, 10),
                (256, 128), (512, 256), (100, 100, 100)]

BINARY_SHAPES = [(1,), (100,), (10, 20), (5, 10, 20), (2, 3, 4, 5), (500, 200),
                 (256, 128), (512, 256)]

BROADCAST_CASES = [
    ((10, 20), (1, 20),       "row broadcast"),
    ((10, 20), (10, 1),       "col broadcast"),
    ((10, 20), (20,),         "1d broadcast"),
    ((5, 1, 20), (1, 10, 20), "mutual broadcast"),
    ((3, 4, 5), (5,),         "trailing dim"),
    ((3, 4, 5), (1, 1, 5),    "leading dims"),
    ((3, 4, 5), (4, 1),       "inner dim"),
    ((1,), (10, 20),          "scalar-like"),
    ((10, 1, 5), (1, 4, 1),   "3d mutual"),
    ((256, 128), (1, 128),    "large row broadcast"),
    ((256, 128), (256, 1),    "large col broadcast"),
    ((64, 64, 64), (64,),     "large trailing dim"),
    ((0, 20), (1, 20),        "empty row broadcast"),
    ((10, 0), (10, 1),        "empty col broadcast"),
]

REDUCE_ALL_SHAPES = [(10,), (10, 20), (5, 10, 20),
                     (256, 128), (512, 256)]

REDUCE_DIM_CASES = [
    ((10, 20),    0,      False),
    ((10, 20),    1,      False),
    ((10, 20),    -1,     False),
    ((10, 20),    0,      True),
    ((5, 10, 20), 2,      False),
    ((5, 10, 20), (0, 2), False),
    ((5, 10, 20), -1,     True),
    ((200, 300),   0,     False),
    ((200, 300),   1,     False),
    ((256, 128),   0,     False),
    ((256, 128),   1,     False),
    ((256, 128),   0,     True),
    ((64, 64, 64), 2,     False),
    ((64, 64, 64), (0, 2), False),
]

CMP_SHAPES = [(10, 20), (3, 10, 20), (256, 128)]

ELEMENTWISE_BINARY_SHAPES = [(10, 20), (5, 10, 20), (256, 128)]

UNARY_OPS = [
    ("neg",     "neg",     randf_centered),
    ("abs",     "abs",     randf_centered),
    ("ceil",    "ceil",    randf_centered),
    ("floor",   "floor",   randf_centered),
    ("round",   "round",   randf_centered),
    ("exp",     "exp",     randf_small),
    ("expm1",   "expm1",   randf_small),
    ("log",     "log",     randf_positive),
    ("log2",    "log2",    randf_positive),
    ("log10",   "log10",   randf_positive),
    ("log1p",   "log1p",   randf_positive),
    ("sqrt",    "sqrt",    randf_positive),
    ("rsqrt",      "rsqrt",      randf_positive),
    ("reciprocal", "reciprocal", randf_nonzero),
    ("sigmoid", "sigmoid", randf_centered),
    ("sin",     "sin",     randf_centered),
    ("cos",     "cos",     randf_centered),
    ("tan",     "tan",     randf_small),
    ("asin",    "asin",    randf_unit),
    ("acos",    "acos",    randf_unit),
    ("atan",    "atan",    randf_centered),
    ("sinh",    "sinh",    randf_small),
    ("cosh",    "cosh",    randf_small),
    ("tanh",    "tanh",    randf_centered),
    ("erf",     "erf",     randf_centered),
    ("erfc",    "erfc",    randf_centered),
]

UNARY_INPLACE_OPS = [
    ("neg_",     randf_centered),
    ("abs_",     randf_centered),
    ("ceil_",    randf_centered),
    ("floor_",   randf_centered),
    ("round_",   randf_centered),
    ("exp_",     randf_small),
    ("expm1_",   randf_small),
    ("log_",     randf_positive),
    ("log2_",    randf_positive),
    ("log10_",   randf_positive),
    ("log1p_",   randf_positive),
    ("sqrt_",    randf_positive),
    ("rsqrt_",      randf_positive),
    ("reciprocal_", randf_nonzero),
    ("sigmoid_", randf_centered),
    ("sin_",     randf_centered),
    ("cos_",     randf_centered),
    ("tan_",     randf_small),
    ("asin_",    randf_unit),
    ("acos_",    randf_unit),
    ("atan_",    randf_centered),
    ("sinh_",    randf_small),
    ("cosh_",    randf_small),
    ("tanh_",    randf_centered),
    ("erf_",     randf_centered),
    ("erfc_",    randf_centered),
]

BINARY_ARITHMETIC_OPS = [
    ("add", "+", randf),
    ("sub", "-", randf),
    ("mul", "*", randf),
    ("div", "/", randf_nonzero),
]

CMP_OPS = ["gt", "ge", "lt", "le", "eq", "ne"]
CMP_SCALARS = [0.0, 0.5, -1.0, 100.0]

REDUCE_ALL_OPS = [
    ("sum",     "sum",     randf),
    ("prod",    "prod",    randf_unit),
    ("max",     "max",     randf),
    ("min",     "min",     randf),
    ("mean",    "mean",    randf),
    ("nansum",  "nansum",  randf),
    ("nanmean", "nanmean", randf),
]

REDUCE_DIM_OPS = [
    ("sum",     "sum",     randf),
    ("mean",    "mean",    randf),
    ("prod",    "prod",    randf_unit),
    ("nansum",  "nansum",  randf),
    ("nanmean", "nanmean", randf),
]

REDUCE_MINMAX_DIM_OPS = [
    ("amax", "amax", randf),
    ("amin", "amin", randf),
]

SCALAR_VALUES = [0, 1, -1, 2, 42, 0.5, -3.14, 1e-7, 1e7]

MM_SHAPES = [
    ((1, 1), (1, 1)),
    ((1, 5), (5, 1)),
    ((3, 4), (4, 5)),
    ((16, 32), (32, 16)),
    ((64, 128), (128, 64)),
    ((256, 128), (128, 256)),
]

MM_DTYPES = [np.float32, np.float64, np.int32, np.int64, np.uint8]

MATMUL_CASES = [
    ("1Dx1D_dot",      (16,),         (16,)),
    ("2Dx1D_mv",       (8, 16),       (16,)),
    ("1Dx2D",          (16,),         (16, 8)),
    ("2Dx2D_mm",       (8, 16),       (16, 4)),
    ("3Dx3D_bmm",      (4, 8, 16),    (4, 16, 6)),
    ("3Dx3D_bcast",    (1, 8, 16),    (4, 16, 6)),
    ("NDx2D_fold",     (2, 3, 8),     (8, 4)),
    ("NDx1D",          (2, 3, 8),     (8,)),
    ("2DxND",          (8, 16),       (3, 16, 4)),
    ("4Dx4D",          (2, 3, 4, 8),  (2, 3, 8, 5)),
    ("4Dx4D broadcast",(2, 1, 4, 8),  (1, 3, 8, 5)),
]


# ===========================================================================
# UNARY OPS
# ===========================================================================

class TestUnaryOps:

    @pytest.mark.parametrize("shape", UNARY_SHAPES)
    @pytest.mark.parametrize(
        "name,method,data_fn", UNARY_OPS,
        ids=[u[0] for u in UNARY_OPS]
    )
    def test_forward(self, shape, name, method, data_fn, device):
        data = data_fn(*shape)
        r, t = make_pair(data, device)
        check_match(getattr(r, method)(), getattr(t, method)(), context=f"{name} {shape}")

    def test_isnan(self, device):
        data = np.array([1.0, float("nan"), 0.0, float("nan"), -1.0], dtype=np.float32)
        r = raxis.from_numpy(data)
        t = torch.from_numpy(data.copy())
        check_match(r.isnan(), t.isnan(), atol=0, context="isnan")

    def test_logical_not(self, device):
        r = raxis.tensor([True, False, True, False])
        t = torch.tensor([True, False, True, False])
        check_match(r.logical_not(), t.logical_not(), atol=0, context="logical_not")

    def test_neg_operator(self, device):
        data = randf_centered(10, 20)
        r, t = make_pair(data, device)
        check_match(-r, -t, context="neg operator")

    def test_abs_operator(self, device):
        data = randf_centered(10, 20)
        r, t = make_pair(data, device)
        check_match(abs(r), abs(t), context="abs operator")


# ===========================================================================
# UNARY IN-PLACE OPS
# ===========================================================================

class TestUnaryInplaceOps:

    @pytest.mark.parametrize(
        "name,data_fn", UNARY_INPLACE_OPS,
        ids=[u[0] for u in UNARY_INPLACE_OPS]
    )
    def test_inplace(self, name, data_fn, device):
        data = data_fn(10, 20)
        r = raxis.from_numpy(data)
        t = torch.from_numpy(data.copy())

        prev_id = id(r)
        getattr(r, name)()
        getattr(t, name)()

        assert id(r) == prev_id, f"{name} created a new object"
        check_match(r, t, context=f"{name} inplace")


# ===========================================================================
# COMPARISON OPS
# ===========================================================================

class TestComparisonOps:

    @pytest.mark.parametrize("op", CMP_OPS)
    @pytest.mark.parametrize("shape", CMP_SHAPES)
    def test_tensor_tensor(self, op, shape, device):
        a, b = randf(*shape, seed=1), randf(*shape, seed=2)
        r_a, t_a = make_pair(a, device)
        r_b, t_b = make_pair(b, device)
        check_match(
            getattr(raxis, op)(r_a, r_b),
            getattr(torch, op)(t_a, t_b),
            atol=0, context=f"{op} tensor-tensor {shape}")

    @pytest.mark.parametrize("op", CMP_OPS)
    @pytest.mark.parametrize("scalar", CMP_SCALARS)
    def test_tensor_scalar(self, op, scalar, device):
        data = randf_centered(10, 20)
        r, t = make_pair(data, device)
        check_match(
            getattr(raxis, op)(r, scalar),
            getattr(torch, op)(t, scalar),
            atol=0, context=f"{op} vs {scalar}")

    @pytest.mark.parametrize("op", CMP_OPS)
    def test_method_form(self, op, device):
        a, b = randf(10, 20, seed=1), randf(10, 20, seed=2)
        r_a, t_a = make_pair(a, device)
        r_b, t_b = make_pair(b, device)
        check_match(
            getattr(r_a, op)(r_b),
            getattr(t_a, op)(t_b),
            atol=0, context=f".{op}() method")

    @pytest.mark.parametrize("op", CMP_OPS)
    @pytest.mark.parametrize("lhs_shape,rhs_shape,desc", BROADCAST_CASES,
                             ids=[c[2] for c in BROADCAST_CASES])
    def test_broadcast(self, op, lhs_shape, rhs_shape, desc, device):
        a, b = randf(*lhs_shape, seed=1), randf(*rhs_shape, seed=2)
        r_a, t_a = make_pair(a, device)
        r_b, t_b = make_pair(b, device)
        check_match(
            getattr(raxis, op)(r_a, r_b),
            getattr(torch, op)(t_a, t_b),
            atol=0, context=f"{op} {desc}")


# ===========================================================================
# BINARY ELEMENTWISE (maximum, minimum, pow, logical_and, logical_or)
# ===========================================================================

class TestBinaryElementwise:

    @pytest.mark.parametrize("op", ["maximum", "minimum"])
    @pytest.mark.parametrize("shape", ELEMENTWISE_BINARY_SHAPES)
    def test_forward(self, op, shape, device):
        a, b = randf(*shape, seed=1), randf(*shape, seed=2)
        r_a, t_a = make_pair(a, device)
        r_b, t_b = make_pair(b, device)
        check_match(
            getattr(r_a, op)(r_b),
            getattr(torch, op)(t_a, t_b),
            context=f"{op} {shape}")

    @pytest.mark.parametrize("op", ["maximum", "minimum"])
    def test_module_level(self, op, device):
        a, b = randf(10, 20, seed=1), randf(10, 20, seed=2)
        r_a, t_a = make_pair(a, device)
        r_b, t_b = make_pair(b, device)
        check_match(
            getattr(raxis, op)(r_a, r_b),
            getattr(torch, op)(t_a, t_b),
            context=f"raxis.{op}")

    @pytest.mark.parametrize("op", ["maximum", "minimum"])
    @pytest.mark.parametrize("lhs_shape,rhs_shape,desc", BROADCAST_CASES,
                             ids=[c[2] for c in BROADCAST_CASES])
    def test_broadcast(self, op, lhs_shape, rhs_shape, desc, device):
        a, b = randf(*lhs_shape, seed=1), randf(*rhs_shape, seed=2)
        r_a, t_a = make_pair(a, device)
        r_b, t_b = make_pair(b, device)
        check_match(
            getattr(raxis, op)(r_a, r_b),
            getattr(torch, op)(t_a, t_b),
            context=f"{op} {desc}")

    @pytest.mark.parametrize("exp", [0.0, 1.0, 0.5, -0.5, -1.0, 2.0, 3.0, -2.0, 2.5])
    def test_pow_scalar_fastpaths(self, exp, device):
        data = randf_positive(10, 20)
        r, t = make_pair(data, device)
        check_match(r.pow(exp), t.pow(exp), context=f"pow({exp})")

    @pytest.mark.parametrize("exp", [0.0, 1.0, 0.5, -0.5, -1.0, 2.0, 3.0, -2.0, 2.5])
    @pytest.mark.parametrize("shape", ELEMENTWISE_BINARY_SHAPES)
    def test_pow_scalar_shapes(self, exp, shape, device):
        data = randf_positive(*shape)
        r, t = make_pair(data, device)
        check_match(r.pow(exp), t.pow(exp), context=f"pow({exp}) {shape}")

    @pytest.mark.parametrize("base", [1.0, 2.0, 0.5, 3.0])
    def test_pow_scalar_base(self, base, device):
        exp_data = randf(10, 20, lo=0.5, hi=3.0)
        r_e, t_e = make_pair(exp_data, device)
        check_match(
            raxis.pow(base, r_e),
            torch.pow(base, t_e),
            context=f"pow({base}, tensor)")

    def test_pow_int_tensor(self, device):
        base = np.array([[2, 3, 4], [5, 1, -1]], dtype=np.int64)
        exp = np.array([[3, 2, 1], [0, 10, 3]], dtype=np.int64)
        r_b, t_b = make_pair(base, device)
        r_e, t_e = make_pair(exp, device)
        check_match(raxis.pow(r_b, r_e), torch.pow(t_b, t_e),
                    atol=0, context="pow int tensor-tensor")

    def test_pow_int_scalar(self, device):
        base = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        r, t = make_pair(base, device)
        check_match(r.pow(3), t.pow(3), atol=0, context="pow int scalar")

    def test_pow_int_negative_exp(self, device):
        base = np.array([1, -1, 2, -2, 3], dtype=np.int64)
        exp = np.array([-1, -1, -1, -2, -3], dtype=np.int64)
        r_b, t_b = make_pair(base, device)
        r_e, t_e = make_pair(exp, device)
        check_match(raxis.pow(r_b, r_e), torch.pow(t_b, t_e),
                    atol=0, context="pow int negative exp")

    def test_pow_zero_exp(self, device):
        data = randf_positive(10, 20)
        r, t = make_pair(data, device)
        check_match(r.pow(0.0), t.pow(0.0), context="pow(0) == 1")

    def test_pow_one_exp(self, device):
        data = randf_positive(10, 20)
        r, t = make_pair(data, device)
        check_match(r.pow(1.0), t.pow(1.0), context="pow(1) == identity")

    def test_pow_scalar_base_one(self, device):
        exp_data = randf(10, 20, lo=0.5, hi=3.0)
        r_e, t_e = make_pair(exp_data, device)
        check_match(
            raxis.pow(1.0, r_e),
            torch.pow(1.0, t_e),
            context="pow(1.0, tensor) == 1")

    def test_logical_and(self, device):
        a = np.array([True, False, True, False])
        b = np.array([True, True, False, False])
        r_out = raxis.logical_and(raxis.tensor(a.tolist()), raxis.tensor(b.tolist()))
        t_out = torch.logical_and(torch.tensor(a), torch.tensor(b))
        check_match(r_out, t_out, atol=0, context="logical_and")

    def test_logical_or(self, device):
        a = np.array([True, False, True, False])
        b = np.array([True, True, False, False])
        r_out = raxis.logical_or(raxis.tensor(a.tolist()), raxis.tensor(b.tolist()))
        t_out = torch.logical_or(torch.tensor(a), torch.tensor(b))
        check_match(r_out, t_out, atol=0, context="logical_or")

    def test_logical_and_method(self, device):
        r_a = raxis.tensor([True, False, True])
        r_b = raxis.tensor([True, True, False])
        t_a = torch.tensor([True, False, True])
        t_b = torch.tensor([True, True, False])
        check_match(r_a.logical_and(r_b), t_a.logical_and(t_b), atol=0)

    def test_logical_or_method(self, device):
        r_a = raxis.tensor([True, False, False])
        r_b = raxis.tensor([False, True, False])
        t_a = torch.tensor([True, False, False])
        t_b = torch.tensor([False, True, False])
        check_match(r_a.logical_or(r_b), t_a.logical_or(t_b), atol=0)


# ===========================================================================
# BINARY ARITHMETIC OPS
# ===========================================================================

class TestBinaryArithmetic:

    @pytest.mark.parametrize("shape", BINARY_SHAPES, ids=[str(s) for s in BINARY_SHAPES])
    @pytest.mark.parametrize(
        "name,op_str,data_fn_rhs", BINARY_ARITHMETIC_OPS,
        ids=[o[0] for o in BINARY_ARITHMETIC_OPS]
    )
    def test_forward(self, shape, name, op_str, data_fn_rhs, device):
        a_data = randf(*shape, seed=42)
        b_data = data_fn_rhs(*shape, seed=99)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        r_out = eval(f"r_a {op_str} r_b")
        t_out = eval(f"t_a {op_str} t_b")
        check_match(r_out, t_out, context=f"{name} {shape}")


# ===========================================================================
# BROADCASTING
# ===========================================================================

class TestBroadcasting:

    @pytest.mark.parametrize("lhs_shape,rhs_shape,desc", BROADCAST_CASES,
                             ids=[c[2] for c in BROADCAST_CASES])
    @pytest.mark.parametrize(
        "name,op_str,data_fn_rhs", BINARY_ARITHMETIC_OPS,
        ids=[o[0] for o in BINARY_ARITHMETIC_OPS]
    )
    def test_broadcast(self, lhs_shape, rhs_shape, desc, name, op_str, data_fn_rhs, device):
        a_data = randf(*lhs_shape, seed=123)
        b_data = data_fn_rhs(*rhs_shape, seed=456)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        r_out = eval(f"r_a {op_str} r_b")
        t_out = eval(f"t_a {op_str} t_b")
        check_match(r_out, t_out, context=f"{name} {desc}")


# ===========================================================================
# SCALAR OPERATIONS
# ===========================================================================

class TestScalarOps:

    @pytest.mark.parametrize("scalar", SCALAR_VALUES)
    def test_tensor_add_scalar(self, scalar, device):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(r + scalar, t + scalar, context=f"tensor + {scalar}")

    @pytest.mark.parametrize("scalar", SCALAR_VALUES)
    def test_scalar_add_tensor(self, scalar, device):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(scalar + r, scalar + t, context=f"{scalar} + tensor")

    @pytest.mark.parametrize("scalar", [s for s in SCALAR_VALUES if s != 0])
    def test_tensor_div_scalar(self, scalar, device):
        data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(r / scalar, t / scalar, context=f"tensor / {scalar}")

    @pytest.mark.parametrize("scalar", SCALAR_VALUES)
    def test_scalar_sub_tensor(self, scalar, device):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(scalar - r, scalar - t, context=f"{scalar} - tensor")

    @pytest.mark.parametrize("scalar", [s for s in SCALAR_VALUES if s != 0])
    def test_scalar_div_tensor(self, scalar, device):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(scalar / r, scalar / t, context=f"{scalar} / tensor")

    @pytest.mark.parametrize("scalar", SCALAR_VALUES)
    def test_tensor_mul_scalar(self, scalar, device):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(r * scalar, t * scalar, context=f"tensor * {scalar}")

    @pytest.mark.parametrize("scalar", SCALAR_VALUES)
    def test_scalar_mul_tensor(self, scalar, device):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(scalar * r, scalar * t, context=f"{scalar} * tensor")


# ===========================================================================
# METHOD vs OPERATOR
# ===========================================================================

class TestMethodVsOperator:

    def _data(self):
        return randf(10, 20, seed=99), randf_nonzero(10, 20, seed=77)

    @pytest.mark.parametrize("op,method", [
        ("+", "add"), ("-", "sub"), ("*", "mul"), ("/", "div"),
    ])
    def test_operator_vs_method(self, op, method, device):
        a, b = self._data()
        r_a, _ = make_pair(a, device)
        r_b, _ = make_pair(b, device)
        op_result = eval(f"r_a {op} r_b")
        method_result = getattr(r_a, method)(r_b)
        check_match(op_result, method_result, context=f"{op} vs .{method}()")

    @pytest.mark.parametrize("method", ["add", "sub", "mul", "div"])
    @pytest.mark.parametrize("scalar", [5, 2.0])
    def test_scalar_method(self, method, scalar, device):
        data = randf(10, seed=1)
        r, t = make_pair(data, device)
        check_match(
            getattr(r, method)(scalar),
            getattr(t, method)(torch.tensor(scalar)),
            context=f".{method}({scalar})")


# ===========================================================================
# MODULE-LEVEL FUNCTIONS
# ===========================================================================

class TestModuleLevelOps:

    @pytest.mark.parametrize("op,data_fn_b", [
        ("add", randf),
        ("sub", randf),
        ("mul", randf),
        ("div", randf_nonzero),
    ])
    def test_binary_module(self, op, data_fn_b, device):
        a = randf(10, 20, seed=1)
        b = data_fn_b(10, 20, seed=2)
        r_a, t_a = make_pair(a, device)
        r_b, t_b = make_pair(b, device)
        check_match(
            getattr(raxis, op)(r_a, r_b),
            getattr(torch, op)(t_a, t_b),
            context=f"raxis.{op}()")

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    @pytest.mark.parametrize("scalar", [10, 2.5])
    def test_binary_module_scalar(self, op, scalar, device):
        data = randf(10, seed=1)
        r, t = make_pair(data, device)
        check_match(
            getattr(raxis, op)(r, scalar),
            getattr(torch, op)(t, scalar),
            context=f"raxis.{op}(scalar={scalar})")

    @pytest.mark.parametrize("op", [
        "neg", "abs", "exp", "log", "sqrt", "sigmoid",
        "ceil", "floor", "round", "isnan", "logical_not",
    ])
    def test_unary_module_matches_method(self, op, device):
        if op == "isnan":
            data = np.array([1.0, float("nan"), 0.0], dtype=np.float32)
            r = raxis.from_numpy(data)
        elif op == "logical_not":
            r = raxis.tensor([True, False, True])
        elif op in ("log", "sqrt"):
            r = raxis.from_numpy(randf_positive(10, 20))
        elif op == "exp":
            r = raxis.from_numpy(randf_small(10, 20))
        else:
            r = raxis.from_numpy(randf_centered(10, 20))

        check_match(getattr(r, op)(), getattr(raxis, op)(r), context=f"raxis.{op} vs .{op}()")


# ===========================================================================
# IN-PLACE ARITHMETIC
# ===========================================================================

class TestInPlaceArithmetic:

    def test_iadd_tensor(self, device):
        r, t = make_pair(randf(10, seed=1), device)
        r_b, t_b = make_pair(randf(10, seed=2), device)
        prev_id = id(r)
        r += r_b; t += t_b
        assert id(r) == prev_id
        check_match(r, t, context="iadd tensor")

    def test_iadd_scalar(self, device):
        r, t = make_pair(randf(10, seed=1), device)
        prev_id = id(r)
        r += 5; t += 5
        assert id(r) == prev_id
        check_match(r, t, context="iadd scalar")

    def test_isub_tensor(self, device):
        r, t = make_pair(randf(10, seed=1), device)
        r_b, t_b = make_pair(randf(10, seed=2), device)
        r -= r_b; t -= t_b
        check_match(r, t, context="isub tensor")

    def test_imul_tensor(self, device):
        r, t = make_pair(randf(10, seed=1), device)
        r_b, t_b = make_pair(randf(10, seed=2), device)
        r *= r_b; t *= t_b
        check_match(r, t, context="imul tensor")

    def test_idiv_scalar(self, device):
        r, t = make_pair(randf(10, seed=1, lo=1.0, hi=100.0), device)
        r /= 5; t /= 5
        check_match(r, t, context="idiv scalar")

    def test_chained_inplace(self, device):
        r, t = make_pair(randf(10, seed=1), device)
        prev_id = id(r)
        r += 10; t += 10
        r *= 2;  t *= 2
        r -= 5;  t -= 5
        r /= 3;  t /= 3
        assert id(r) == prev_id
        check_match(r, t, context="chained inplace")

    def test_fill_and_zero(self, device):
        r = raxis.ones((3, 4))
        t = torch.ones((3, 4))
        r.fill_(42); t.fill_(42)
        check_match(r, t, context="fill_(42)")
        r.zero_(); t.zero_()
        check_match(r, t, context="zero_()")


# ===========================================================================
# CLONE and COPY
# ===========================================================================

class TestCloneCopy:

    def test_clone_values(self, device):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(r.clone(), t.clone(), context="clone")

    def test_clone_is_independent(self, device):
        r = raxis.tensor([1.0, 2.0, 3.0])
        r_clone = r.clone()
        r += 100
        check_match(r_clone, torch.tensor([1.0, 2.0, 3.0]), context="clone independence")

    def test_copy_same_dtype(self, device):
        r_src = raxis.tensor([10.0, 20.0, 30.0])
        r_dst = raxis.tensor([0.0, 0.0, 0.0])
        r_dst.copy_(r_src)
        check_match(r_dst, torch.tensor([10.0, 20.0, 30.0]), context="copy_ same dtype")

    def test_copy_cross_dtype(self, device):
        r_src = raxis.tensor([1, 2, 3], dtype=raxis.int32)
        r_dst = raxis.zeros((3,), dtype=raxis.float32)
        r_dst.copy_(r_src)
        t_src = torch.tensor([1, 2, 3], dtype=torch.int32)
        t_dst = torch.zeros(3, dtype=torch.float32)
        t_dst.copy_(t_src)
        check_match(r_dst, t_dst, context="copy_ cross dtype")
        assert r_dst.dtype == raxis.float32


# ===========================================================================
# REDUCTION OPS -- ALL-REDUCE
# ===========================================================================

class TestReductionAll:

    @pytest.mark.parametrize(
        "name,method,data_fn", REDUCE_ALL_OPS,
        ids=[o[0] for o in REDUCE_ALL_OPS]
    )
    @pytest.mark.parametrize("shape", REDUCE_ALL_SHAPES)
    def test_forward(self, name, method, data_fn, shape, device):
        data = data_fn(*shape)
        r, t = make_pair(data, device)
        check_match(getattr(r, method)(), getattr(t, method)(), context=f"{name}() {shape}")

    @pytest.mark.parametrize("op", ["sum", "max", "min"])
    def test_module_level(self, op, device):
        data = randf(10, 20)
        r, t = make_pair(data, device)
        check_match(getattr(raxis, op)(r), getattr(torch, op)(t), context=f"raxis.{op}")


# ===========================================================================
# REDUCTION OPS -- DIM-AWARE
# ===========================================================================

class TestReductionDim:

    @pytest.mark.parametrize(
        "name,method,data_fn", REDUCE_DIM_OPS,
        ids=[o[0] for o in REDUCE_DIM_OPS]
    )
    @pytest.mark.parametrize("shape,dim,keepdim", REDUCE_DIM_CASES,
                             ids=[f"{s}_d{d}_k{k}" for s, d, k in REDUCE_DIM_CASES])
    def test_forward(self, name, method, data_fn, shape, dim, keepdim, device):
        data = data_fn(*shape)
        r, t = make_pair(data, device)
        check_match(
            self.reduce_call(r, method, dim=dim, keepdim=keepdim),
            self.reduce_call(t, method, dim=dim, keepdim=keepdim),
            context=f"{name} dim={dim} keepdim={keepdim}")

    def reduce_call(self, x, method, dim, keepdim):
        fn = getattr(x, method)

        if method == "prod" and isinstance(dim, tuple):
            for d in sorted(dim, reverse=True):
                x = fn(dim=d, keepdim=keepdim)
                fn = getattr(x, method)
            return x
        else:
            return fn(dim=dim, keepdim=keepdim)

    @pytest.mark.parametrize("op", ["nansum", "nanmean"])
    @pytest.mark.parametrize("shape,dim,keepdim", REDUCE_DIM_CASES,
                             ids=[f"{s}_d{d}_k{k}" for s, d, k in REDUCE_DIM_CASES])
    def test_nan_reduce_dim(self, op, shape, dim, keepdim, device):
        data = randf(*shape)
        rng = np.random.RandomState(42)
        data[rng.rand(*shape) < 0.1] = float("nan")
        r, t = make_pair(data, device)
        check_match(
            getattr(r, op)(dim=dim, keepdim=keepdim),
            getattr(t, op)(dim=dim, keepdim=keepdim),
            context=f"{op} dim={dim}")

    @pytest.mark.parametrize(
        "name,method,data_fn", REDUCE_MINMAX_DIM_OPS,
        ids=[o[0] for o in REDUCE_MINMAX_DIM_OPS]
    )
    @pytest.mark.parametrize("shape,dim,keepdim", REDUCE_DIM_CASES,
                             ids=[f"{s}_d{d}_k{k}" for s, d, k in REDUCE_DIM_CASES])
    def test_amax_amin(self, name, method, data_fn, shape, dim, keepdim, device):
        data = data_fn(*shape)
        r, t = make_pair(data, device)
        check_match(
            getattr(r, method)(dim=dim, keepdim=keepdim),
            getattr(t, method)(dim=dim, keepdim=keepdim),
            context=f"{name} dim={dim}")

    @pytest.mark.parametrize("op", ["sum", "mean", "prod"])
    def test_module_level_dim(self, op, device):
        data = randf(10, 20)
        r, t = make_pair(data, device)
        check_match(
            getattr(raxis, op)(r, dim=1),
            getattr(torch, op)(t, dim=1),
            context=f"raxis.{op}(dim=1)")


# ===========================================================================
# SHAPE / VIEW OPERATIONS
# ===========================================================================

class TestShapeOps:

    def test_reshape(self, device):
        r, t = make_pair(randf(100), device)
        check_match(r.reshape((10, 10)), t.reshape((10, 10)), context="reshape")

    def test_reshape_infer(self, device):
        r, t = make_pair(randf(10, 20), device)
        check_match(r.reshape((-1, 50)), t.reshape((-1, 50)), context="reshape -1")

    def test_view(self, device):
        r, t = make_pair(randf(10, 20), device)
        check_match(r.view((20, 10)), t.view((20, 10)), context="view")

    def test_transpose(self, device):
        data = randf(10, 20, 30)
        r, t = make_pair(data, device)
        check_match(r.transpose(0, 2), t.transpose(0, 2), context="transpose")

    def test_permute(self, device):
        data = randf(3, 5, 7)
        r, t = make_pair(data, device)
        check_match(r.permute((2, 0, 1)), t.permute((2, 0, 1)), context="permute")

    def test_swapdims(self, device):
        data = randf(3, 5, 7)
        r, t = make_pair(data, device)
        check_match(r.swapdims(0, 2), t.swapdims(0, 2), context="swapdims")

    @pytest.mark.parametrize("src,dst", [(0, 1), (2, 0), (-1, 0)])
    def test_movedim(self, src, dst, device):
        data = randf(3, 5, 7)
        r, t = make_pair(data, device)
        check_match(r.movedim(src, dst), t.movedim(src, dst), context=f"movedim({src},{dst})")

    def test_squeeze_all(self, device):
        r, t = make_pair(randf(1, 5, 1, 3), device)
        check_match(r.squeeze(), t.squeeze(), context="squeeze all")

    def test_squeeze_dim(self, device):
        r, t = make_pair(randf(1, 5, 1, 3), device)
        check_match(r.squeeze(0), t.squeeze(0), context="squeeze(0)")

    def test_squeeze_no_effect(self, device):
        r, t = make_pair(randf(5, 3), device)
        check_match(r.squeeze(0), t.squeeze(0), context="squeeze no effect")

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_unsqueeze(self, dim, device):
        r, t = make_pair(randf(10, 20), device)
        check_match(r.unsqueeze(dim), t.unsqueeze(dim), context=f"unsqueeze({dim})")

    def test_contiguous(self, device):
        r, t = make_pair(randf(10, 20), device)
        r_t = r.transpose(0, 1).contiguous()
        t_t = t.transpose(0, 1).contiguous()
        check_match(r_t, t_t, context="contiguous after transpose")
        assert r_t.is_contiguous()

    def test_ops_on_noncontiguous(self, device):
        data = randf(10, 20)
        r, t = make_pair(data, device)
        r_nc = r.transpose(0, 1)
        t_nc = t.transpose(0, 1)
        check_match(r_nc + r_nc, t_nc + t_nc, context="add on transposed")
        check_match(r_nc.sum(), t_nc.sum(), context="sum on transposed")


# ===========================================================================
# BROADCAST UTILITIES
# ===========================================================================

class TestBroadcastUtils:

    @pytest.mark.parametrize("src_shape,dst_shape", [
        ((3, 1),    (3, 5)),
        ((1, 5),    (3, 5)),
        ((5,),      (2, 3, 5)),
        ((1,),      (4, 4)),
        ((1, 1, 5), (2, 3, 5)),
    ])
    def test_broadcast_to(self, src_shape, dst_shape, device):
        r, t = make_pair(randf(*src_shape), device)
        check_match(
            r.broadcast_to(dst_shape),
            t.broadcast_to(dst_shape),
            context=f"broadcast_to {src_shape}->{dst_shape}")

    @pytest.mark.parametrize("src_shape,dst_shape", [
        ((3, 1), (3, 5)),
        ((1,),   (2, 3, 5)),
    ])
    def test_broadcast_to_module(self, src_shape, dst_shape, device):
        r, t = make_pair(randf(*src_shape), device)
        check_match(
            raxis.broadcast_to(r, dst_shape),
            torch.broadcast_to(t, dst_shape),
            context=f"raxis.broadcast_to {src_shape}->{dst_shape}")

    def test_broadcast_tensors(self, device):
        r_a, t_a = make_pair(randf(3, 1, seed=1), device)
        r_b, t_b = make_pair(randf(1, 5, seed=2), device)
        r_results = raxis.broadcast_tensors([r_a, r_b])
        t_results = torch.broadcast_tensors(t_a, t_b)
        for r_out, t_out in zip(r_results, t_results):
            check_match(r_out, t_out, context="broadcast_tensors")

    @pytest.mark.parametrize("shapes,expected", [
        ([(3, 1), (1, 5)],       (3, 5)),
        ([(5,), (2, 3, 5)],      (2, 3, 5)),
        ([(1,), (3,), (1, 3)],   (1, 3)),
        ([(2, 1, 5), (1, 3, 1)], (2, 3, 5)),
    ])
    def test_broadcast_shapes(self, shapes, expected, device):
        result = raxis.broadcast_shapes(shapes)
        assert tuple(result) == expected, (
            f"broadcast_shapes {shapes} -> {result}, expected {expected}")


# ===========================================================================
# INDEXING AND SLICING
# ===========================================================================

class TestIndexing:

    def test_single_index(self, device):
        r, t = make_pair(randf(5, 10), device)
        check_match(r[2], t[2], context="single index")

    def test_slice(self, device):
        r, t = make_pair(randf(10, 20), device)
        check_match(r[2:5], t[2:5], context="slice")
        check_match(r[2:5, 3:10], t[2:5, 3:10], context="multi-slice")
        check_match(r[:, :5], t[:, :5], context="slice with :")

    def test_negative_index(self, device):
        r, t = make_pair(randf(10, 20), device)
        check_match(r[-1], t[-1], context="negative index")

    def test_setitem_scalar(self, device):
        r, t = raxis.zeros((3, 3)), torch.zeros((3, 3))
        r[1, 1] = 42.0; t[1, 1] = 42.0
        check_match(r, t, context="setitem scalar")

    def test_setitem_tensor(self, device):
        r, t = raxis.zeros((4, 4)), torch.zeros((4, 4))
        r[0] = raxis.tensor([1.0, 2.0, 3.0, 4.0])
        t[0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        check_match(r, t, context="setitem tensor")


# ===========================================================================
# EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_scalar_tensor_ops(self, device):
        r_a, t_a = raxis.tensor(3.0), torch.tensor(3.0)
        r_b, t_b = raxis.tensor(4.0), torch.tensor(4.0)
        check_match(r_a + r_b, t_a + t_b, context="scalar + scalar")
        check_match(r_a * r_b, t_a * t_b, context="scalar * scalar")
        check_match(r_a - r_b, t_a - t_b, context="scalar - scalar")
        check_match(r_a / r_b, t_a / t_b, context="scalar / scalar")

    def test_scalar_tensor_broadcast(self, device):
        r_s, t_s = raxis.tensor(10.0), torch.tensor(10.0)
        r_v, t_v = raxis.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.0])
        check_match(r_s + r_v, t_s + t_v, context="0d + 1d")
        check_match(r_v + r_s, t_v + t_s, context="1d + 0d")

    def test_add_zero_identity(self, device):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(r + 0, t + 0, context="+ 0")

    def test_mul_one_identity(self, device):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(r * 1, t * 1, context="* 1")

    def test_sub_self_is_zero(self, device):
        data = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(r - r, t - t, context="a - a")


# ===========================================================================
# item()
# ===========================================================================

class TestItem:

    def test_item_float(self, device):
        assert abs(raxis.tensor(3.14).item() - torch.tensor(3.14).item()) < 1e-5

    def test_item_int(self, device):
        assert raxis.tensor(42, dtype=raxis.int32).item() == torch.tensor(42, dtype=torch.int32).item()

    def test_item_bool(self, device):
        r = raxis.tensor(True, dtype=raxis.bool)
        t = torch.tensor(True, dtype=torch.bool)
        assert r.item() == t.item()
        assert type(r.item()) is type(t.item())

    def test_item_nd(self, device):
        assert (raxis.tensor([[[2.32]]]).item() - 2.32) < 1e-5

    def test_item_error(self, device):
        with pytest.raises(RuntimeError):
            raxis.tensor([1.0, 2.0]).item()


# ===========================================================================
# tolist() and numpy() ROUND-TRIP
# ===========================================================================

class TestRoundTrips:

    def test_tolist_1d(self, device):
        assert raxis.tensor([1.0, 2.0, 3.0]).tolist() == pytest.approx([1.0, 2.0, 3.0])

    def test_tolist_2d(self, device):
        data = [[1.0, 2.0], [3.0, 4.0]]
        result = raxis.tensor(data).tolist()
        for row_r, row_d in zip(result, data):
            assert row_r == pytest.approx(row_d)

    def test_numpy_roundtrip(self, device):
        data = randf(5, 10)
        r_np = raxis.tensor(data.tolist()).numpy()
        np.testing.assert_allclose(r_np, data, atol=1e-6)

    def test_from_numpy_roundtrip(self, device):
        data = randf(5, 10)
        r_np = raxis.from_numpy(data).numpy()
        np.testing.assert_allclose(r_np, data, atol=1e-6)


# ===========================================================================
# FACTORIES
# ===========================================================================

class TestFactories:

    @pytest.mark.parametrize("factory,expected", [
        ("zeros", 0.0),
        ("ones",  1.0),
    ])
    def test_constant_factories(self, factory, expected, device):
        r = getattr(raxis, factory)((3, 4))
        t = getattr(torch, factory)((3, 4))
        check_match(r, t, context=f"{factory}")

    def test_full(self, device):
        r = raxis.full((3, 4), 7.5)
        t = torch.full((3, 4), 7.5)
        check_match(r, t, context="full")

    def test_empty_shape(self, device):
        r = raxis.empty((5, 3))
        assert tuple(r.shape) == (5, 3)

    @pytest.mark.parametrize("factory", ["zeros_like", "ones_like", "empty_like"])
    def test_like_shape_and_dtype(self, factory, device):
        src = raxis.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = getattr(raxis, factory)(src)
        assert tuple(out.shape) == (2, 2)
        assert out.dtype == src.dtype

    @pytest.mark.parametrize("factory", ["zeros_like", "ones_like"])
    def test_like_dtype_override(self, factory, device):
        src = raxis.tensor([1.0, 2.0])
        out = getattr(raxis, factory)(src, dtype=raxis.int32)
        assert out.dtype == raxis.int32

    def test_full_like(self, device):
        src = raxis.tensor([1.0, 2.0, 3.0])
        r = raxis.full_like(src, 42.0)
        t = torch.full_like(torch.tensor([1.0, 2.0, 3.0]), 42.0)
        check_match(r, t, context="full_like")

    def test_rand_range(self, device):
        raxis.manual_seed(42)
        r = raxis.rand((1000,))
        r_np = np.array(r)
        assert r_np.min() >= 0.0
        assert r_np.max() <= 1.0
        assert tuple(r.shape) == (1000,)

    def test_randn_stats(self, device):
        raxis.manual_seed(42)
        r = raxis.randn((10000,))
        r_np = np.array(r)
        assert abs(r_np.mean()) < 0.1
        assert abs(r_np.std() - 1.0) < 0.1

    def test_randint_range(self, device):
        raxis.manual_seed(42)
        r = raxis.randint(0, 10, (1000,))
        r_np = np.array(r)
        assert r_np.min() >= 0
        assert r_np.max() < 10

    def test_rand_like(self, device):
        src = raxis.ones((5, 5))
        raxis.manual_seed(42)
        r = raxis.rand_like(src)
        assert tuple(r.shape) == (5, 5)
        assert r.dtype == src.dtype
        r_np = np.array(r)
        assert r_np.min() >= 0.0
        assert r_np.max() <= 1.0

    def test_randn_like(self, device):
        src = raxis.ones((5, 5))
        r = raxis.randn_like(src)
        assert tuple(r.shape) == (5, 5)
        assert r.dtype == src.dtype

    def test_randint_like(self, device):
        src = raxis.ones((5, 5), dtype=raxis.int64)
        r = raxis.randint_like(src, 0, 100)
        assert tuple(r.shape) == (5, 5)
        r_np = np.array(r)
        assert r_np.min() >= 0
        assert r_np.max() < 100


class TestRandom:
    """Philox-based rand/randn/randint. The values differ from torch's for the
    same seed (torch draws from mt19937), so these tests check the
    distributions and the reproducibility guarantees."""

    # Around the batch boundaries: 64 floats, 32 doubles or int64s per batch,
    # 16-element Box-Muller chunks, 8192-element parallel tasks
    SIZES = [1, 15, 17, 33, 63, 65, 1000, 100_003]
    DTYPES = [raxis.float32, raxis.float64]

    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_uniform_moments(self, n, dtype):
        raxis.manual_seed(0)
        x = np.array(raxis.zeros((n,), dtype=dtype).uniform_(-3.0, 5.0))
        assert x.min() >= -3.0 and x.max() < 5.0
        if n >= 1000:
            # U(-3, 5): mean 1, variance 64 / 12
            assert abs(x.mean() - 1.0) < 5 * np.sqrt(64 / 12 / n)
            assert abs(x.var() / (64 / 12) - 1.0) < 0.1

    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_normal_moments(self, n, dtype):
        raxis.manual_seed(0)
        x = np.array(raxis.zeros((n,), dtype=dtype).normal_(2.0, 3.0))
        assert np.isfinite(x).all()
        if n >= 100_000:
            z = (x - 2.0) / 3.0
            assert abs(z.mean()) < 5 / np.sqrt(n)
            assert abs(z.std() - 1.0) < 0.01
            assert abs((z ** 3).mean()) < 0.05      # skewness 0
            assert abs((z ** 4).mean() - 3.0) < 0.1  # kurtosis 3
            assert abs((np.abs(z) < 1).mean() - 0.6827) < 0.01
            # The two outputs of each Box-Muller pair (j, j + 8 in a chunk of
            # 16) must be uncorrelated
            chunks = z[: n // 16 * 16].reshape(-1, 16)
            assert abs(np.corrcoef(chunks[:, :8].ravel(), chunks[:, 8:].ravel())[0, 1]) < 0.02

    def test_randn_std_zero(self):
        x = np.array(raxis.zeros((100,)).normal_(1.5, 0.0))
        np.testing.assert_array_equal(x, 1.5)

    @pytest.mark.parametrize("dtype", [raxis.int64, raxis.int32, raxis.uint8])
    def test_randint_counts(self, dtype):
        raxis.manual_seed(0)
        n = 100_000
        x = np.array(raxis.randint(3, 13, (n,), dtype=dtype))
        counts = np.bincount(x.astype(np.int64) - 3, minlength=10)
        assert len(counts) == 10
        # Each count is Binomial(n, 0.1): standard deviation ~95
        assert np.abs(counts - n / 10).max() < 5 * np.sqrt(n * 0.1 * 0.9)

    def test_randint_wide_and_degenerate_ranges(self):
        raxis.manual_seed(0)
        low, high = -(2 ** 62), 2 ** 62
        x = np.array(raxis.randint(low, high, (10_000,)))
        assert x.min() >= low and x.max() < high
        assert abs(x.mean() / high) < 0.05
        np.testing.assert_array_equal(np.array(raxis.randint(7, 8, (50,))), 7)
        b = np.array(raxis.randint(0, 2, (1000,), dtype=raxis.bool))
        assert 0.4 < b.mean() < 0.6

    @pytest.mark.parametrize("factory", ["rand", "randn", "randint"])
    def test_seeded_and_independent_draws(self, factory):
        def draw():
            if factory == "randint":
                return np.array(raxis.randint(0, 1000, (5000,)))
            return np.array(getattr(raxis, factory)((5000,)))
        raxis.manual_seed(11)
        first, second = draw(), draw()
        raxis.manual_seed(11)
        np.testing.assert_array_equal(draw(), first)
        # Each call reserves new Philox counters
        assert (first != second).mean() > 0.9

    def test_fill_order_is_row_major_for_views(self):
        # A non-contiguous tensor gets the values of a contiguous tensor of
        # its shape, in row-major order
        raxis.manual_seed(5)
        expected = np.array(raxis.rand((40, 50)))
        base = raxis.zeros((50, 40))
        view = base.transpose(0, 1)
        raxis.manual_seed(5)
        view.uniform_()
        np.testing.assert_array_equal(np.array(view), expected)
        np.testing.assert_array_equal(np.array(base), expected.T)

    def test_independent_of_thread_count(self):
        threads = raxis.get_num_threads()
        results = []
        try:
            for n in (1, threads):
                raxis.set_num_threads(n)
                raxis.manual_seed(3)
                results.append([np.array(raxis.rand((100_003,))),
                                np.array(raxis.randn((100_003,), dtype=raxis.float64)),
                                np.array(raxis.randint(-50, 50, (100_003,)))])
        finally:
            raxis.set_num_threads(threads)
        for a, b in zip(*results):
            np.testing.assert_array_equal(a, b)

    def test_independent_of_cpu_capability(self):
        # The SIMD and scalar Philox paths draw the same words, so uniform and
        # randint match exactly; randn goes through SIMD log/sin/cos (sleef)
        # or the scalar libm ones, which may differ in the last ulp.
        code = ("import json, raxis, numpy as np; raxis.manual_seed(9); "
                "print(json.dumps([np.array(raxis.rand((1001,))).tolist(), "
                "np.array(raxis.randint(0, 10**12, (1001,))).tolist(), "
                "np.array(raxis.randn((1001,))).tolist()]))")
        outputs = []
        for capability in ("default", ""):
            env = dict(os.environ, RAXIS_CPU_CAPABILITY=capability)
            result = subprocess.run([sys.executable, "-c", code], env=env,
                                    capture_output=True, text=True, check=True)
            outputs.append(json.loads(result.stdout))
        (rand_a, randint_a, randn_a), (rand_b, randint_b, randn_b) = outputs
        assert rand_a == rand_b
        assert randint_a == randint_b
        np.testing.assert_allclose(randn_a, randn_b, rtol=1e-5, atol=1e-6)


# ===========================================================================
# LINEAR ALGEBRA
# ===========================================================================

class TestDot:

    SIZES = [1, 4, 16, 128, 1024]

    @pytest.mark.parametrize("n", SIZES)
    @pytest.mark.parametrize("dtype", MM_DTYPES, ids=[np.dtype(d).name for d in MM_DTYPES])
    def test_dot(self, n, dtype, device):
        a_data = randf(n, seed=42, lo=-2.0, hi=2.0).astype(dtype)
        b_data = randf(n, seed=99, lo=-2.0, hi=2.0).astype(dtype)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        atol = 1e-4 if dtype == np.float32 else 1e-10
        check_match(raxis.dot(r_a, r_b), torch.dot(t_a, t_b),
                    atol=atol, context=f"dot n={n} {np.dtype(dtype).name}")

    def test_dot_method(self, device):
        a_data = randf(64, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(64, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(r_a.dot(r_b), t_a.dot(t_b), context="dot method")

    def test_dot_strided(self, device):
        a_data = randf(64, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(64, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(raxis.dot(r_a[::2], r_b[::2]),
                    torch.dot(t_a[::2], t_b[::2]),
                    context="dot strided")

    def test_dot_size_mismatch(self, device):
        with pytest.raises(RuntimeError):
            raxis.dot(raxis.randn([3]), raxis.randn([5]))

    def test_dot_dim_error(self, device):
        with pytest.raises(RuntimeError):
            raxis.dot(raxis.randn([3, 4]), raxis.randn([3, 4]))


class TestMV:

    SHAPES = [
        ((4, 8), (8,)),
        ((16, 32), (32,)),
        ((128, 64), (64,)),
        ((256, 128), (128,)),
    ]

    @pytest.mark.parametrize("shapes", SHAPES, ids=[f"{a}@{b}" for a, b in SHAPES])
    @pytest.mark.parametrize("dtype", MM_DTYPES, ids=[np.dtype(d).name for d in MM_DTYPES])
    def test_mv(self, shapes, dtype, device):
        (m, k), (k_,) = shapes
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0).astype(dtype)
        v_data = randf(k, seed=99, lo=-2.0, hi=2.0).astype(dtype)
        r_a, t_a = make_pair(a_data, device)
        r_v, t_v = make_pair(v_data, device)
        atol = 1e-4 if dtype == np.float32 else 1e-10
        check_match(raxis.mv(r_a, r_v), torch.mv(t_a, t_v),
                    atol=atol, context=f"mv {shapes} {np.dtype(dtype).name}")

    def test_mv_method(self, device):
        a_data = randf(16, 32, seed=42, lo=-2.0, hi=2.0)
        v_data = randf(32, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_v, t_v = make_pair(v_data, device)
        check_match(r_a.mv(r_v), t_a.mv(t_v), context="mv method")

    def test_mv_transposed(self, device):
        a_data = randf(32, 16, seed=42, lo=-2.0, hi=2.0)
        v_data = randf(32, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_v, t_v = make_pair(v_data, device)
        check_match(raxis.mv(r_a.transpose(0, 1), r_v),
                    torch.mv(t_a.transpose(0, 1), t_v),
                    context="mv transposed mat")


class TestAddmv:

    def test_basic(self, device):
        m, k = 16, 32
        self_data = randf(m, seed=1, lo=-2.0, hi=2.0)
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0)
        v_data = randf(k, seed=99, lo=-2.0, hi=2.0)
        r_self, t_self = make_pair(self_data, device)
        r_a, t_a = make_pair(a_data, device)
        r_v, t_v = make_pair(v_data, device)
        check_match(
            raxis.addmv(r_self, r_a, r_v),
            torch.addmv(t_self, t_a, t_v),
            context="addmv basic")

    def test_with_scalars(self, device):
        m, k = 8, 16
        self_data = randf(m, seed=1, lo=-2.0, hi=2.0)
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0)
        v_data = randf(k, seed=99, lo=-2.0, hi=2.0)
        r_self, t_self = make_pair(self_data, device)
        r_a, t_a = make_pair(a_data, device)
        r_v, t_v = make_pair(v_data, device)
        check_match(
            raxis.addmv(r_self, r_a, r_v, beta=0.5, alpha=2.0),
            torch.addmv(t_self, t_a, t_v, beta=0.5, alpha=2.0),
            context="addmv beta=0.5 alpha=2.0")

    def test_beta_zero(self, device):
        m, k = 4, 8
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0)
        v_data = randf(k, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_v, t_v = make_pair(v_data, device)
        r_self = raxis.zeros([m])
        t_self = torch.zeros(m)
        check_match(
            raxis.addmv(r_self, r_a, r_v, beta=0, alpha=1),
            torch.addmv(t_self, t_a, t_v, beta=0, alpha=1),
            context="addmv beta=0")


class TestMM:

    @pytest.mark.parametrize("shapes", MM_SHAPES, ids=[f"{a}@{b}" for a, b in MM_SHAPES])
    @pytest.mark.parametrize("dtype", MM_DTYPES, ids=[np.dtype(d).name for d in MM_DTYPES])
    def test_mm(self, shapes, dtype, device):
        (m, k), (_, n) = shapes
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0).astype(dtype)
        b_data = randf(k, n, seed=99, lo=-2.0, hi=2.0).astype(dtype)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        atol = 1e-4 if dtype == np.float32 else 1e-10
        check_match(raxis.mm(r_a, r_b), torch.mm(t_a, t_b),
                    atol=atol, context=f"mm {shapes} {np.dtype(dtype).name}")

    @pytest.mark.parametrize("shapes", MM_SHAPES, ids=[f"{a}@{b}" for a, b in MM_SHAPES])
    def test_mm_method(self, shapes, device):
        (m, k), (_, n) = shapes
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(k, n, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(r_a.mm(r_b), t_a.mm(t_b), context=f"mm method {shapes}")

    def test_mm_transposed_inputs(self, device):
        a_data = randf(32, 64, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(48, 64, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(raxis.mm(r_a, r_b.transpose(0, 1)),
                    torch.mm(t_a, t_b.transpose(0, 1)), context="mm transposed B")
        check_match(raxis.mm(r_a.transpose(0, 1), r_a),
                    torch.mm(t_a.transpose(0, 1), t_a), context="mm transposed A (gram)")

    def test_mm_shape_error(self, device):
        with pytest.raises(RuntimeError):
            raxis.mm(raxis.randn([3, 4]), raxis.randn([5, 6]))

    def test_mm_dim_error(self, device):
        with pytest.raises(RuntimeError):
            raxis.mm(raxis.randn([3, 4, 5]), raxis.randn([5, 6]))


class TestAddmm:

    @pytest.mark.parametrize("dtype", MM_DTYPES, ids=[np.dtype(d).name for d in MM_DTYPES])
    def test_basic(self, dtype, device):
        m, k, n = 16, 32, 8
        self_data = randf(m, n, seed=1, lo=-2.0, hi=2.0).astype(dtype)
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0).astype(dtype)
        b_data = randf(k, n, seed=99, lo=-2.0, hi=2.0).astype(dtype)
        r_self, t_self = make_pair(self_data, device)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        atol = 1e-4 if dtype == np.float32 else 1e-10
        check_match(raxis.addmm(r_self, r_a, r_b),
                    torch.addmm(t_self, t_a, t_b),
                    atol=atol, context=f"addmm basic {np.dtype(dtype).name}")

    def test_with_scalars(self, device):
        m, k, n = 8, 16, 4
        self_data = randf(m, n, seed=1, lo=-2.0, hi=2.0)
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(k, n, seed=99, lo=-2.0, hi=2.0)
        r_self, t_self = make_pair(self_data, device)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(
            raxis.addmm(r_self, r_a, r_b, beta=0.5, alpha=2.0),
            torch.addmm(t_self, t_a, t_b, beta=0.5, alpha=2.0),
            context="addmm beta=0.5 alpha=2.0")

    def test_beta_zero(self, device):
        m, k, n = 4, 8, 4
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(k, n, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        r_self = raxis.zeros([m, n])
        t_self = torch.zeros(m, n)
        check_match(
            raxis.addmm(r_self, r_a, r_b, beta=0, alpha=1),
            torch.addmm(t_self, t_a, t_b, beta=0, alpha=1),
            context="addmm beta=0")

    def test_bias_broadcast(self, device):
        m, k, n = 8, 16, 4
        bias_data = randf(n, seed=1, lo=-2.0, hi=2.0)
        a_data = randf(m, k, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(k, n, seed=99, lo=-2.0, hi=2.0)
        r_bias, t_bias = make_pair(bias_data, device)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(
            raxis.addmm(r_bias, r_a, r_b),
            torch.addmm(t_bias, t_a, t_b),
            context="addmm bias broadcast")


class TestBMM:

    BMM_SHAPES = [
        ((2, 4, 8), (2, 8, 4)),
        ((4, 16, 32), (4, 32, 16)),
        ((8, 64, 64), (8, 64, 64)),
    ]

    @pytest.mark.parametrize("shapes", BMM_SHAPES, ids=[f"{a}@{b}" for a, b in BMM_SHAPES])
    @pytest.mark.parametrize("dtype", MM_DTYPES, ids=[np.dtype(d).name for d in MM_DTYPES])
    def test_bmm(self, shapes, dtype, device):
        sa, sb = shapes
        a_data = randf(*sa, seed=42, lo=-2.0, hi=2.0).astype(dtype)
        b_data = randf(*sb, seed=99, lo=-2.0, hi=2.0).astype(dtype)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        atol = 1e-4 if dtype == np.float32 else 1e-10
        check_match(raxis.bmm(r_a, r_b), torch.bmm(t_a, t_b),
                    atol=atol, context=f"bmm {shapes} {np.dtype(dtype).name}")

    # m * n * k < 400 goes through a naive kernel (also for integers); larger
    # floating batches through one batched gemm when every item is a plain
    # or transposed matrix, and the rest through one gemm per item
    @pytest.mark.parametrize("layout", ["contiguous", "transposed", "sliced", "broadcast_batch"])
    @pytest.mark.parametrize("sizes", [(5, 3, 4, 6), (6, 33, 20, 17), (1, 40, 30, 20), (3, 1, 50, 1)],
                             ids=lambda s: "x".join(map(str, s)))
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64], ids=lambda d: np.dtype(d).name)
    def test_bmm_paths(self, dtype, sizes, layout):
        bs, n, k, m = sizes
        a_data = (randf(bs, n, k, seed=1, lo=-2.0, hi=2.0) * 4).astype(dtype)
        b_data = (randf(bs, k, m, seed=2, lo=-2.0, hi=2.0) * 4).astype(dtype)
        if layout == "transposed":
            a_data = np.ascontiguousarray(a_data.transpose(0, 2, 1))
            b_data = np.ascontiguousarray(b_data.transpose(0, 2, 1))
        if layout == "sliced":
            a_data = np.concatenate([a_data, a_data], axis=2)
        r_a, t_a = make_pair(a_data)
        r_b, t_b = make_pair(b_data)
        if layout == "transposed":
            r_a, t_a = r_a.transpose(1, 2), t_a.transpose(1, 2)
            r_b, t_b = r_b.transpose(1, 2), t_b.transpose(1, 2)
        if layout == "sliced":
            r_a, t_a = r_a[:, :, ::2], t_a[:, :, ::2]
        if layout == "broadcast_batch":
            r_b, t_b = r_b[:1].expand(bs, k, m), t_b[:1].expand(bs, k, m)
        atol = 1e-4 if dtype == np.float32 else 1e-10
        check_match(raxis.bmm(r_a, r_b), torch.bmm(t_a, t_b), atol=atol)

    def test_bmm_empty_contraction(self):
        r_a, t_a = make_pair(np.zeros((3, 4, 0), dtype=np.float32))
        r_b, t_b = make_pair(np.zeros((3, 0, 5), dtype=np.float32))
        check_match(raxis.bmm(r_a, r_b), torch.bmm(t_a, t_b))

    def test_bmm_method(self, device):
        a_data = randf(4, 8, 16, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(4, 16, 6, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(r_a.bmm(r_b), t_a.bmm(t_b), context="bmm method")


class TestAddbmm:

    def test_basic(self, device):
        bs, m, k, n = 4, 8, 16, 6
        self_data = randf(m, n, seed=1, lo=-2.0, hi=2.0)
        a_data = randf(bs, m, k, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(bs, k, n, seed=99, lo=-2.0, hi=2.0)
        r_self, t_self = make_pair(self_data, device)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(
            raxis.addbmm(r_self, r_a, r_b),
            torch.addbmm(t_self, t_a, t_b),
            context="addbmm basic")

    def test_with_scalars(self, device):
        bs, m, k, n = 3, 4, 8, 5
        self_data = randf(m, n, seed=1, lo=-2.0, hi=2.0)
        a_data = randf(bs, m, k, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(bs, k, n, seed=99, lo=-2.0, hi=2.0)
        r_self, t_self = make_pair(self_data, device)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(
            raxis.addbmm(r_self, r_a, r_b, beta=0.5, alpha=2.0),
            torch.addbmm(t_self, t_a, t_b, beta=0.5, alpha=2.0),
            context="addbmm beta=0.5 alpha=2.0")

    def test_beta_zero(self, device):
        bs, m, k, n = 4, 4, 8, 4
        a_data = randf(bs, m, k, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(bs, k, n, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        r_self = raxis.zeros([m, n])
        t_self = torch.zeros(m, n)
        check_match(
            raxis.addbmm(r_self, r_a, r_b, beta=0, alpha=1),
            torch.addbmm(t_self, t_a, t_b, beta=0, alpha=1),
            context="addbmm beta=0")


class TestMatmul:

    @pytest.mark.parametrize("desc,shape_a,shape_b", MATMUL_CASES,
                             ids=[c[0] for c in MATMUL_CASES])
    @pytest.mark.parametrize("dtype", MM_DTYPES, ids=[np.dtype(d).name for d in MM_DTYPES])
    def test_matmul(self, desc, shape_a, shape_b, dtype, device):
        a_data = randf(*shape_a, seed=42, lo=-2.0, hi=2.0).astype(dtype)
        b_data = randf(*shape_b, seed=99, lo=-2.0, hi=2.0).astype(dtype)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        r_out = raxis.matmul(r_a, r_b)
        t_out = torch.matmul(t_a, t_b)
        assert tuple(r_out.shape) == tuple(t_out.shape), \
            f"Shape mismatch {desc}: raxis={r_out.shape}, torch={t_out.shape}"
        atol = 1e-4 if dtype == np.float32 else 1e-10
        rtol = 1e-4 if dtype == np.float32 else 1e-10
        check_match(r_out, t_out, atol=atol, rtol=rtol,
                    context=f"matmul {desc} {np.dtype(dtype).name}")

    def test_matmul_operator(self, device):
        a_data = randf(8, 16, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(16, 4, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(r_a @ r_b, t_a @ t_b, context="matmul @ operator")

    def test_matmul_method(self, device):
        a_data = randf(8, 16, seed=42, lo=-2.0, hi=2.0)
        b_data = randf(16, 4, seed=99, lo=-2.0, hi=2.0)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        check_match(r_a.matmul(r_b), t_a.matmul(t_b), context="matmul method")


# ===========================================================================
# CAT / STACK / UNBIND
# ===========================================================================

class TestCatStackUnbind:

    def test_cat_basic(self):
        a_r, a_t = make_pair(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b_r, b_t = make_pair(np.array([[5, 6], [7, 8]], dtype=np.float32))
        check_match(raxis.cat([a_r, b_r], dim=0), torch.cat([a_t, b_t], dim=0), context="cat dim=0")
        check_match(raxis.cat([a_r, b_r], dim=1), torch.cat([a_t, b_t], dim=1), context="cat dim=1")

    def test_cat_negative_dim(self):
        a_r, a_t = make_pair(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b_r, b_t = make_pair(np.array([[5, 6]], dtype=np.float32))
        check_match(raxis.cat([a_r, b_r], dim=-2), torch.cat([a_t, b_t], dim=-2), context="cat dim=-2")

    def test_cat_dtype_promotion(self):
        a_r = raxis.tensor([1, 2, 3], dtype=raxis.int32)
        b_r = raxis.tensor([4.0, 5.0, 6.0], dtype=raxis.float32)
        a_t = torch.tensor([1, 2, 3], dtype=torch.int32)
        b_t = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
        r = raxis.cat([a_r, b_r])
        t = torch.cat([a_t, b_t])
        check_match(r, t, context="cat dtype promotion")
        assert r.dtype == raxis.float32

    def test_cat_single_tensor(self):
        a_r, a_t = make_pair(np.array([1, 2, 3], dtype=np.float32))
        check_match(raxis.cat([a_r]), torch.cat([a_t]), context="cat single tensor")

    def test_cat_non_contiguous(self):
        a_r, a_t = make_pair(np.arange(12, dtype=np.float32).reshape(3, 4))
        a_r_t = a_r.transpose(0, 1)
        a_t_t = a_t.transpose(0, 1)
        b_r, b_t = make_pair(np.ones((4, 3), dtype=np.float32))
        check_match(raxis.cat([a_r_t, b_r], dim=0), torch.cat([a_t_t, b_t], dim=0),
                     context="cat non-contiguous")

    def test_cat_many_tensors(self):
        rs = [raxis.tensor([float(i)]) for i in range(10)]
        ts = [torch.tensor([float(i)]) for i in range(10)]
        check_match(raxis.cat(rs), torch.cat(ts), context="cat many tensors")

    def test_cat_3d_dim1(self):
        a_r, a_t = make_pair(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        b_r, b_t = make_pair(np.ones((2, 5, 4), dtype=np.float32))
        check_match(raxis.cat([a_r, b_r], dim=1), torch.cat([a_t, b_t], dim=1),
                     context="cat 3d dim=1")

    # Above GRAIN_SIZE (32768 elements) the copies run in parallel, through
    # one reused iterator when every input has the same sizes and strides
    @pytest.mark.parametrize("dim", [0, 1, 2])
    @pytest.mark.parametrize("case", ["same", "different", "non_contiguous", "promotion", "int"])
    def test_cat_large(self, case, dim):
        shapes = [(40, 30, 50)] * 3
        if case == "different":
            shapes = [tuple(s + 7 * i if d == dim else s for d, s in enumerate(shapes[0])) for i in range(3)]
        dtypes = [np.float32] * 3
        if case == "promotion":
            dtypes = [np.float32, np.float64, np.int32]
        if case == "int":
            dtypes = [np.int64] * 3
        pairs = [make_pair((randf(*s, seed=i) * 100).astype(dt)) for i, (s, dt) in enumerate(zip(shapes, dtypes))]
        rs, ts = [p[0] for p in pairs], [p[1] for p in pairs]
        if case == "non_contiguous":
            # Same sizes, other strides: the per-input copy path
            rs[1] = rs[1].permute(2, 1, 0).contiguous().permute(2, 1, 0)
            ts[1] = ts[1].permute(2, 1, 0).contiguous().permute(2, 1, 0)
        check_match(raxis.cat(rs, dim), torch.cat(ts, dim))

    def test_cat_large_skips_empty(self):
        r, t = make_pair(randf(200, 300))
        r_e, t_e = make_pair(np.zeros(0, dtype=np.float32))
        check_match(raxis.cat([r_e, r, r_e, r], 0), torch.cat([t_e, t, t_e, t], 0))

    @pytest.mark.parametrize("dim", [0, 1, 2, -1])
    @pytest.mark.parametrize("case", ["same", "non_contiguous", "promotion"])
    def test_stack_large(self, case, dim):
        pairs = [make_pair(randf(60, 70, 30, seed=i)) for i in range(4)]
        rs, ts = [p[0] for p in pairs], [p[1] for p in pairs]
        if case == "non_contiguous":
            rs[2], ts[2] = rs[2].permute(2, 1, 0).contiguous().permute(2, 1, 0), \
                ts[2].permute(2, 1, 0).contiguous().permute(2, 1, 0)
        if case == "promotion":
            rs[0], ts[0] = rs[0].to(raxis.float64), ts[0].to(torch.float64)
        r_out, t_out = raxis.stack(rs, dim), torch.stack(ts, dim)
        check_match(r_out, t_out)
        assert str(r_out.dtype) == str(t_out.dtype).replace("torch.", "")

    def test_stack_basic(self):
        a_r, a_t = make_pair(np.array([1, 2, 3], dtype=np.float32))
        b_r, b_t = make_pair(np.array([4, 5, 6], dtype=np.float32))
        check_match(raxis.stack([a_r, b_r], dim=0), torch.stack([a_t, b_t], dim=0), context="stack dim=0")
        check_match(raxis.stack([a_r, b_r], dim=1), torch.stack([a_t, b_t], dim=1), context="stack dim=1")

    def test_stack_negative_dim(self):
        a_r, a_t = make_pair(np.array([1, 2, 3], dtype=np.float32))
        b_r, b_t = make_pair(np.array([4, 5, 6], dtype=np.float32))
        check_match(raxis.stack([a_r, b_r], dim=-1), torch.stack([a_t, b_t], dim=-1),
                     context="stack dim=-1")

    def test_stack_2d(self):
        a_r, a_t = make_pair(np.array([[1, 2], [3, 4]], dtype=np.float32))
        b_r, b_t = make_pair(np.array([[5, 6], [7, 8]], dtype=np.float32))
        for dim in range(3):
            check_match(raxis.stack([a_r, b_r], dim=dim), torch.stack([a_t, b_t], dim=dim),
                         context=f"stack 2d dim={dim}")

    def test_unbind_basic(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        r, t = make_pair(data)
        r_list = raxis.unbind(r, dim=0)
        t_list = torch.unbind(t, dim=0)
        assert len(r_list) == len(t_list)
        for i, (ri, ti) in enumerate(zip(r_list, t_list)):
            check_match(ri, ti, context=f"unbind dim=0 [{i}]")

    def test_unbind_dim1(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        r, t = make_pair(data)
        r_list = raxis.unbind(r, dim=1)
        t_list = torch.unbind(t, dim=1)
        assert len(r_list) == len(t_list)
        for i, (ri, ti) in enumerate(zip(r_list, t_list)):
            check_match(ri, ti, context=f"unbind dim=1 [{i}]")

    def test_unbind_returns_views(self):
        r = raxis.tensor([[1.0, 2.0], [3.0, 4.0]])
        parts = raxis.unbind(r, dim=0)
        assert len(parts) == 2

    def test_narrow_basic(self):
        data = np.arange(10, dtype=np.float32)
        r, t = make_pair(data)
        check_match(r.narrow(0, 2, 5), t.narrow(0, 2, 5), context="narrow 1d")

    def test_narrow_2d(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        r, t = make_pair(data)
        check_match(r.narrow(0, 1, 2), t.narrow(0, 1, 2), context="narrow 2d dim=0")
        check_match(r.narrow(1, 0, 2), t.narrow(1, 0, 2), context="narrow 2d dim=1")

    def test_stack_then_unbind_roundtrip(self):
        a_r = raxis.tensor([1.0, 2.0, 3.0])
        b_r = raxis.tensor([4.0, 5.0, 6.0])
        stacked = raxis.stack([a_r, b_r], dim=0)
        parts = raxis.unbind(stacked, dim=0)
        check_match(parts[0], torch.tensor([1.0, 2.0, 3.0]), context="stack-unbind roundtrip [0]")
        check_match(parts[1], torch.tensor([4.0, 5.0, 6.0]), context="stack-unbind roundtrip [1]")

    CAT_DTYPES = [
        (np.float32, "float32"),
        (np.float64, "float64"),
        (np.int32,   "int32"),
        (np.int64,   "int64"),
    ]

    @pytest.mark.parametrize("dtype,desc", CAT_DTYPES, ids=[d[1] for d in CAT_DTYPES])
    def test_cat_single_dtype(self, dtype, desc):
        a = np.arange(6, dtype=dtype).reshape(2, 3)
        b = np.arange(6, 12, dtype=dtype).reshape(2, 3)
        r_a, t_a = make_pair(a)
        r_b, t_b = make_pair(b)
        check_match(raxis.cat([r_a, r_b], dim=0), torch.cat([t_a, t_b], dim=0),
                     context=f"cat {desc} dim=0")

    @pytest.mark.parametrize("dtype,desc", CAT_DTYPES, ids=[d[1] for d in CAT_DTYPES])
    def test_stack_single_dtype(self, dtype, desc):
        a = np.arange(6, dtype=dtype).reshape(2, 3)
        b = np.arange(6, 12, dtype=dtype).reshape(2, 3)
        r_a, t_a = make_pair(a)
        r_b, t_b = make_pair(b)
        check_match(raxis.stack([r_a, r_b], dim=0), torch.stack([t_a, t_b], dim=0),
                     context=f"stack {desc} dim=0")

    MIXED_DTYPE_CASES = [
        (np.int32,   np.float32, "int32+float32"),
        (np.int32,   np.float64, "int32+float64"),
        (np.int64,   np.float32, "int64+float32"),
        (np.float32, np.float64, "float32+float64"),
        (np.int32,   np.int64,   "int32+int64"),
    ]

    @pytest.mark.parametrize("dtype_a,dtype_b,desc", MIXED_DTYPE_CASES,
                             ids=[c[2] for c in MIXED_DTYPE_CASES])
    def test_cat_mixed_dtype(self, dtype_a, dtype_b, desc):
        a = np.arange(6, dtype=dtype_a).reshape(2, 3)
        b = np.arange(6, 12, dtype=dtype_b).reshape(2, 3)
        r_a, t_a = make_pair(a)
        r_b, t_b = make_pair(b)
        r_out = raxis.cat([r_a, r_b], dim=0)
        t_out = torch.cat([t_a, t_b], dim=0)
        check_match(r_out, t_out, context=f"cat mixed {desc}")
        assert r_out.dtype.name == str(t_out.dtype).replace("torch.", "")

    @pytest.mark.parametrize("dtype_a,dtype_b,desc", MIXED_DTYPE_CASES,
                             ids=[c[2] for c in MIXED_DTYPE_CASES])
    def test_stack_mixed_dtype(self, dtype_a, dtype_b, desc):
        a = np.arange(6, dtype=dtype_a).reshape(2, 3)
        b = np.arange(6, 12, dtype=dtype_b).reshape(2, 3)
        r_a, t_a = make_pair(a)
        r_b, t_b = make_pair(b)
        r_out = raxis.stack([r_a, r_b], dim=0)
        t_out = torch.stack([t_a, t_b], dim=0)
        check_match(r_out, t_out, context=f"stack mixed {desc}")
        assert r_out.dtype.name == str(t_out.dtype).replace("torch.", "")

    def test_cat_both_non_contiguous(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        b = np.arange(12, 24, dtype=np.float32).reshape(3, 4)
        r_a, t_a = make_pair(a)
        r_b, t_b = make_pair(b)
        r_a_t, t_a_t = r_a.transpose(0, 1), t_a.transpose(0, 1)
        r_b_t, t_b_t = r_b.transpose(0, 1), t_b.transpose(0, 1)
        check_match(raxis.cat([r_a_t, r_b_t], dim=0), torch.cat([t_a_t, t_b_t], dim=0),
                     context="cat both non-contiguous dim=0")
        check_match(raxis.cat([r_a_t, r_b_t], dim=1), torch.cat([t_a_t, t_b_t], dim=1),
                     context="cat both non-contiguous dim=1")

    def test_cat_mixed_contiguity(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        b = np.ones((4, 3), dtype=np.float32)
        r_a, t_a = make_pair(a)
        r_b, t_b = make_pair(b)
        r_a_t, t_a_t = r_a.transpose(0, 1), t_a.transpose(0, 1)
        check_match(raxis.cat([r_a_t, r_b], dim=0), torch.cat([t_a_t, t_b], dim=0),
                     context="cat mixed contiguity")

    def test_stack_non_contiguous(self):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        b = np.arange(12, 24, dtype=np.float32).reshape(3, 4)
        r_a, t_a = make_pair(a)
        r_b, t_b = make_pair(b)
        r_a_t, t_a_t = r_a.transpose(0, 1), t_a.transpose(0, 1)
        r_b_t, t_b_t = r_b.transpose(0, 1), t_b.transpose(0, 1)
        for dim in range(3):
            check_match(raxis.stack([r_a_t, r_b_t], dim=dim),
                         torch.stack([t_a_t, t_b_t], dim=dim),
                         context=f"stack non-contiguous dim={dim}")

    def test_cat_non_contiguous_mixed_dtype(self):
        a = np.arange(12, dtype=np.int32).reshape(3, 4)
        b = np.arange(12, 24, dtype=np.float32).reshape(3, 4)
        r_a, t_a = make_pair(a)
        r_b, t_b = make_pair(b)
        r_a_t, t_a_t = r_a.transpose(0, 1), t_a.transpose(0, 1)
        r_b_t, t_b_t = r_b.transpose(0, 1), t_b.transpose(0, 1)
        r_out = raxis.cat([r_a_t, r_b_t], dim=0)
        t_out = torch.cat([t_a_t, t_b_t], dim=0)
        check_match(r_out, t_out, context="cat non-contiguous mixed dtype")

    def test_cat_sliced_inputs(self):
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        r, t = make_pair(data)
        check_match(raxis.cat([r[0:2], r[2:4]], dim=0),
                     torch.cat([t[0:2], t[2:4]], dim=0), context="cat sliced inputs")

    def test_stack_sliced_inputs(self):
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        r, t = make_pair(data)
        check_match(raxis.stack([r[0], r[1], r[2]], dim=0),
                     torch.stack([t[0], t[1], t[2]], dim=0), context="stack sliced inputs")