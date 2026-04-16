
import pytest
import raxis
import torch
import numpy as np


def get_devices():
    devices = ["cpu"]
    if hasattr(raxis, "cuda") and hasattr(raxis.cuda, "is_available") and raxis.cuda.is_available():
        devices.append("cuda")
    return devices

@pytest.fixture(params=get_devices(), scope="session")
def device(request):
    return request.param


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def randf(*shape, seed=42, lo=0.0, hi=1.0):
    rng = np.random.RandomState(seed)
    return (rng.rand(*shape) * (hi - lo) + lo).astype(np.float32)

def randf_positive(*shape, seed=42):
    return randf(*shape, seed=seed, lo=0.01, hi=10.0)

def randf_nonzero(*shape, seed=42):
    return randf(*shape, seed=seed, lo=0.1, hi=5.0)

def randf_centered(*shape, seed=42):
    return randf(*shape, seed=seed, lo=-5.0, hi=5.0)

def randf_small(*shape, seed=42):
    return randf(*shape, seed=seed, lo=-2.0, hi=2.0)

def randf_unit(*shape, seed=42):
    return randf(*shape, seed=seed, lo=0.1, hi=0.9)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_match(r_out, t_out, atol=1e-5, rtol=1e-4, context=""):
    assert tuple(r_out.shape) == tuple(t_out.shape), (
        f"Shape mismatch {context}: raxis={r_out.shape}, torch={t_out.shape}"
    )

    r_np = np.array(r_out)
    t_np = t_out.detach().cpu().numpy()

    if r_np.dtype != t_np.dtype:
        common = np.result_type(r_np.dtype, t_np.dtype)
        r_np = r_np.astype(common)
        t_np = t_np.astype(common)

    np.testing.assert_allclose(
        r_np, t_np, atol=atol, rtol=rtol,
        err_msg=f"Value mismatch ({context})"
    )


def make_pair(data, device="cpu"):
    r = raxis.from_numpy(data)
    t = torch.from_numpy(data.copy())

    if device != "cpu":
        try:
            r = r.to(device)
        except RuntimeError:
            pytest.skip(f"raxis.to('{device}') not implemented")
        t = t.to(device)

    return r, t


def try_backward(r_out, t_out, r_inputs, t_inputs, context=""):
    """Attempt backward pass. Skip if autograd not implemented."""
    t_out.sum().backward()
    try:
        r_out.sum().backward()
    except (RuntimeError, AttributeError):
        pytest.skip(f"{context}: raxis backward not implemented")

    for r_in, t_in in zip(r_inputs, t_inputs):
        if t_in.requires_grad and hasattr(r_in, "grad") and r_in.grad is not None:
            check_match(r_in.grad, t_in.grad, context=f"{context} grad")


def try_requires_grad(r, t):
    """Try enabling requires_grad on both tensors. Return whether it succeeded."""
    t = t.requires_grad_(True)
    try:
        r = r.requires_grad_(True)
        return r, t, True
    except (RuntimeError, AttributeError):
        return r, t, False


# ===========================================================================
# Op tables
#
# Adding a new op = adding one tuple to the appropriate list.
# Backward and CUDA support are detected at runtime
# ===========================================================================

# -- Shapes per category --
UNARY_SHAPES  = [(5,), (10, 20), (3, 10, 20)]
BINARY_SHAPES = [(1,), (100,), (10, 20), (5, 10, 20), (2, 3, 4, 5)]

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
]

REDUCE_DIM_CASES = [
    ((10, 20),    0,      False),
    ((10, 20),    1,      False),
    ((10, 20),    -1,     False),
    ((10, 20),    0,      True),
    ((5, 10, 20), 2,      False),
    ((5, 10, 20), (0, 2), False),
    ((5, 10, 20), -1,     True),
]

# -- Unary ops --
# (name, method, data_fn)
UNARY_OPS = [
    ("neg",     "neg",     randf_centered),
    ("abs",     "abs",     randf_centered),
    ("ceil",    "ceil",    randf_centered),
    ("floor",   "floor",   randf_centered),
    ("round",   "round",   randf_centered),
    ("exp",     "exp",     randf_small),
    ("log",     "log",     randf_positive),
    ("sqrt",    "sqrt",    randf_positive),
    ("sigmoid", "sigmoid", randf_centered),
]

UNARY_INPLACE_OPS = [
    ("neg_",     randf_centered),
    ("abs_",     randf_centered),
    ("ceil_",    randf_centered),
    ("floor_",   randf_centered),
    ("round_",   randf_centered),
    ("exp_",     randf_small),
    ("log_",     randf_positive),
    ("sqrt_",    randf_positive),
    ("sigmoid_", randf_centered),
]

# -- Binary arithmetic ops --
# (name, operator_str, data_fn_rhs)
BINARY_ARITHMETIC_OPS = [
    ("add", "+", randf),
    ("sub", "-", randf),
    ("mul", "*", randf),
    ("div", "/", randf_nonzero),
]

# -- Comparison ops --
CMP_OPS = ["gt", "ge", "lt", "le", "eq", "ne"]
CMP_SCALARS = [0.0, 0.5, -1.0, 100.0]

# -- Reduction ops (all-reduce) --
# (name, method, data_fn)
REDUCE_ALL_OPS = [
    ("sum",     "sum",     randf),
    ("prod",    "prod",    randf_unit),
    ("max",     "max",     randf),
    ("min",     "min",     randf),
    ("mean",    "mean",    randf),
    ("nansum",  "nansum",  randf),
    ("nanmean", "nanmean", randf),
]

# -- Reduction ops (dim-aware) --
# (name, method, data_fn)
REDUCE_DIM_OPS = [
    ("sum",     "sum",     randf),
    ("mean",    "mean",    randf),
    ("prod",    "prod",    randf_unit),
    ("nansum",  "nansum",  randf),
    ("nanmean", "nanmean", randf),
]

# amax/amin have different call signature (Vec dim, no dtype kwarg)
REDUCE_MINMAX_DIM_OPS = [
    ("amax", "amax", randf),
    ("amin", "amin", randf),
]

SCALAR_VALUES = [0, 1, -1, 2, 42, 0.5, -3.14, 1e-7, 1e7]


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

    @pytest.mark.parametrize("shape", UNARY_SHAPES)
    @pytest.mark.parametrize(
        "name,method,data_fn", UNARY_OPS,
        ids=[u[0] for u in UNARY_OPS]
    )
    def test_backward(self, shape, name, method, data_fn, device):
        data = data_fn(*shape)
        r, t = make_pair(data, device)
        r, t, grad_ok = try_requires_grad(r, t)
        if not grad_ok:
            pytest.skip(f"{name}: requires_grad not supported")

        r_out = getattr(r, method)()
        t_out = getattr(t, method)()
        try_backward(r_out, t_out, [r], [t], context=f"{name} {shape}")

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
    @pytest.mark.parametrize("shape", [(10, 20), (3, 10, 20)])
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

    # -- maximum / minimum --

    @pytest.mark.parametrize("op", ["maximum", "minimum"])
    @pytest.mark.parametrize("shape", [(10, 20), (5, 10, 20)])
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

    # -- pow --

    @pytest.mark.parametrize("exp", [0.5, 2.0, -1.0, 3])
    def test_pow_scalar(self, exp, device):
        data = randf_positive(10, 20)
        r, t = make_pair(data, device)
        check_match(r.pow(exp), t.pow(exp), context=f"pow({exp})")

    def test_pow_tensor(self, device):
        base = randf_positive(10, 20, seed=1)
        exp = randf(10, 20, seed=2, lo=0.5, hi=3.0)
        r_b, t_b = make_pair(base, device)
        r_e, t_e = make_pair(exp, device)
        check_match(r_b.pow(r_e), t_b.pow(t_e), context="pow(tensor)")

    def test_pow_operator(self, device):
        data = randf_positive(10, 20)
        r, t = make_pair(data, device)
        check_match(r ** 2.0, t ** 2.0, context="** operator")

    def test_pow_module(self, device):
        data = randf_positive(10, 20)
        r, t = make_pair(data, device)
        check_match(raxis.pow(r, 2.0), torch.pow(t, 2.0), context="raxis.pow")

    @pytest.mark.parametrize("lhs_shape,rhs_shape,desc", BROADCAST_CASES,
                             ids=[c[2] for c in BROADCAST_CASES])
    def test_pow_broadcast(self, lhs_shape, rhs_shape, desc, device):
        base = randf_positive(*lhs_shape, seed=1)
        exp = randf(*rhs_shape, seed=2, lo=0.5, hi=3.0)
        r_b, t_b = make_pair(base, device)
        r_e, t_e = make_pair(exp, device)
        check_match(r_b.pow(r_e), t_b.pow(t_e), context=f"pow {desc}")

    def test_pow_backward(self, device):
        data = randf_positive(10, 20)
        r, t = make_pair(data, device)
        r, t, grad_ok = try_requires_grad(r, t)
        if not grad_ok:
            pytest.skip("pow: requires_grad not supported")
        try_backward(r.pow(2.0), t.pow(2.0), [r], [t], context="pow")

    # -- logical_and / logical_or --

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
# BINARY ARITHMETIC OPS (add, sub, mul, div)
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

    @pytest.mark.parametrize("shape", [(10, 20), (5, 10, 20)])
    @pytest.mark.parametrize(
        "name,op_str,data_fn_rhs", BINARY_ARITHMETIC_OPS,
        ids=[o[0] for o in BINARY_ARITHMETIC_OPS]
    )
    def test_backward(self, shape, name, op_str, data_fn_rhs, device):
        a_data = randf(*shape, seed=42)
        b_data = data_fn_rhs(*shape, seed=99)
        r_a, t_a = make_pair(a_data, device)
        r_b, t_b = make_pair(b_data, device)
        r_a, t_a, grad_a = try_requires_grad(r_a, t_a)
        r_b, t_b, grad_b = try_requires_grad(r_b, t_b)
        if not grad_a and not grad_b:
            pytest.skip(f"{name}: requires_grad not supported")

        r_out = eval(f"r_a {op_str} r_b")
        t_out = eval(f"t_a {op_str} t_b")
        try_backward(r_out, t_out, [r_a, r_b], [t_a, t_b], context=f"{name} {shape}")


# ===========================================================================
# BROADCASTING (arithmetic)
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
# SCALAR OPERATIONS (tensor-scalar, scalar-tensor)
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
# METHOD-STYLE vs OPERATOR-STYLE
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
# IN-PLACE ARITHMETIC OPERATIONS
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
    @pytest.mark.parametrize("shape", [(10,), (10, 20), (5, 10, 20)])
    def test_forward(self, name, method, data_fn, shape, device):
        data = data_fn(*shape)
        r, t = make_pair(data, device)
        check_match(getattr(r, method)(), getattr(t, method)(), context=f"{name}() {shape}")

    @pytest.mark.parametrize(
        "name,method,data_fn", REDUCE_ALL_OPS,
        ids=[o[0] for o in REDUCE_ALL_OPS]
    )
    @pytest.mark.parametrize("shape", [(10,), (10, 20)])
    def test_backward(self, name, method, data_fn, shape, device):
        data = data_fn(*shape)
        r, t = make_pair(data, device)
        r, t, grad_ok = try_requires_grad(r, t)
        if not grad_ok:
            pytest.skip(f"{name}: requires_grad not supported")

        r_out = getattr(r, method)()
        t_out = getattr(t, method)()
        try_backward(r_out, t_out, [r], [t], context=f"{name}() {shape}")

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

    @pytest.mark.parametrize(
        "name,method,data_fn", REDUCE_DIM_OPS,
        ids=[o[0] for o in REDUCE_DIM_OPS]
    )
    @pytest.mark.parametrize("shape,dim,keepdim", [
        ((10, 20), 0, False),
        ((10, 20), 1, True),
        ((5, 10, 20), -1, False),
    ])
    def test_backward(self, name, method, data_fn, shape, dim, keepdim, device):
        data = data_fn(*shape)
        r, t = make_pair(data, device)
        r, t, grad_ok = try_requires_grad(r, t)
        if not grad_ok:
            pytest.skip(f"{name}: requires_grad not supported")

        r_out = self.reduce_call(r, method, dim=dim, keepdim=keepdim)
        t_out = self.reduce_call(t, method, dim=dim, keepdim=keepdim)
        try_backward(r_out, t_out, [r], [t], context=f"{name} dim={dim}")

    def reduce_call(self, x, method, dim, keepdim):
        fn = getattr(x, method)
        
        if method == "prod" and isinstance(dim, tuple):
            # NOTE: PyTorch API inconsistency, apply sequentially
            for d in sorted(dim, reverse=True):
                x = fn(dim=d, keepdim=keepdim)
                fn = getattr(x, method)
            return x
        else:
            return fn(dim=dim, keepdim=keepdim)

    @pytest.mark.parametrize("op", ["nansum", "nanmean"])
    @pytest.mark.parametrize("shape,dim,keepdim", [
        ((10, 20),    0,  False),
        ((10, 20),    1,  False),
        ((5, 10, 20), -1, True),
    ])
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

    # @pytest.mark.parametrize("src,dst", [(0, 1), (2, 0), (-1, 0)])
    # def test_movedim(self, src, dst, device):
    #     data = randf(3, 5, 7)
    #     r, t = make_pair(data, device)
    #     check_match(r.movedim(src, dst), t.movedim(src, dst), context=f"movedim({src},{dst})")

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
        r, t = make_pair(randf(5, 3), device)
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