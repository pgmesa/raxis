
import sys
import subprocess

import pytest
import torch
import raxis
import numpy as np


# Dtype pairs: (raxis_dtype, torch_dtype)
# Only the dtypes both frameworks share (uint16 excluded, no torch equivalent)
SHARED_DTYPES = [
    (raxis.bool,    torch.bool),
    (raxis.uint8,   torch.uint8),
    (raxis.int32,   torch.int32),
    (raxis.int64,   torch.int64),
    (raxis.float32, torch.float32),
    (raxis.float64, torch.float64),
]

RAXIS_DTYPES = [d[0] for d in SHARED_DTYPES]
TORCH_DTYPES = [d[1] for d in SHARED_DTYPES]

# Map raxis dtype -> torch dtype for quick lookup
R2T = {r: t for r, t in SHARED_DTYPES}
T2R = {t: r for r, t in SHARED_DTYPES}

SHARED_DTYPES_NP = [
    (raxis.bool,    bool),
    (raxis.uint8,   np.uint8),
    (raxis.int32,   np.int32),
    (raxis.int64,   np.int64),
    (raxis.float32, np.float32),
    (raxis.float64, np.float64),
]

R2N = {r: n for r, n in SHARED_DTYPES_NP}
N2R = {n: r for r, n in SHARED_DTYPES_NP}

DTYPE_NAMES = {d: d.name for d in RAXIS_DTYPES}


def assert_dtype_eq(raxis_tensor, torch_tensor, context=""):
    """Assert that the raxis tensor's dtype matches the torch tensor's dtype."""
    r_dtype = raxis_tensor.dtype
    t_dtype = torch_tensor.dtype

    # Map torch dtype to raxis equivalent for comparison
    expected_raxis = T2R.get(t_dtype)
    if expected_raxis is None:
        # torch may promote to a type we don't have (e.g. int16), skip
        pytest.skip(f"Torch dtype {t_dtype} has no raxis equivalent")

    assert r_dtype == expected_raxis, (
        f"Dtype mismatch{' (' + context + ')' if context else ''}: "
        f"raxis={r_dtype}, torch={t_dtype} (expected raxis={expected_raxis})"
    )


def assert_values_close(raxis_tensor, torch_tensor, atol=1e-6, rtol=1e-5, context=""):
    """Assert that values match between raxis and torch tensors."""
    r_np = np.array(raxis_tensor.detach().cpu())
    t_np = torch_tensor.detach().cpu().numpy()

    # Cast to same numpy type for comparison (use torch's type as reference)
    if r_np.dtype != t_np.dtype:
        t_np = t_np.astype(r_np.dtype)

    np.testing.assert_allclose(
        r_np, t_np, atol=atol, rtol=rtol,
        err_msg=f"Value mismatch{' (' + context + ')' if context else ''}"
    )


def make_test_data(dtype, shape=(4,)):
    """Create deterministic test data suitable for a given dtype."""
    if dtype in (raxis.bool, torch.bool):
        return [True, False, True, False][:np.prod(shape)]
    elif dtype in (raxis.uint8, torch.uint8):
        return list(range(1, np.prod(shape) + 1))
    elif dtype in (raxis.int32, torch.int32, raxis.int64, torch.int64):
        return list(range(-2, np.prod(shape) - 2))
    elif dtype in (raxis.float32, torch.float32, raxis.float64, torch.float64):
        return [x * 0.5 + 0.1 for x in range(np.prod(shape))]
    return list(range(np.prod(shape)))


# ============================================================================
# TENSOR CREATION: dtype inference
# ============================================================================

class TestTensorCreationDtype:
    """Verify that raxis.tensor() infers the same dtype as torch.tensor()."""

    # --- Python scalars ---

    def test_from_python_int(self):
        r = raxis.tensor(42)
        t = torch.tensor(42)
        assert_dtype_eq(r, t, "python int")

    def test_from_python_float(self):
        r = raxis.tensor(3.14)
        t = torch.tensor(3.14)
        assert_dtype_eq(r, t, "python float")

    def test_from_python_bool(self):
        r = raxis.tensor(True)
        t = torch.tensor(True)
        assert_dtype_eq(r, t, "python bool")

    # --- Homogeneous lists ---

    def test_from_int_list(self):
        r = raxis.tensor([1, 2, 3])
        t = torch.tensor([1, 2, 3])
        assert_dtype_eq(r, t, "int list")
        assert_values_close(r, t)

    def test_from_float_list(self):
        r = raxis.tensor([1.0, 2.0, 3.0])
        t = torch.tensor([1.0, 2.0, 3.0])
        assert_dtype_eq(r, t, "float list")
        assert_values_close(r, t)

    def test_from_bool_list(self):
        r = raxis.tensor([True, False, True])
        t = torch.tensor([True, False, True])
        assert_dtype_eq(r, t, "bool list")
        assert_values_close(r, t)

    # --- Mixed lists (promotion) ---

    def test_from_int_float_list(self):
        """[1, 2.0, 3] should promote to float."""
        r = raxis.tensor([1, 2.0, 3])
        t = torch.tensor([1, 2.0, 3])
        assert_dtype_eq(r, t, "int+float list")
        assert_values_close(r, t)

    def test_from_bool_int_list(self):
        """[True, 1, 2] should promote to int (not bool)."""
        r = raxis.tensor([True, 1, 2])
        t = torch.tensor([True, 1, 2])
        assert_dtype_eq(r, t, "bool+int list")
        assert_values_close(r, t)

    def test_from_bool_float_list(self):
        """[True, 1.0] should promote to float."""
        r = raxis.tensor([True, 1.0])
        t = torch.tensor([True, 1.0])
        assert_dtype_eq(r, t, "bool+float list")
        assert_values_close(r, t)

    def test_from_bool_int_float_list(self):
        """[True, 1, 2.0] should promote to float."""
        r = raxis.tensor([True, 1, 2.0])
        t = torch.tensor([True, 1, 2.0])
        assert_dtype_eq(r, t, "bool+int+float list")
        assert_values_close(r, t)

    # --- Nested lists ---

    def test_from_nested_int_list(self):
        data = [[1, 2], [3, 4]]
        r = raxis.tensor(data)
        t = torch.tensor(data)
        assert_dtype_eq(r, t, "nested int")
        assert r.shape == t.shape
        assert_values_close(r, t)

    def test_from_nested_float_list(self):
        data = [[1.0, 2.0], [3.0, 4.0]]
        r = raxis.tensor(data)
        t = torch.tensor(data)
        assert_dtype_eq(r, t, "nested float")
        assert r.shape == t.shape
        assert_values_close(r, t)

    def test_from_nested_mixed_list(self):
        data = [[1, 2.0], [3, 4]]
        r = raxis.tensor(data)
        t = torch.tensor(data)
        assert_dtype_eq(r, t, "nested mixed")
        assert r.shape == t.shape
        assert_values_close(r, t)

    # --- Explicit dtype override ---

    @pytest.mark.parametrize("r_dtype,t_dtype", SHARED_DTYPES, ids=DTYPE_NAMES.values())
    def test_explicit_dtype_from_int_list(self, r_dtype, t_dtype):
        """tensor([1, 2, 3], dtype=X) should produce dtype X."""
        r = raxis.tensor([1, 2, 3], dtype=r_dtype)
        t = torch.tensor([1, 2, 3], dtype=t_dtype)
        assert_dtype_eq(r, t, f"explicit {r_dtype}")
        assert_values_close(r, t)

    @pytest.mark.parametrize("r_dtype,t_dtype", SHARED_DTYPES, ids=DTYPE_NAMES.values())
    def test_explicit_dtype_from_float_list(self, r_dtype, t_dtype):
        """tensor([1.0, 2.5], dtype=X) should produce dtype X."""
        try:
            t = torch.tensor([1.0, 2.5], dtype=t_dtype)
        except TypeError:
            t = torch.from_numpy(np.array([1.0, 2.5], dtype=R2N[r_dtype]))
            return
        
        r = raxis.tensor([1.0, 2.5], dtype=r_dtype)
        assert_dtype_eq(r, t, f"float list -> {r_dtype}")
        assert_values_close(r, t)

    def test_without_numpy(self):
        """numpy is optional: tensor() must not import it."""
        # A None entry in sys.modules makes `import numpy` fail, as if it
        # were not installed
        code = (
            "import sys; sys.modules['numpy'] = None\n"
            "import raxis\n"
            "assert raxis.tensor([[1, 2], [3, 4]]).sum().item() == 10\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


# ============================================================================
# TYPE PROMOTION: tensor-tensor
# ============================================================================

class TestTensorTensorPromotion:
    """
    For every pair of shared dtypes, check that binary ops produce the 
    same result dtype as PyTorch.
    """

    @pytest.fixture(params=[
        ("add", lambda a, b: a + b),
        ("sub", lambda a, b: a - b),
        ("mul", lambda a, b: a * b),
        ("div", lambda a, b: a / b),
    ], ids=lambda p: p[0])
    def op(self, request):
        return request.param

    @pytest.fixture(params=SHARED_DTYPES, ids=[n for n in DTYPE_NAMES.values()])
    def lhs_dtype(self, request):
        return request.param  # (raxis_dtype, torch_dtype)

    @pytest.fixture(params=SHARED_DTYPES, ids=[n for n in DTYPE_NAMES.values()])
    def rhs_dtype(self, request):
        return request.param

    def test_promotion_dtype(self, op, lhs_dtype, rhs_dtype):
        op_name, op_fn = op
        r_lhs_dt, t_lhs_dt = lhs_dtype
        r_rhs_dt, t_rhs_dt = rhs_dtype

        lhs_data = make_test_data(r_lhs_dt)
        rhs_data = make_test_data(r_rhs_dt)

        # Avoid division by zero
        if op_name == "div":
            rhs_data = [max(x, 1) if not isinstance(x, bool) else True for x in rhs_data]

        try:
            t_lhs = torch.tensor(lhs_data, dtype=t_lhs_dt)
            t_rhs = torch.tensor(rhs_data, dtype=t_rhs_dt)
            t_out = op_fn(t_lhs, t_rhs)
        except RuntimeError:
            # If PyTorch errors, raxis should too
            r_lhs = raxis.tensor(lhs_data, dtype=r_lhs_dt)
            r_rhs = raxis.tensor(rhs_data, dtype=r_rhs_dt)
            with pytest.raises(RuntimeError):
                op_fn(r_lhs, r_rhs)
            return

        r_lhs = raxis.tensor(lhs_data, dtype=r_lhs_dt)
        r_rhs = raxis.tensor(rhs_data, dtype=r_rhs_dt)
        r_out = op_fn(r_lhs, r_rhs)

        context = f"{op_name}: {r_lhs_dt}-{r_rhs_dt}"
        assert_dtype_eq(r_out, t_out, context)
        assert_values_close(r_out, t_out, atol=1e-5, rtol=1e-4, context=context)


# ============================================================================
# TYPE PROMOTION: tensor-scalar & scalar-tensor
# ============================================================================

class TestTensorScalarPromotion:
    """
    PyTorch's scalar promotion rules:
      - int_tensor + int_scalar   -> int_tensor's dtype (scalar doesn't upcast)
      - int_tensor + float_scalar -> float (scalar forces promotion)
      - float_tensor + int_scalar -> float_tensor's dtype (scalar doesn't upcast)
      - bool_tensor + int_scalar  -> int (promotes out of bool)
    """

    SCALARS = [
        (True,  "bool"),
        (3,     "int"),
        (2.5,   "float"),
    ]

    @pytest.fixture(params=SHARED_DTYPES, ids=[n for n in DTYPE_NAMES.values()])
    def tensor_dtype(self, request):
        return request.param

    @pytest.fixture(params=SCALARS, ids=[s[1] for s in SCALARS])
    def scalar(self, request):
        return request.param

    @pytest.mark.parametrize("op_name,op_fn", [
        ("add", lambda t, s: t + s),
        ("sub", lambda t, s: t - s),
        ("mul", lambda t, s: t * s),
        ("div", lambda t, s: t / s),
    ])
    def test_tensor_op_scalar(self, tensor_dtype, scalar, op_name, op_fn):
        r_dt, t_dt = tensor_dtype
        scalar_val, scalar_name = scalar

        # Avoid div by zero for bool scalar
        if op_name == "div" and scalar_val in (0, False):
            pytest.skip("div by zero")

        data = make_test_data(r_dt)

        try:
            t = torch.tensor(data, dtype=t_dt)
            t_out = op_fn(t, scalar_val)
        except RuntimeError:
            r = raxis.tensor(data, dtype=r_dt)
            with pytest.raises(RuntimeError):
                op_fn(r, scalar_val)
            return

        r = raxis.tensor(data, dtype=r_dt)
        r_out = op_fn(r, scalar_val)

        context = f"{r_dt} {op_name} {scalar_name}({scalar_val})"
        assert_dtype_eq(r_out, t_out, context)
        assert_values_close(r_out, t_out, atol=1e-5, rtol=1e-4, context=context)

    @pytest.mark.parametrize("op_name,op_fn", [
        ("radd", lambda s, t: s + t),
        ("rsub", lambda s, t: s - t),
        ("rmul", lambda s, t: s * t),
        ("rdiv", lambda s, t: s / t),
    ])
    def test_scalar_op_tensor(self, tensor_dtype, scalar, op_name, op_fn):
        """Reverse operations: scalar-tensor."""
        r_dt, t_dt = tensor_dtype
        scalar_val, scalar_name = scalar

        data = make_test_data(r_dt)

        # Avoid div by zero
        if op_name == "rdiv":
            data = [max(x, 1) if not isinstance(x, bool) else True for x in data]

        try:
            t = torch.tensor(data, dtype=t_dt)
            t_out = op_fn(scalar_val, t)
        except RuntimeError:
            r = raxis.tensor(data, dtype=r_dt)
            with pytest.raises(RuntimeError):
                op_fn(scalar_val, r)
            return

        r = raxis.tensor(data, dtype=r_dt)
        r_out = op_fn(scalar_val, r)

        context = f"{scalar_name}({scalar_val}) {op_name} {r_dt}"
        assert_dtype_eq(r_out, t_out, context)
        assert_values_close(r_out, t_out, atol=1e-5, rtol=1e-4, context=context)


# ============================================================================
# IN-PLACE OP TYPE SAFETY
# ============================================================================

class TestInPlaceTypeSafety:
    """
    In-place ops should reject unsafe casts. PyTorch rules:
      - float_tensor += int_scalar   -> OK (int fits in float)
      - int_tensor += float_scalar   -> ERROR (can't narrow float -> int)
      - int_tensor += float_tensor   -> ERROR
      - float32_tensor += float64_tensor -> OK (narrows but same category)
    """

    def test_int_tensor_iadd_float_scalar_fails(self):
        """int32 tensor += 2.5 should fail (float -> int narrowing)."""
        t_torch = torch.tensor([1, 2, 3], dtype=torch.int32)
        t_raxis = raxis.tensor([1, 2, 3], dtype=raxis.int32)

        # PyTorch raises RuntimeError for this
        with pytest.raises(RuntimeError):
            t_torch += 2.5

        with pytest.raises(RuntimeError):
            t_raxis += 2.5

    def test_int_tensor_iadd_float_tensor_fails(self):
        """int32 tensor += float32 tensor should fail."""
        t_raxis = raxis.tensor([1, 2, 3], dtype=raxis.int32)
        f_raxis = raxis.tensor([1.0, 2.0, 3.0], dtype=raxis.float32)

        with pytest.raises(RuntimeError):
            t_raxis += f_raxis

    def test_float_tensor_iadd_int_scalar_ok(self):
        """float32 tensor += 2 should work (int safely fits in float)."""
        r = raxis.tensor([1.0, 2.0, 3.0], dtype=raxis.float32)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        r += 2
        t += 2
        assert_dtype_eq(r, t, "float += int_scalar")
        assert_values_close(r, t)

    def test_float_tensor_iadd_int_tensor_ok(self):
        """float32 tensor += int32 tensor should work."""
        r = raxis.tensor([1.0, 2.0, 3.0], dtype=raxis.float32)
        r_int = raxis.tensor([10, 20, 30], dtype=raxis.int32)
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        t_int = torch.tensor([10, 20, 30], dtype=torch.int32)
        r += r_int
        t += t_int
        assert_dtype_eq(r, t, "float += int_tensor")
        assert_values_close(r, t)

    def test_bool_tensor_iadd_int_fails(self):
        """bool tensor += 1 should fail (int -> bool not allowed)."""
        r = raxis.tensor([True, False], dtype=raxis.bool)

        with pytest.raises(RuntimeError):
            r += 1

    @pytest.mark.parametrize("op_name,op_fn", [
        ("iadd", lambda t, s: t.__iadd__(s)),
        ("isub", lambda t, s: t.__isub__(s)),
        ("imul", lambda t, s: t.__imul__(s)),
        ("idiv", lambda t, s: t.__itruediv__(s)),
    ])
    def test_inplace_same_dtype_ok(self, op_name, op_fn):
        """In-place with same dtype should always work."""
        for r_dt, t_dt in SHARED_DTYPES:
            if r_dt == raxis.bool:
                continue  # bool arithmetic is weird, skip

            data_a = make_test_data(r_dt)
            data_b = make_test_data(r_dt)

            # Avoid div by zero
            if op_name == "idiv":
                data_b = [max(abs(x), 1) if not isinstance(x, bool) else True for x in data_b]

            r = raxis.tensor(data_a, dtype=r_dt)
            t = torch.tensor(data_a, dtype=t_dt)
            r_other = raxis.tensor(data_b, dtype=r_dt)
            t_other = torch.tensor(data_b, dtype=t_dt)

            try:
                op_fn(t, t_other)
            except RuntimeError:
                with pytest.raises(RuntimeError):
                    op_fn(r, r_other)
                continue

            op_fn(r, r_other)
            assert_dtype_eq(r, t, f"{op_name} same dtype {r_dt}")
            assert_values_close(r, t, atol=1e-4, rtol=1e-3,
                                context=f"{op_name} same dtype {r_dt}")


# ============================================================================
# SPECIFIC VALUE CORRECTNESS EDGE CASES
# ============================================================================

class TestValueEdgeCases:
    """Value-level correctness for tricky dtype combos."""

    def test_int_division_produces_float(self):
        """PyTorch: int / int -> float. Verify raxis matches."""
        r = raxis.tensor([6, 7, 10], dtype=raxis.int32)
        t = torch.tensor([6, 7, 10], dtype=torch.int32)
        r_out = r / raxis.tensor([3, 2, 4], dtype=raxis.int32)
        t_out = t / torch.tensor([3, 2, 4], dtype=torch.int32)

        assert_dtype_eq(r_out, t_out, "int / int dtype")
        assert_values_close(r_out, t_out, context="int / int values")

    def test_bool_plus_bool(self):
        """True + True -> 2 (as int)."""
        r = raxis.tensor([True]) + raxis.tensor([True])
        t = torch.tensor([True]) + torch.tensor([True])
        assert_dtype_eq(r, t, "bool + bool")
        assert_values_close(r, t)

    def test_bool_times_float(self):
        """True * 3.0 -> 3.0 (float)."""
        r = raxis.tensor([True, False]) * 3.0
        t = torch.tensor([True, False]) * 3.0
        assert_dtype_eq(r, t, "bool * float_scalar")
        assert_values_close(r, t)

    def test_bool_plus_int_tensor(self):
        """bool_tensor + int_tensor -> int."""
        r = raxis.tensor([True, False]) + raxis.tensor([10, 20], dtype=raxis.int32)
        t = torch.tensor([True, False]) + torch.tensor([10, 20], dtype=torch.int32)
        assert_dtype_eq(r, t, "bool + int tensor")
        assert_values_close(r, t)

    def test_uint8_overflow_wraps(self):
        """uint8: 250 + 10 should wrap to 4 (modular arithmetic)."""
        r = raxis.tensor([250], dtype=raxis.uint8) + raxis.tensor([10], dtype=raxis.uint8)
        t = torch.tensor([250], dtype=torch.uint8) + torch.tensor([10], dtype=torch.uint8)
        assert_dtype_eq(r, t, "uint8 overflow")
        assert_values_close(r, t, context="uint8 wrap")

    def test_float32_float64_promotion(self):
        """float32 + float64 -> float64."""
        r = raxis.tensor([1.0], dtype=raxis.float32) + raxis.tensor([2.0], dtype=raxis.float64)
        t = torch.tensor([1.0], dtype=torch.float32) + torch.tensor([2.0], dtype=torch.float64)
        assert_dtype_eq(r, t, "f32 + f64")
        assert_values_close(r, t)

    def test_int32_plus_float32_tensor(self):
        """int32 + float32 -> float32."""
        r = raxis.tensor([1, 2], dtype=raxis.int32) + raxis.tensor([0.5, 1.5], dtype=raxis.float32)
        t = torch.tensor([1, 2], dtype=torch.int32) + torch.tensor([0.5, 1.5], dtype=torch.float32)
        assert_dtype_eq(r, t, "i32 + f32")
        assert_values_close(r, t)

    def test_scalar_int_does_not_upcast_float_tensor(self):
        """float32_tensor + 3 -> float32 (int scalar doesn't widen)."""
        r = raxis.tensor([1.5, 2.5], dtype=raxis.float32) + 3
        t = torch.tensor([1.5, 2.5], dtype=torch.float32) + 3
        assert_dtype_eq(r, t, "f32 + int_scalar")
        assert_values_close(r, t)

    def test_scalar_float_upcasts_int_tensor(self):
        """int32_tensor + 2.5 -> float (float scalar forces promotion)."""
        r = raxis.tensor([1, 2, 3], dtype=raxis.int32) + 2.5
        t = torch.tensor([1, 2, 3], dtype=torch.int32) + 2.5
        assert_dtype_eq(r, t, "i32 + float_scalar")
        assert_values_close(r, t)


# ============================================================================
# EXPLICIT CAST WITH .to(dtype)
# ============================================================================

class TestExplicitCast:
    """Verify .to(dtype) produces correct dtype and values."""

    @pytest.mark.parametrize("src_r,src_t", SHARED_DTYPES, ids=DTYPE_NAMES.values())
    @pytest.mark.parametrize("dst_r,dst_t", SHARED_DTYPES, ids=DTYPE_NAMES.values())
    def test_to_dtype(self, src_r, src_t, dst_r, dst_t):
        data = [1, 0, 1, 0]  # Safe for all types including bool
        t = torch.tensor(data, dtype=src_t).to(dst_t)
        r = raxis.tensor(data, dtype=src_r).to(dst_r)

        assert_dtype_eq(r, t, f".to({dst_r}) from {src_r}")
        assert_values_close(r, t, context=f".to({dst_r}) from {src_r}")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

class TestFactoryDtypes:
    """Verify factory functions respect dtype argument."""

    @pytest.mark.parametrize("r_dt,t_dt", SHARED_DTYPES, ids=DTYPE_NAMES.values())
    def test_zeros_dtype(self, r_dt, t_dt):
        r = raxis.zeros((3, 4), dtype=r_dt)
        t = torch.zeros((3, 4), dtype=t_dt)
        assert_dtype_eq(r, t, f"zeros({r_dt})")
        assert_values_close(r, t)

    @pytest.mark.parametrize("r_dt,t_dt", SHARED_DTYPES, ids=DTYPE_NAMES.values())
    def test_ones_dtype(self, r_dt, t_dt):
        r = raxis.ones((3, 4), dtype=r_dt)
        t = torch.ones((3, 4), dtype=t_dt)
        assert_dtype_eq(r, t, f"ones({r_dt})")
        assert_values_close(r, t)

    @pytest.mark.parametrize("r_dt,t_dt", SHARED_DTYPES, ids=DTYPE_NAMES.values())
    def test_full_dtype(self, r_dt, t_dt):
        r = raxis.full((3, 4), 7, dtype=r_dt)
        t = torch.full((3, 4), 7, dtype=t_dt)
        assert_dtype_eq(r, t, f"full({r_dt})")
        assert_values_close(r, t)

    def test_zeros_default_dtype(self):
        """zeros() with no dtype should use default float."""
        r = raxis.zeros((2, 3))
        t = torch.zeros((2, 3))
        assert_dtype_eq(r, t, "zeros default")

    def test_ones_default_dtype(self):
        r = raxis.ones((2, 3))
        t = torch.ones((2, 3))
        assert_dtype_eq(r, t, "ones default")

    def test_full_float_value_infers_float(self):
        """full(shape, 3.14) should be float."""
        r = raxis.full((2, 2), 3.14)
        t = torch.full((2, 2), 3.14)
        assert_dtype_eq(r, t, "full(float_val)")

    def test_full_int_value_infers_float(self):
        """In PyTorch, full(shape, 3) is still float by default."""
        r = raxis.full((2, 2), 3)
        t = torch.full((2, 2), 3)
        assert_dtype_eq(r, t, "full(int_val)")


# ============================================================================
# fill_ and zero_ dtype preservation
# ============================================================================

class TestFillPreservesDtype:
    """fill_ and zero_ should not change the tensor's dtype."""

    @pytest.mark.parametrize("r_dt,t_dt", SHARED_DTYPES, ids=DTYPE_NAMES.values())
    def test_fill_preserves_dtype(self, r_dt, t_dt):
        r = raxis.empty((4,), dtype=r_dt)
        r.fill_(42)
        assert r.dtype == r_dt, f"fill_ changed dtype from {r_dt} to {r.dtype}"

    @pytest.mark.parametrize("r_dt,t_dt", SHARED_DTYPES, ids=DTYPE_NAMES.values())
    def test_zero_preserves_dtype(self, r_dt, t_dt):
        r = raxis.ones((4,), dtype=r_dt)
        r.zero_()
        assert r.dtype == r_dt, f"zero_ changed dtype from {r_dt} to {r.dtype}"

@pytest.mark.parametrize("r_dtype,t_dtype", SHARED_DTYPES)
def test_dtype_properties(r_dtype, t_dtype):
    assert r_dtype.is_floating == t_dtype.is_floating_point
    assert r_dtype.is_signed == t_dtype.is_signed
    assert r_dtype.is_integral == (not t_dtype.is_floating_point and t_dtype != torch.bool)
    assert r_dtype.itemsize == t_dtype.itemsize
