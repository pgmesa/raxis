
import pytest
import raxis
import torch
import numpy as np
import torch.nn.functional as F

from test_utils import (
    check_match, make_pair, make_grad_pair, randf, randf_centered, device
)


def backward_both(r_out, t_out, seed=7):
    """Backward with the same random upstream gradient on both sides."""
    grad = randf_centered(*t_out.shape, seed=seed) if t_out.dim() > 0 else np.float32(1.0)
    grad = np.asarray(grad, dtype=np.float32)
    t_out.backward(torch.from_numpy(grad.copy()))
    r_out.backward(raxis.from_numpy(grad.copy()))


# ===========================================================================
# ACTIVATIONS
# ===========================================================================

class TestRelu:

    @pytest.mark.parametrize("shape", [(5, 10), (3, 4, 5), ()])
    def test_forward_backward(self, shape, device):
        r, t = make_grad_pair(randf_centered(*shape), device)
        r_out, t_out = raxis.relu(r), torch.relu(t)
        check_match(r_out, t_out, context="relu")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, context="relu grad")

    def test_propagates_nan(self, device):
        data = np.array([np.nan, -1.0, 2.0], dtype=np.float32)
        r, t = make_pair(data, device)
        check_match(r.relu(), t.relu(), context="relu nan")

    @pytest.mark.parametrize("dtype", ["int32", "int64"])
    def test_integer(self, dtype, device):
        data = np.array([-3, 0, 5], dtype=dtype)
        r, t = make_pair(data, device)
        check_match(r.relu(), t.relu(), context=f"relu {dtype}")

    def test_inplace(self, device):
        r, t = make_pair(randf_centered(4, 5), device)
        r.relu_()
        t.relu_()
        check_match(r, t, context="relu_")


# ===========================================================================
# SOFTMAX / LOG_SOFTMAX
# ===========================================================================

SOFTMAX_CASES = [
    ((5, 10), 1),
    ((5, 10), 0),
    ((5, 10), -1),
    ((3, 4, 5), 0),
    ((3, 4, 5), 1),
    ((3, 4, 5), 2),
    ((2, 3, 37), -1),     # tail that does not fill a vector
    ((7,), 0),
    ((), 0),
]


class TestSoftmax:

    @pytest.mark.parametrize("op", ["softmax", "log_softmax"])
    @pytest.mark.parametrize("shape, dim", SOFTMAX_CASES,
                             ids=[f"{s}-dim{d}" for s, d in SOFTMAX_CASES])
    def test_forward_backward(self, op, shape, dim, device):
        r, t = make_grad_pair(randf_centered(*shape), device)
        r_out = getattr(raxis, op)(r, dim)
        t_out = getattr(torch, op)(t, dim)
        check_match(r_out, t_out, context=op)
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, atol=1e-5, context=f"{op} grad")

    @pytest.mark.parametrize("op", ["softmax", "log_softmax"])
    def test_large_logits_are_stable(self, op, device):
        data = np.array([[1000.0, 0.0, -1000.0], [88.0, 89.0, 90.0]], dtype=np.float32)
        r, t = make_pair(data, device)
        r_out = getattr(raxis, op)(r, 1)
        assert np.isfinite(np.array(r_out)).all()
        check_match(r_out, getattr(torch, op)(t, 1), context=f"{op} large")

    # A non-uniform incoming gradient (after sum() the softmax gradient is ~0).
    # (2, 64, 3000) on dim 1 splits inner into chunks, the last one partial.
    @pytest.mark.parametrize("op", ["softmax", "log_softmax"])
    @pytest.mark.parametrize("shape, dim", [((64, 10), 1), ((300, 1000), 1), ((4, 7, 3, 9), 2),
                                            ((2, 64, 3000), 1), ((5, 37), 0)])
    def test_backward_random_grad(self, op, shape, dim, device):
        r, t = make_grad_pair(randf_centered(*shape), device)
        w = randf_centered(*shape, seed=7)
        (getattr(raxis, op)(r, dim) * raxis.from_numpy(w)).sum().backward()
        (getattr(torch, op)(t, dim) * torch.from_numpy(w)).sum().backward()
        check_match(r.grad, t.grad, atol=1e-5, context=f"{op} grad")

    def test_float64(self, device):
        r, t = make_pair(randf_centered(4, 6).astype(np.float64), device)
        check_match(r.log_softmax(1), t.log_softmax(1), context="log_softmax f64")

    def test_integer_input_raises(self):
        with pytest.raises(RuntimeError):
            raxis.softmax(raxis.tensor([1, 2, 3]), 0)


# ===========================================================================
# LOSSES
# ===========================================================================

REDUCTIONS = ["mean", "sum", "none"]


def class_targets(n, num_classes, seed=3):
    return np.random.RandomState(seed).randint(0, num_classes, size=n).astype(np.int64)


class TestNllLoss:

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_2d(self, reduction, device):
        r, t = make_grad_pair(randf_centered(8, 5), device)
        r_y, t_y = make_pair(class_targets(8, 5), device)
        r_out = raxis._C._nn.nll_loss(r, r_y, reduction=reduction)
        t_out = F.nll_loss(t, t_y, reduction=reduction)
        check_match(r_out, t_out, context="nll_loss")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, context="nll_loss grad")

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_1d(self, reduction, device):
        r, t = make_grad_pair(randf_centered(5), device)
        r_y, t_y = make_pair(np.array(3, dtype=np.int64), device)
        r_out = raxis._C._nn.nll_loss(r, r_y, reduction=reduction)
        t_out = F.nll_loss(t, t_y, reduction=reduction)
        check_match(r_out, t_out, context="nll_loss 1d")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, context="nll_loss 1d grad")

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_ignore_index(self, reduction, device):
        target = class_targets(8, 5)
        target[[1, 4]] = -100
        r, t = make_grad_pair(randf_centered(8, 5), device)
        r_y, t_y = make_pair(target, device)
        r_out = raxis._C._nn.nll_loss(r, r_y, reduction=reduction)
        t_out = F.nll_loss(t, t_y, reduction=reduction)
        check_match(r_out, t_out, context="nll_loss ignore")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, context="nll_loss ignore grad")

    def test_out_of_range_target_raises(self):
        x = raxis.zeros((2, 3))
        with pytest.raises(RuntimeError):
            raxis._C._nn.nll_loss(x, raxis.tensor([0, 3]))

    def test_non_int64_target_raises(self):
        x = raxis.zeros((2, 3))
        with pytest.raises(RuntimeError):
            raxis._C._nn.nll_loss(x, raxis.tensor([0.0, 1.0]))


class TestCompositeLosses:

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_cross_entropy(self, reduction, device):
        r, t = make_grad_pair(randf_centered(8, 5), device)
        r_y, t_y = make_pair(class_targets(8, 5), device)
        r_out = raxis._C._nn.cross_entropy(r, r_y, reduction=reduction)
        t_out = F.cross_entropy(t, t_y, reduction=reduction)
        check_match(r_out, t_out, context="cross_entropy")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, context="cross_entropy grad")

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_mse(self, reduction, device):
        r, t = make_grad_pair(randf_centered(6, 4), device)
        r_y, t_y = make_pair(randf_centered(6, 4, seed=9), device)
        r_out = raxis._C._nn.mse_loss(r, r_y, reduction=reduction)
        t_out = F.mse_loss(t, t_y, reduction=reduction)
        check_match(r_out, t_out, context="mse")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, context="mse grad")

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_binary_cross_entropy(self, reduction, device):
        probs = randf(6, 4, lo=0.05, hi=0.95)
        labels = (randf(6, 4, seed=5) > 0.5).astype(np.float32)
        r, t = make_grad_pair(probs, device)
        r_y, t_y = make_pair(labels, device)
        r_out = raxis._C._nn.binary_cross_entropy(r, r_y, reduction=reduction)
        t_out = F.binary_cross_entropy(t, t_y, reduction=reduction)
        check_match(r_out, t_out, context="bce")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, atol=1e-4, context="bce grad")

    def test_binary_cross_entropy_saturated(self, device):
        r, t = make_pair(np.array([0.0, 1.0], dtype=np.float32), device)
        r_y, t_y = make_pair(np.array([1.0, 0.0], dtype=np.float32), device)
        check_match(raxis._C._nn.binary_cross_entropy(r, r_y),
                    F.binary_cross_entropy(t, t_y), context="bce clamp")

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    def test_binary_cross_entropy_with_logits(self, reduction, device):
        logits = randf_centered(6, 4) * 20  # includes large magnitudes
        labels = (randf(6, 4, seed=5) > 0.5).astype(np.float32)
        r, t = make_grad_pair(logits, device)
        r_y, t_y = make_pair(labels, device)
        r_out = raxis._C._nn.binary_cross_entropy_with_logits(r, r_y, reduction=reduction)
        t_out = F.binary_cross_entropy_with_logits(t, t_y, reduction=reduction)
        check_match(r_out, t_out, atol=1e-4, context="bce logits")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, atol=1e-5, context="bce logits grad")

    def test_invalid_reduction_raises(self):
        x = raxis.zeros((2, 3))
        with pytest.raises(RuntimeError):
            raxis._C._nn.mse_loss(x, x, reduction="avg")


# ===========================================================================
# LINEAR / FLATTEN / DROPOUT
# ===========================================================================

class TestLinear:

    @pytest.mark.parametrize("x_shape", [(8, 6), (2, 8, 6), (6,)])
    @pytest.mark.parametrize("bias", [True, False])
    def test_forward_backward(self, x_shape, bias, device):
        r_x, t_x = make_grad_pair(randf_centered(*x_shape), device)
        r_w, t_w = make_grad_pair(randf_centered(4, 6, seed=2), device)
        r_b, t_b = make_grad_pair(randf_centered(4, seed=3), device) if bias else (None, None)
        r_out = raxis._C._nn.linear(r_x, r_w, r_b)
        t_out = F.linear(t_x, t_w, t_b)
        check_match(r_out, t_out, atol=1e-4, context="linear")
        backward_both(r_out, t_out)
        check_match(r_x.grad, t_x.grad, atol=1e-4, context="linear x grad")
        check_match(r_w.grad, t_w.grad, atol=1e-4, context="linear w grad")
        if bias:
            check_match(r_b.grad, t_b.grad, atol=1e-4, context="linear b grad")


class TestFlatten:

    @pytest.mark.parametrize("shape, start, end", [
        ((2, 3, 4), 0, -1),
        ((2, 3, 4), 1, -1),
        ((2, 3, 4, 5), 1, 2),
        ((2, 3), 1, 1),
        ((), 0, -1),
    ])
    def test_forward_backward(self, shape, start, end, device):
        r, t = make_grad_pair(randf_centered(*shape), device)
        r_out = r.flatten(start, end)
        t_out = t.flatten(start, end)
        check_match(r_out, t_out, context="flatten")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, context="flatten grad")

    def test_start_after_end_raises(self):
        with pytest.raises(RuntimeError):
            raxis.zeros((2, 3, 4)).flatten(2, 1)


class TestDropout:

    def test_eval_and_zero_p_are_identity(self):
        x = raxis.rand((4, 5))
        check_match(raxis.dropout(x, 0.5, train=False), torch.from_numpy(np.array(x)))
        check_match(raxis.dropout(x, 0.0, train=True), torch.from_numpy(np.array(x)))

    def test_p_one_zeroes_everything(self):
        out = raxis.dropout(raxis.rand((4, 5)), 1.0, train=True)
        assert (np.array(out) == 0).all()

    def test_kept_values_are_scaled_and_grad_matches_mask(self):
        raxis.manual_seed(0)
        p = 0.3
        x = raxis.ones((200, 100), requires_grad=True)
        out = raxis.dropout(x, p, train=True)
        values = np.array(out.detach())
        kept = values != 0
        np.testing.assert_allclose(values[kept], 1.0 / (1.0 - p), rtol=1e-6)
        assert abs(kept.mean() - (1.0 - p)) < 0.02

        out.sum().backward()
        np.testing.assert_allclose(np.array(x.grad), values, rtol=1e-6)

    def test_invalid_p_raises(self):
        with pytest.raises(RuntimeError):
            raxis.dropout(raxis.rand((2, 2)), 1.5, train=True)


# ===========================================================================
# TENSOR CREATION FROM NUMPY
# ===========================================================================

class TestTensorFromNumpy:

    @pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "int32", "uint8", "bool"])
    def test_keeps_dtype_and_values(self, dtype):
        data = (randf(3, 4) * 10).astype(dtype)
        r = raxis.tensor(data)
        assert str(r.dtype) == str(torch.tensor(data).dtype).replace("torch.", "")
        np.testing.assert_array_equal(np.array(r), data)

    def test_copies(self):
        data = np.zeros(3, dtype=np.float32)
        r = raxis.tensor(data)
        data[0] = 5.0
        assert np.array(r)[0] == 0.0

    def test_dtype_conversion(self):
        r = raxis.tensor(np.ones((2, 2)), dtype=raxis.float32)
        assert r.dtype == raxis.float32

    def test_non_contiguous(self):
        data = randf(4, 6)[:, ::2].T
        np.testing.assert_array_equal(np.array(raxis.tensor(data)), data)


class TestIteration:

    def test_iterates_over_dim0(self):
        data = randf(3, 4)
        rows = list(raxis.tensor(data))
        assert len(rows) == 3
        for row, expected in zip(rows, data):
            np.testing.assert_array_equal(np.array(row), expected)

    def test_rows_keep_the_graph(self):
        x = raxis.rand((3, 2), requires_grad=True)
        sum(row.sum() for row in x).backward()
        np.testing.assert_array_equal(np.array(x.grad), np.ones((3, 2), dtype=np.float32))

    def test_zero_dim_raises(self):
        # TypeError, as torch
        with pytest.raises(TypeError):
            iter(raxis.tensor(1.0))
