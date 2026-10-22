
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
# CONVOLUTION
# ===========================================================================

CONV2D_CASES = [
    # x_shape, w_shape, kwargs
    ((2, 3, 7, 9), (4, 3, 3, 3), {}),
    ((2, 3, 7, 9), (4, 3, 3, 2), dict(stride=(2, 3), padding=(1, 2), dilation=(2, 1))),
    ((2, 3, 8, 8), (5, 3, 3, 3), dict(stride=2, padding=1)),
    ((3, 4, 8, 8), (6, 2, 3, 3), dict(groups=2, padding=1)),
    ((2, 6, 6, 6), (6, 1, 3, 3), dict(groups=6, padding=1)),
    ((2, 4, 5, 5), (8, 4, 1, 1), {}),
    ((2, 4, 5, 5), (8, 4, 1, 1), dict(stride=2)),
    ((2, 4, 5, 5), (8, 4, 1, 1), dict(padding=1)),
    ((1, 2, 3, 3), (2, 2, 5, 5), dict(padding=1)),
    ((2, 2, 9, 9), (3, 2, 3, 3), dict(dilation=3, stride=2, padding=2)),
    # More images than threads: the batch is split in parallel chunks
    ((17, 3, 6, 6), (4, 3, 3, 3), dict(padding=1)),
]


def conv_both(x_shape, w_shape, bias, dtype, conv_r, conv_t, **kwargs):
    """Runs the convolution and its backward on both sides; returns the
    (raxis, torch) pairs of output, input grad, weight grad and bias grad."""
    x = randf_centered(*x_shape, seed=1).astype(dtype)
    w = randf_centered(*w_shape, seed=2).astype(dtype)
    r_x, t_x = make_grad_pair(x)
    r_w, t_w = make_grad_pair(w)
    r_b, t_b = make_grad_pair(randf_centered(w_shape[0], seed=3).astype(dtype)) if bias else (None, None)
    r_out = conv_r(r_x, r_w, r_b, **kwargs)
    t_out = conv_t(t_x, t_w, t_b, **kwargs)
    grad = randf_centered(*t_out.shape, seed=4).astype(dtype)
    t_out.backward(torch.from_numpy(grad.copy()))
    r_out.backward(raxis.from_numpy(grad.copy()))
    pairs = [(r_out, t_out), (r_x.grad, t_x.grad), (r_w.grad, t_w.grad)]
    if bias:
        pairs.append((r_b.grad, t_b.grad))
    return pairs


class TestConvolution:

    @pytest.mark.parametrize("x_shape, w_shape, kwargs", CONV2D_CASES)
    @pytest.mark.parametrize("bias", [True, False])
    def test_conv2d(self, x_shape, w_shape, kwargs, bias):
        pairs = conv_both(x_shape, w_shape, bias, np.float64, raxis.conv2d, F.conv2d, **kwargs)
        for (r, t), name in zip(pairs, ["output", "input grad", "weight grad", "bias grad"]):
            check_match(r, t, atol=1e-10, rtol=1e-10, context=f"conv2d {name}")

    @pytest.mark.parametrize("x_shape, w_shape, kwargs", [
        ((20, 3, 12, 12), (8, 3, 5, 5), dict(padding=2)),
        ((4, 16, 10, 10), (16, 4, 3, 3), dict(groups=4, stride=2)),
    ])
    def test_conv2d_float32(self, x_shape, w_shape, kwargs):
        pairs = conv_both(x_shape, w_shape, True, np.float32, raxis.conv2d, F.conv2d, **kwargs)
        for (r, t), name in zip(pairs, ["output", "input grad", "weight grad", "bias grad"]):
            check_match(r, t, atol=1e-4, context=f"conv2d float32 {name}")

    @pytest.mark.parametrize("x_shape, w_shape, kwargs", [
        ((2, 3, 10), (5, 3, 4), dict(padding=2, stride=2)),
        ((2, 4, 9), (4, 2, 3), dict(groups=2, dilation=2)),
        ((3, 2, 7), (3, 2, 1), {}),
    ])
    def test_conv1d(self, x_shape, w_shape, kwargs):
        pairs = conv_both(x_shape, w_shape, True, np.float64, raxis.conv1d, F.conv1d, **kwargs)
        for (r, t), name in zip(pairs, ["output", "input grad", "weight grad", "bias grad"]):
            check_match(r, t, atol=1e-10, rtol=1e-10, context=f"conv1d {name}")

    def test_unbatched(self):
        for x_shape, w_shape, conv_r, conv_t in [
            ((3, 9, 9), (2, 3, 3, 3), raxis.conv2d, F.conv2d),
            ((3, 9), (2, 3, 3), raxis.conv1d, F.conv1d),
        ]:
            pairs = conv_both(x_shape, w_shape, True, np.float64, conv_r, conv_t)
            for r, t in pairs:
                check_match(r, t, atol=1e-10, rtol=1e-10, context=f"unbatched {x_shape}")

    def test_empty_batch(self):
        pairs = conv_both((0, 2, 5, 5), (3, 2, 3, 3), True, np.float64, raxis.conv2d, F.conv2d)
        for r, t in pairs:
            check_match(r, t, context="empty batch")

    def test_only_weight_requires_grad(self):
        # First layer of a network: the input does not require grad
        x = randf_centered(2, 3, 6, 6)
        r_w, t_w = make_grad_pair(randf_centered(4, 3, 3, 3, seed=2))
        r_out = raxis.conv2d(raxis.from_numpy(x), r_w)
        t_out = F.conv2d(torch.from_numpy(x), t_w)
        backward_both(r_out, t_out)
        check_match(r_w.grad, t_w.grad, atol=1e-4, context="weight grad")

    def test_grad_fn(self):
        x = raxis.randn(1, 2, 5, 5, requires_grad=True)
        w = raxis.randn(3, 2, 3, 3, requires_grad=True)
        out = raxis.conv2d(x, w)
        assert out.grad_fn.name() == "ConvolutionBackward0"
        assert out.grad_fn._saved_input._cdata == x._cdata
        assert out.grad_fn._saved_weight._cdata == w._cdata

    def test_errors(self):
        x = raxis.randn(1, 3, 5, 5)
        with pytest.raises(RuntimeError, match="to have 6 channels, but got 3 channels"):
            raxis.conv2d(x, raxis.randn(4, 6, 3, 3))
        with pytest.raises(RuntimeError, match="Kernel size can't be greater than actual input size"):
            raxis.conv2d(x, raxis.randn(4, 3, 7, 7))
        with pytest.raises(RuntimeError, match="expected weight to be divisible by 2"):
            raxis.conv2d(raxis.randn(1, 4, 5, 5), raxis.randn(3, 2, 3, 3), groups=2)
        with pytest.raises(RuntimeError, match="expected bias to be 1-dimensional with 4 elements"):
            raxis.conv2d(x, raxis.randn(4, 3, 3, 3), raxis.randn(3))
        with pytest.raises(RuntimeError, match="non-positive stride"):
            raxis.conv2d(x, raxis.randn(4, 3, 3, 3), stride=0)
        with pytest.raises(RuntimeError, match="should be the same"):
            raxis.conv2d(x, raxis.randn(4, 3, 3, 3, dtype=raxis.float64))
        with pytest.raises(RuntimeError, match=r"Expected 3D \(unbatched\) or 4D \(batched\) input to conv2d"):
            raxis.conv2d(raxis.randn(5, 5), raxis.randn(4, 3, 3, 3))


# ===========================================================================
# POOLING
# ===========================================================================

MAX_POOL_CASES = [
    # x_shape, kwargs
    ((2, 3, 8, 8), dict(kernel_size=2)),
    ((2, 3, 9, 7), dict(kernel_size=3, stride=2)),
    ((2, 3, 9, 7), dict(kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))),
    ((2, 2, 10, 10), dict(kernel_size=3, stride=2, padding=1, dilation=2)),
    ((1, 2, 7, 7), dict(kernel_size=2, ceil_mode=True)),
    ((1, 2, 8, 8), dict(kernel_size=3, stride=2, padding=1, ceil_mode=True)),
    ((3, 6, 6), dict(kernel_size=2)),
    ((2, 1, 5, 5), dict(kernel_size=5)),
]

AVG_POOL_CASES = [
    ((2, 3, 8, 8), dict(kernel_size=2)),
    ((2, 3, 9, 7), dict(kernel_size=3, stride=2)),
    ((2, 3, 9, 7), dict(kernel_size=(3, 2), stride=(2, 1), padding=(1, 1))),
    ((2, 3, 9, 7), dict(kernel_size=3, padding=1, count_include_pad=False)),
    ((1, 2, 7, 7), dict(kernel_size=2, ceil_mode=True)),
    ((1, 2, 8, 8), dict(kernel_size=3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)),
    ((1, 2, 6, 6), dict(kernel_size=3, padding=1, divisor_override=5)),
    ((3, 6, 6), dict(kernel_size=2)),
]


def pool_both(x, r_fn, t_fn, **kwargs):
    r_x, t_x = make_grad_pair(x)
    r_out, t_out = r_fn(r_x, **kwargs), t_fn(t_x, **kwargs)
    backward_both(r_out, t_out)
    return (r_out, t_out), (r_x.grad, t_x.grad)


class TestMaxPool2d:

    @pytest.mark.parametrize("x_shape, kwargs", MAX_POOL_CASES)
    def test_forward_backward(self, x_shape, kwargs):
        out, grad = pool_both(randf_centered(*x_shape), raxis.max_pool2d, F.max_pool2d, **kwargs)
        check_match(*out, atol=0, context="max_pool2d")
        check_match(*grad, atol=0, context="max_pool2d grad")

    def test_overlapping_windows_accumulate(self):
        # stride < kernel: an element can be the max of several windows
        x = np.zeros((1, 1, 5, 5), dtype=np.float32)
        x[0, 0, 2, 2] = 1.0
        out, grad = pool_both(x, raxis.max_pool2d, F.max_pool2d, kernel_size=3, stride=1)
        check_match(*grad, atol=0, context="overlapping grad")

    def test_return_indices(self):
        r, t = make_pair(randf_centered(2, 3, 9, 7))
        r_out, r_idx = raxis.max_pool2d_with_indices(r, 3, stride=2, padding=1)
        t_out, t_idx = F.max_pool2d(t, 3, stride=2, padding=1, return_indices=True)
        check_match(r_out, t_out, atol=0)
        check_match(r_idx, t_idx, atol=0)
        assert r_idx.dtype == raxis.int64

    def test_nan_propagates(self):
        x = randf_centered(1, 1, 4, 4)
        x[0, 0, 1, 2] = np.nan
        r, t = make_pair(x)
        check_match(raxis.max_pool2d(r, 2), F.max_pool2d(t, 2), atol=0)

    @pytest.mark.parametrize("dtype", ["int32", "int64", "float64"])
    def test_dtypes(self, dtype):
        data = (randf_centered(2, 2, 6, 6) * 100).astype(dtype)
        r, t = make_pair(data)
        check_match(raxis.max_pool2d(r, 3, stride=2), F.max_pool2d(t, 3, stride=2), atol=0)

    def test_empty_batch(self):
        r, t = make_pair(np.zeros((0, 2, 4, 4), dtype=np.float32))
        check_match(raxis.max_pool2d(r, 2), F.max_pool2d(t, 2))

    def test_grad_fn(self):
        x = raxis.randn(1, 2, 4, 4, requires_grad=True)
        out = raxis.max_pool2d(x, 2)
        assert out.grad_fn.name() == "MaxPool2DWithIndicesBackward0"
        assert out.grad_fn._saved_result1.dtype == raxis.int64

    def test_errors(self):
        x = raxis.randn(1, 2, 5, 5)
        with pytest.raises(RuntimeError, match="pad should be at most half of effective kernel size"):
            raxis.max_pool2d(x, 2, padding=2)
        with pytest.raises(RuntimeError, match="Output size is too small"):
            raxis.max_pool2d(x, 7)
        with pytest.raises(RuntimeError, match="kernel_size must either be a single int"):
            raxis.max_pool2d(x, (2, 2, 2))
        with pytest.raises(RuntimeError, match="non-empty 3D or 4D"):
            raxis.max_pool2d(raxis.randn(5, 5), 2)
        with pytest.raises(RuntimeError, match="optional 0 dim batch size"):
            raxis.max_pool2d(raxis.randn(1, 0, 5, 5), 2)


class TestAvgPool2d:

    @pytest.mark.parametrize("x_shape, kwargs", AVG_POOL_CASES)
    def test_forward_backward(self, x_shape, kwargs):
        out, grad = pool_both(randf_centered(*x_shape), raxis.avg_pool2d, F.avg_pool2d, **kwargs)
        check_match(*out, atol=1e-6, context="avg_pool2d")
        check_match(*grad, atol=1e-6, context="avg_pool2d grad")

    def test_grad_fn(self):
        x = raxis.randn(1, 2, 4, 4, requires_grad=True)
        assert raxis.avg_pool2d(x, 2).grad_fn.name() == "AvgPool2DBackward0"

    def test_errors(self):
        x = raxis.randn(1, 2, 5, 5)
        with pytest.raises(RuntimeError, match="divisor must be not zero"):
            raxis.avg_pool2d(x, 2, divisor_override=0)
        with pytest.raises(RuntimeError, match="pad should be at most half"):
            raxis.avg_pool2d(x, 2, padding=2)


# ===========================================================================
# BATCH NORM
# ===========================================================================

# 2-D inputs with more rows than threads take the [N, C] rows path
BATCH_NORM_SHAPES = [(8, 4, 5, 5), (6, 3, 7), (64, 5), (3, 5), (2, 3, 1, 1)]


def batch_norm_both(shape, training, affine=True, running=True, seed=0):
    rng = np.random.RandomState(seed)
    channels = shape[1]
    x = (rng.randn(*shape) * 3 + 1).astype(np.float32)
    r_x, t_x = make_grad_pair(x)
    r_w, t_w = make_grad_pair(rng.randn(channels).astype(np.float32)) if affine else (None, None)
    r_b, t_b = make_grad_pair(rng.randn(channels).astype(np.float32)) if affine else (None, None)
    if running:
        r_rm, t_rm = make_pair(rng.randn(channels).astype(np.float32))
        r_rv, t_rv = make_pair((rng.rand(channels) + 0.5).astype(np.float32))
    else:
        r_rm = t_rm = r_rv = t_rv = None
    r_out = raxis.nn.functional.batch_norm(r_x, r_rm, r_rv, r_w, r_b, training, 0.1, 1e-5)
    t_out = F.batch_norm(t_x, t_rm, t_rv, t_w, t_b, training, 0.1, 1e-5)
    backward_both(r_out, t_out)
    pairs = {"output": (r_out, t_out), "input grad": (r_x.grad, t_x.grad)}
    if affine:
        pairs["weight grad"] = (r_w.grad, t_w.grad)
        pairs["bias grad"] = (r_b.grad, t_b.grad)
    if running:
        pairs["running_mean"] = (r_rm, t_rm)
        pairs["running_var"] = (r_rv, t_rv)
    return pairs


class TestBatchNorm:

    @pytest.mark.parametrize("shape", BATCH_NORM_SHAPES)
    @pytest.mark.parametrize("training", [True, False])
    def test_forward_backward(self, shape, training):
        for name, (r, t) in batch_norm_both(shape, training).items():
            check_match(r, t, atol=1e-5, context=f"batch_norm {shape} training={training} {name}")

    @pytest.mark.parametrize("shape", [(8, 4, 5, 5), (64, 5)])
    def test_no_affine_no_running_stats(self, shape):
        for name, (r, t) in batch_norm_both(shape, True, affine=False, running=False).items():
            check_match(r, t, atol=1e-5, context=f"batch_norm {name}")

    def test_float64(self):
        x = randf_centered(4, 3, 6, 6).astype(np.float64)
        r_x, t_x = make_grad_pair(x)
        r_out = raxis.nn.functional.batch_norm(r_x, None, None, training=True)
        t_out = F.batch_norm(t_x, None, None, training=True)
        check_match(r_out, t_out, atol=1e-12)
        grad = x + 0.5
        t_out.backward(torch.from_numpy(grad))
        r_out.backward(raxis.from_numpy(grad))
        check_match(r_x.grad, t_x.grad, atol=1e-12)

    def test_grad_fn(self):
        x = raxis.randn(4, 3, requires_grad=True)
        out = raxis.nn.functional.batch_norm(x, raxis.zeros(3), raxis.ones(3), training=True)
        assert out.grad_fn.name() == "NativeBatchNormBackward0"
        assert tuple(out.grad_fn._saved_result1.shape) == (3,)  # save_mean
        assert tuple(out.grad_fn._saved_result2.shape) == (3,)  # save_invstd

    def test_errors(self):
        x = raxis.randn(4, 3, 5)
        bn = raxis.nn.functional.batch_norm
        with pytest.raises(RuntimeError, match="running_mean should contain 3 elements not 4"):
            bn(x, raxis.zeros(4), raxis.ones(4))
        with pytest.raises(RuntimeError, match="must either both be None or neither be None"):
            bn(x, raxis.zeros(3), None, training=True)
        with pytest.raises(ValueError, match="Expected more than 1 value per channel when training"):
            bn(raxis.randn(1, 3), None, None, training=True)
        with pytest.raises(ValueError, match="eps must be positive during training"):
            bn(x, None, None, training=True, eps=0.0)


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
