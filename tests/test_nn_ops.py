
import os
import subprocess
import sys

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


class TestGelu:

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    @pytest.mark.parametrize("shape", [(5, 10), (3, 4, 5), (), (200, 300)])
    def test_forward_backward(self, shape, approximate, device):
        r, t = make_grad_pair(np.asarray(randf_centered(*shape) * 4), device)
        r_out = raxis.nn.functional.gelu(r, approximate=approximate)
        t_out = F.gelu(t, approximate=approximate)
        # torch computes the erf version with oneDNN (1-2 ulp apart), and
        # where tanh saturates its vector tanh and std::tanh differ by ~1e-6;
        # which elements take the vector path depends on the thread split
        check_match(r_out, t_out, atol=1e-5, context=f"gelu {approximate}")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, atol=1e-5, context=f"gelu {approximate} grad")
        assert r_out.grad_fn.name() == "GeluBackward0"

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_float64(self, approximate, device):
        x = randf_centered(50, 7).astype(np.float64) * 4
        r, t = make_grad_pair(x, device)
        r_out = raxis.nn.functional.gelu(r, approximate=approximate)
        t_out = F.gelu(t, approximate=approximate)
        check_match(r_out, t_out, atol=1e-14)
        r_out.sum().backward()
        t_out.sum().backward()
        check_match(r.grad, t.grad, atol=1e-14)

    @pytest.mark.parametrize("approximate", ["none", "tanh"])
    def test_double_backward(self, approximate, device):
        """gelu_backward is recorded with create_graph: its derivatives with
        respect to the incoming gradient and to x match torch's."""
        results = []
        for lib in (raxis, torch):
            x = lib.tensor(randf_centered(4, 5) * 3, requires_grad=True)
            g = lib.tensor(randf_centered(4, 5, seed=3), requires_grad=True)
            out = lib.nn.functional.gelu(x, approximate=approximate)
            gx, = lib.autograd.grad(out, x, g, create_graph=True)
            results.append(lib.autograd.grad((gx * gx).sum(), (x, g)))
        for r, t in zip(*results):
            check_match(r, t, atol=1e-5)

    def test_errors(self, device):
        with pytest.raises(RuntimeError, match="approximate argument must be either none or tanh"):
            raxis.nn.functional.gelu(raxis.ones(3), approximate="fast")
        with pytest.raises(RuntimeError, match="not implemented for"):
            raxis.nn.functional.gelu(raxis.ones(3, dtype=raxis.int64))


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

    @pytest.mark.parametrize("reduction", REDUCTIONS)
    @pytest.mark.parametrize("shape", [(8, 5), (5,)])
    def test_weight(self, shape, reduction, device):
        target = class_targets(shape[0], 5) if len(shape) == 2 else np.array(3, dtype=np.int64)
        if len(shape) == 2:
            target[[1, 4]] = -100
        weight = np.array([0.5, 2.0, 1.0, 3.0, 0.25], dtype=np.float32)
        r, t = make_grad_pair(randf_centered(*shape), device)
        r_y, t_y = make_pair(target, device)
        r_w, t_w = make_pair(weight, device)
        r_out = raxis._C._nn.nll_loss(r, r_y, r_w, reduction=reduction)
        t_out = F.nll_loss(t, t_y, t_w, reduction=reduction)
        check_match(r_out, t_out, context="nll_loss weight")
        backward_both(r_out, t_out)
        check_match(r.grad, t.grad, context="nll_loss weight grad")

    @pytest.mark.parametrize("weighted", [False, True])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_mean_is_bitwise_torch(self, dtype, weighted):
        # Large batch: the cascade summation (8 levels of partial sums in
        # scalar_t) reproduces torch's rounding exactly
        n, c = 100_003, 10
        x = (np.random.default_rng(0).standard_normal((n, c)) * 3).astype(dtype)
        target = class_targets(n, c)
        weight = np.random.default_rng(1).random(c).astype(dtype) if weighted else None
        r_out = raxis._C._nn.nll_loss(raxis.from_numpy(x), raxis.from_numpy(target),
                                      raxis.from_numpy(weight) if weighted else None)
        t_out = F.nll_loss(torch.from_numpy(x), torch.from_numpy(target),
                           torch.from_numpy(weight) if weighted else None)
        np.testing.assert_array_equal(np.array(r_out), t_out.numpy())

    def test_bad_weight_raises(self):
        x = raxis.zeros((2, 3))
        y = raxis.tensor([0, 1])
        with pytest.raises(RuntimeError, match="weight tensor should be defined"):
            raxis._C._nn.nll_loss(x, y, raxis.ones((4,)))
        with pytest.raises(RuntimeError):
            raxis._C._nn.nll_loss(x, y, raxis.ones((3,), dtype=raxis.float64))

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

    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    @pytest.mark.parametrize("dtype", [raxis.float32, raxis.float64])
    def test_keep_rate_and_scale(self, p, dtype):
        raxis.manual_seed(1)
        x = raxis.ones((300, 1001), dtype=dtype)  # not a multiple of the 64-element batches
        values = np.array(raxis.dropout(x, p, train=True))
        kept = values != 0
        # 300k Bernoulli draws: the standard deviation of the rate is < 0.001
        assert abs(kept.mean() - (1.0 - p)) < 0.005
        np.testing.assert_allclose(values[kept], 1.0 / (1.0 - p), rtol=1e-6)

    def test_grad_fn_and_backward(self):
        x = raxis.randn((64, 50), requires_grad=True)
        out = raxis.dropout(x, 0.4, train=True)
        assert out.grad_fn.name() == "NativeDropoutBackward0"
        mask = np.array(out.grad_fn._saved_result1)
        assert mask.dtype == np.bool_
        grad = randf_centered(64, 50)
        out.backward(raxis.from_numpy(grad))
        np.testing.assert_allclose(np.array(x.grad), grad * mask / 0.6, rtol=1e-6)
        np.testing.assert_allclose(np.array(out.detach()), np.array(x.detach()) * mask / 0.6, rtol=1e-6)

    def test_seeded_and_independent_draws(self):
        x = raxis.ones((100, 100))
        raxis.manual_seed(7)
        first = np.array(raxis.dropout(x, 0.5, train=True))
        second = np.array(raxis.dropout(x, 0.5, train=True))
        raxis.manual_seed(7)
        np.testing.assert_array_equal(np.array(raxis.dropout(x, 0.5, train=True)), first)
        # Each call reserves new Philox counters, so masks do not repeat
        assert (first != second).mean() > 0.4

    def test_mask_independent_of_thread_count(self):
        x = raxis.ones((513, 999))
        masks = []
        threads = raxis.get_num_threads()
        try:
            for n in (1, threads):
                raxis.set_num_threads(n)
                raxis.manual_seed(3)
                masks.append(np.array(raxis.dropout(x, 0.3, train=True)))
        finally:
            raxis.set_num_threads(threads)
        np.testing.assert_array_equal(masks[0], masks[1])

    def test_mask_independent_of_cpu_capability(self):
        # The SIMD and scalar Philox paths must draw the same stream
        code = ("import raxis, numpy as np; raxis.manual_seed(5); "
                "print(np.array(raxis.dropout(raxis.ones((97, 131)), 0.3, True)).sum())")
        sums = set()
        for capability in ("default", "avx2", None):  # None: the best one the CPU has
            env = {k: v for k, v in os.environ.items() if k != "RAXIS_CPU_CAPABILITY"}
            if capability is not None:
                env["RAXIS_CPU_CAPABILITY"] = capability
            result = subprocess.run([sys.executable, "-c", code], env=env,
                                    capture_output=True, text=True, check=True)
            sums.add(result.stdout.strip())
        assert len(sums) == 1


class TestPhilox:

    # Known-answer vectors of Philox4x32-10 from Random123 (kat_vectors)
    @pytest.mark.parametrize("counter, key, expected", [
        ((0, 0, 0, 0), 0, (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
        ((0xffffffff,) * 4, 0xffffffffffffffff, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
        ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), 0x299f31d0a4093822,
         (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)),
    ])
    def test_known_answers(self, counter, key, expected):
        assert tuple(raxis._C._philox4x32_10(list(counter), key)) == expected


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
# EMBEDDING
# ===========================================================================

# (num_embeddings, embedding_dim, indices shape, padding_idx, scale_grad_by_freq):
# repeated indices, a negative padding_idx, more rows than one thread's grain
EMBEDDING_CASES = [
    (65, 32, (12, 64), None, False), (10, 7, (5,), 3, False), (10, 7, (4, 6), -1, True),
    (5000, 16, (3, 1000), None, False), (5000, 64, (4, 100), 0, True), (4, 3, (0,), None, False),
]


class TestEmbedding:

    @pytest.mark.parametrize("num, dim, shape, padding_idx, freq", EMBEDDING_CASES)
    def test_forward_backward(self, num, dim, shape, padding_idx, freq):
        rng = np.random.RandomState(0)
        r_w, t_w = make_grad_pair(rng.randn(num, dim).astype(np.float32))
        idx = rng.randint(0, num, size=shape)
        r_out = raxis.nn.functional.embedding(raxis.tensor(idx), r_w, padding_idx, scale_grad_by_freq=freq)
        t_out = F.embedding(torch.tensor(idx), t_w, padding_idx, scale_grad_by_freq=freq)
        check_match(r_out, t_out, atol=0)
        backward_both(r_out, t_out)
        # Each row is summed by one thread in index order: same order as torch
        check_match(r_w.grad, t_w.grad, atol=0)
        assert r_out.grad_fn.name() == "EmbeddingBackward0"

    def test_int32_indices_and_float64(self):
        w = randf_centered(6, 4).astype(np.float64)
        idx = np.array([[0, 5, 5], [2, 0, 1]], dtype=np.int32)
        r_w, t_w = make_grad_pair(w)
        r_out = raxis.nn.functional.embedding(raxis.tensor(idx), r_w)
        t_out = F.embedding(torch.tensor(idx), t_w)
        check_match(r_out, t_out)
        backward_both(r_out, t_out)
        check_match(r_w.grad, t_w.grad)

    def test_double_backward(self):
        results = []
        for lib in (raxis, torch):
            w = lib.tensor(randf_centered(6, 3), requires_grad=True)
            idx = lib.tensor([[1, 2, 2], [0, 5, 2]])
            out = lib.nn.functional.embedding(idx, w, padding_idx=2)
            g = lib.tensor(randf_centered(2, 3, 3, seed=2), requires_grad=True)
            gw, = lib.autograd.grad(out, w, g, create_graph=True)
            gg, = lib.autograd.grad((gw * lib.tensor(randf_centered(6, 3, seed=3))).sum(), g)
            results.append(gg)
        check_match(results[0], results[1])

    def test_errors(self):
        w = raxis.randn(5, 3)
        with pytest.raises(IndexError, match="index out of range in self"):
            raxis.nn.functional.embedding(raxis.tensor([1, 5]), w)
        with pytest.raises(RuntimeError, match="'indices' to have one of the following scalar types: Long, Int"):
            raxis.nn.functional.embedding(raxis.tensor([1.0]), w)
        with pytest.raises(RuntimeError, match="'weight' must be 2-D"):
            raxis.nn.functional.embedding(raxis.tensor([1]), raxis.randn(5))
        with pytest.raises(AssertionError, match="Padding_idx must be within num_embeddings"):
            raxis.nn.functional.embedding(raxis.tensor([1]), w, padding_idx=5)


# ===========================================================================
# SCALED DOT PRODUCT ATTENTION
# ===========================================================================

def sdpa_both(q_shape, kv_len=None, seed=0, dtype=np.float32, heads_last=False, **kwargs):
    """Output and q/k/v gradients of raxis and torch for the same inputs;
    numpy arrays in kwargs become tensors of each library. heads_last
    stores q, k and v as [B, L, H, E] and passes the [B, H, L, E] view, as
    a transformer does after splitting the projection into heads."""
    rng = np.random.RandomState(seed)
    *batch, L, E = q_shape
    S = kv_len or L
    q = rng.randn(*q_shape).astype(dtype)
    k, v = (rng.randn(*batch, S, E).astype(dtype) for _ in range(2))
    grad = rng.randn(*q_shape).astype(dtype)
    results = []
    for lib in (raxis, torch):
        if heads_last:
            leaves = [lib.tensor(np.ascontiguousarray(a.swapaxes(1, 2)), requires_grad=True) for a in (q, k, v)]
            tensors = [t.transpose(1, 2) for t in leaves]
        else:
            leaves = tensors = [lib.tensor(a, requires_grad=True) for a in (q, k, v)]
        kw = {name: lib.tensor(val) if isinstance(val, np.ndarray) else val for name, val in kwargs.items()}
        out = lib.nn.functional.scaled_dot_product_attention(*tensors, **kw)
        out.backward(lib.tensor(grad))
        results.append([out] + [t.grad for t in leaves])
    return results


class TestScaledDotProductAttention:

    @pytest.mark.parametrize("q_shape, kv_len", [((2, 4, 16, 8), None), ((3, 7, 5), 9), ((1, 2, 64, 32), None)])
    @pytest.mark.parametrize("is_causal", [False, True])
    def test_forward_backward(self, q_shape, kv_len, is_causal):
        # 4-D inputs run the flash kernel in both libraries, 3-D the math path
        r, t = sdpa_both(q_shape, kv_len, is_causal=is_causal)
        assert r[0].grad_fn.name() == t[0].grad_fn.name()
        for r_val, t_val in zip(r, t):
            check_match(r_val, t_val, atol=1e-5)

    @pytest.mark.parametrize("q_shape, kv_len, is_causal", [
        ((1, 2, 700, 16), None, True),    # query blocks of 64, two key blocks
        ((1, 2, 800, 8), 1100, False),    # query blocks of 256, three key blocks
        ((2, 3, 20, 4), 7, True),         # more queries than keys
    ])
    def test_flash_blocks(self, q_shape, kv_len, is_causal):
        for r, t in zip(*sdpa_both(q_shape, kv_len, is_causal=is_causal)):
            check_match(r, t, atol=1e-5)

    def test_flash_layout_and_double(self):
        r, t = sdpa_both((2, 4, 64, 32), heads_last=True, is_causal=True)
        # The output keeps the [B, L, H, E] storage of the inputs, so the
        # transpose + view back to [B, L, H * E] is free (torch does the same)
        assert r[0].strides == t[0].stride()
        for r_val, t_val in zip(r, t):
            check_match(r_val, t_val, atol=1e-5)
        for r_val, t_val in zip(*sdpa_both((2, 3, 17, 6), 23, dtype=np.float64, scale=0.3, is_causal=True)):
            check_match(r_val, t_val, atol=1e-12)

    def test_math_fallback(self):
        # A mask that requires grad keeps the math path, which differentiates it
        T = 6
        rng = np.random.RandomState(2)
        q, mask = rng.randn(1, 2, T, 4).astype(np.float32), rng.randn(T, T).astype(np.float32)
        grads = []
        for lib in (raxis, torch):
            x = lib.tensor(q, requires_grad=True)
            m = lib.tensor(mask, requires_grad=True)
            out = lib.nn.functional.scaled_dot_product_attention(x, x, x, attn_mask=m)
            assert out.grad_fn.name() != "ScaledDotProductFlashAttentionForCpuBackward0"
            out.sum().backward()
            grads.append((x.grad, m.grad))
        for r_val, t_val in zip(*grads):
            check_match(r_val, t_val, atol=1e-5)

    def test_masks_and_scale(self):
        T = 10
        rng = np.random.RandomState(1)
        cases = [
            dict(scale=0.3), dict(scale=-0.5),
            # The first row is fully masked: zeros, not NaN (_safe_softmax)
            dict(attn_mask=np.tril(np.ones((T, T), dtype=bool), -1)),
            dict(attn_mask=rng.randn(T, T).astype(np.float32)),
            dict(attn_mask=rng.rand(2, 1, T, T) > 0.3),
            dict(attn_mask=rng.randn(1, 3, 1, T).astype(np.float32)),
        ]
        for kwargs in cases:
            for r, t in zip(*sdpa_both((2, 3, T, 4), **kwargs)):
                check_match(r, t, atol=1e-5)

    def test_dropout(self):
        q = raxis.randn(2, 2, 8, 4, requires_grad=True)
        F_ = raxis.nn.functional
        out = F_.scaled_dot_product_attention(q, q, q, dropout_p=0.5, is_causal=True)
        out.sum().backward()
        assert q.grad is not None
        # p=1 drops every weight
        zero = F_.scaled_dot_product_attention(q, q, q, dropout_p=1.0)
        assert np.all(np.array(zero.detach()) == 0)

    def test_errors(self):
        q = raxis.randn(1, 4, 3)
        with pytest.raises(RuntimeError, match="Explicit attn_mask should not be set when is_causal=True"):
            raxis.nn.functional.scaled_dot_product_attention(q, q, q, raxis.ones(4, 4), is_causal=True)


# ===========================================================================
# LAYER NORM
# ===========================================================================

# (input shape, normalized_shape): rows shorter and longer than a vector, a
# row long enough for the cascade of Welford chunks, several normalized dims
LAYER_NORM_CASES = [
    ((4, 7), (7,)), ((3, 5, 37), (37,)), ((2, 3, 4, 5), (4, 5)), ((2, 3000), (3000,)), ((6, 1), (1,)),
]


def layer_norm_both(shape, normalized_shape, dtype=np.float32, affine=True, seed=0):
    rng = np.random.RandomState(seed)
    x = (rng.randn(*shape) * 3 + 2).astype(dtype)
    r_x, t_x = make_grad_pair(x)
    r_w, t_w = make_grad_pair(rng.randn(*normalized_shape).astype(dtype)) if affine else (None, None)
    r_b, t_b = make_grad_pair(rng.randn(*normalized_shape).astype(dtype)) if affine else (None, None)
    r_out = raxis.nn.functional.layer_norm(r_x, normalized_shape, r_w, r_b)
    t_out = F.layer_norm(t_x, normalized_shape, t_w, t_b)
    grad = rng.randn(*shape).astype(dtype)
    t_out.backward(torch.from_numpy(grad.copy()))
    r_out.backward(raxis.from_numpy(grad.copy()))
    pairs = {"output": (r_out, t_out), "input grad": (r_x.grad, t_x.grad)}
    if affine:
        pairs["weight grad"] = (r_w.grad, t_w.grad)
        pairs["bias grad"] = (r_b.grad, t_b.grad)
    return pairs


class TestLayerNorm:

    @pytest.mark.parametrize("shape, normalized_shape", LAYER_NORM_CASES)
    @pytest.mark.parametrize("affine", [True, False])
    def test_forward_backward(self, shape, normalized_shape, affine):
        for name, (r, t) in layer_norm_both(shape, normalized_shape, affine=affine).items():
            check_match(r, t, atol=1e-4, context=f"layer_norm {shape} {name}")

    def test_float64(self):
        for name, (r, t) in layer_norm_both((3, 5, 37), (5, 37), dtype=np.float64).items():
            check_match(r, t, atol=1e-10, context=f"layer_norm float64 {name}")

    def test_weight_without_bias(self):
        x = randf_centered(4, 6)
        w = randf_centered(6, seed=2)
        r_out = raxis.nn.functional.layer_norm(raxis.tensor(x), (6,), raxis.tensor(w))
        check_match(r_out, F.layer_norm(torch.tensor(x), (6,), torch.tensor(w)), atol=1e-5)

    def test_empty_batch(self):
        w = raxis.ones(4, requires_grad=True)
        x = raxis.zeros(0, 4, requires_grad=True)
        out = raxis.nn.functional.layer_norm(x, (4,), w)
        assert tuple(out.shape) == (0, 4)
        out.sum().backward()
        check_match(w.grad, torch.zeros(4))

    def test_grad_fn(self):
        x = raxis.randn(2, 3, 4, requires_grad=True)
        out = raxis.nn.functional.layer_norm(x, (3, 4))
        assert out.grad_fn.name() == "NativeLayerNormBackward0"
        # mean and rstd keep the leading dims, as in torch
        assert tuple(out.grad_fn._saved_result1.shape) == (2, 1, 1)
        assert tuple(out.grad_fn._saved_result2.shape) == (2, 1, 1)

    def test_errors(self):
        x = raxis.randn(2, 3, 4)
        ln = raxis.nn.functional.layer_norm
        with pytest.raises(RuntimeError, match=r"Given normalized_shape=\[3\], expected input with shape"):
            ln(x, (3,))
        with pytest.raises(RuntimeError, match="Expected weight to be of same shape as normalized_shape"):
            ln(x, (4,), raxis.ones(3))
        with pytest.raises(RuntimeError, match="Expected normalized_shape to be at least 1-dimensional"):
            ln(x, ())


# ===========================================================================
# PADDING
# ===========================================================================

PAD_MODES = ["constant", "reflect", "replicate", "circular"]

# (input shape, pad): 1, 2 and 3 padded dims with and without a batch dim,
# negative pads (crop), a row long enough for the vectorized interior copy
PAD_CASES = [
    ((2, 3, 7), (2, 1)), ((3, 7), (0, 3)), ((2, 3, 5, 6), (1, 2, 3, 1)), ((3, 5, 6), (2, 2, 1, 0)),
    ((1, 2, 3, 4, 5), (1, 1, 2, 0, 1, 2)), ((2, 3, 4, 5), (1, 1, 2, 0, 1, 2)), ((2, 3, 9), (-2, 3)),
    ((2, 3, 5, 6), (-1, 2, 2, -2)), ((0, 3, 5), (1, 1)), ((2, 3, 40), (5, 7)),
]


class TestPad:

    @pytest.mark.parametrize("mode", PAD_MODES)
    @pytest.mark.parametrize("shape, pad", PAD_CASES)
    def test_forward_backward(self, shape, pad, mode):
        r_x, t_x = make_grad_pair(randf_centered(*shape))
        r_out = raxis.nn.functional.pad(r_x, pad, mode=mode)
        t_out = F.pad(t_x, pad, mode=mode)
        check_match(r_out, t_out, context=f"pad {mode} {shape} {pad}")
        backward_both(r_out, t_out)
        check_match(r_x.grad, t_x.grad, atol=1e-5, context=f"pad {mode} {shape} {pad} grad")

    @pytest.mark.parametrize("mode", PAD_MODES)
    def test_double_backward(self, mode):
        """The derivative of the pad backward with respect to the incoming
        gradient pads it again."""
        x = randf_centered(1, 2, 4, 5)
        w = randf_centered(1, 2, 4, 5, seed=4)
        results = []
        for lib in (raxis, torch):
            lx = lib.tensor(x, requires_grad=True)
            out = lib.nn.functional.pad(lx, (2, 1, 1, 2), mode=mode)
            g = lib.tensor(randf_centered(*out.shape, seed=5), requires_grad=True)
            gx, = lib.autograd.grad(out, lx, g, create_graph=True)
            gg, = lib.autograd.grad(gx, g, lib.tensor(w))
            results.append(gg)
        check_match(results[0], results[1], atol=1e-5)

    @pytest.mark.parametrize("dtype", [np.float64, np.int64])
    def test_dtypes(self, dtype):
        x = (randf_centered(2, 3, 6) * 10).astype(dtype)
        for mode in PAD_MODES:
            check_match(raxis.nn.functional.pad(raxis.tensor(x), (2, 3), mode=mode),
                        F.pad(torch.tensor(x), (2, 3), mode=mode), context=mode)

    def test_constant_value(self):
        x = randf_centered(2, 3)
        check_match(raxis.nn.functional.pad(raxis.tensor(x), (1, 2, 0, 1), value=-3.5),
                    F.pad(torch.tensor(x), (1, 2, 0, 1), value=-3.5))
        check_match(raxis.constant_pad_nd(raxis.tensor(x), [1, 1], 2),
                    torch.constant_pad_nd(torch.tensor(x), [1, 1], 2))

    def test_grad_fn(self):
        x = raxis.randn(1, 2, 4, 4, requires_grad=True)
        pad = raxis.nn.functional.pad
        assert pad(x, (1, 1)).grad_fn.name() == "ConstantPadNdBackward0"
        assert pad(x[0], (1, 1), mode="reflect").grad_fn.name() == "ReflectionPad1DBackward0"
        assert pad(x, (1, 1, 1, 1), mode="replicate").grad_fn.name() == "ReplicationPad2DBackward0"

    def test_errors(self):
        x = raxis.randn(2, 3, 4)
        pad = raxis.nn.functional.pad
        with pytest.raises(RuntimeError, match="Padding size should be less than the corresponding input"):
            pad(x, (4, 0), mode="reflect")
        with pytest.raises(NotImplementedError, match="Padding size 4 is not supported for 2D input"):
            pad(raxis.randn(2, 3), (1, 1, 1, 1), mode="reflect")
        with pytest.raises(NotImplementedError, match="Unrecognised padding mode"):
            pad(x, (1, 1), mode="mirror")
        with pytest.raises(RuntimeError, match="doesn't take in value argument"):
            pad(x, (1, 1), mode="replicate", value=1.0)
        with pytest.raises(RuntimeError, match="Padding value causes wrapping around more than once"):
            pad(x, (5, 0), mode="circular")
        with pytest.raises(RuntimeError, match="Padding length must be divisible by 2"):
            pad(x, (1,))
        with pytest.raises(RuntimeError, match="length must be non-negative"):
            pad(x, (-3, -2))


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
