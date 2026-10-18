
import pytest
import raxis
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------

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
    return np.asarray(rng.rand(*shape) * (hi - lo) + lo).astype(np.float32)

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


def make_grad_pair(data, device="cpu"):
    """Create a pair of tensors with requires_grad=True. Skip if not supported."""
    r, t = make_pair(data, device)
    t = t.requires_grad_(True)
    try:
        r = r.requires_grad_(True)
    except (RuntimeError, AttributeError):
        pytest.skip("requires_grad not supported")
    return r, t