
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import raxis  # noqa: E402
import torch  # noqa: E402

# float32 convolutions on a GPU default to TF32 in both libraries (cuDNN in
# torch), which rounds the inputs to a 10-bit mantissa: the comparisons
# against torch need full float32. tests/test_cuda.py checks the TF32 paths.
torch.backends.cudnn.allow_tf32 = False
raxis.backends.cudnn.allow_tf32 = False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A cuda-parametrized test that hits an op without a CUDA kernel is an
    expected failure until the kernel exists (it passes on its own then).
    Tests that expect NotImplementedError catch it with pytest.raises, so it
    never reaches here."""
    outcome = yield
    report = outcome.get_result()
    callspec = getattr(item, "callspec", None)
    if (call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError)
            and callspec is not None and callspec.params.get("device") == "cuda"):
        report.outcome = "skipped"
        report.wasxfail = f"no CUDA kernel yet: {call.excinfo.value}"
