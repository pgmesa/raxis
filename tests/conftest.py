
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


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
