import os
import subprocess
import sys

import pytest
import raxis
import torch


class TestCpuCapability:

    def test_matches_torch(self):
        expected = torch.backends.cpu.get_cpu_capability()
        # torch also compiles SVE kernels for ARM servers; raxis runs NEON there
        if expected.startswith("SVE"):
            expected = "DEFAULT"
        assert raxis.backends.cpu.get_cpu_capability() == expected

    @pytest.mark.parametrize("capability", ["default", "avx2"])
    def test_environment_lowers_it(self, capability):
        if raxis.backends.cpu.get_cpu_capability() == "DEFAULT":
            pytest.skip("no SIMD dispatch levels on this CPU")
        code = "import raxis; print(raxis.backends.cpu.get_cpu_capability())"
        env = dict(os.environ, RAXIS_CPU_CAPABILITY=capability)
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, check=True)
        assert result.stdout.strip() == capability.upper()
