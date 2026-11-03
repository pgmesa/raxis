"""torch.backends.cpu: which SIMD instruction set the CPU kernels run with."""
from raxis import _C


__all__ = ["get_cpu_capability"]


def get_cpu_capability() -> str:
    """The instruction set raxis picked for its CPU kernels at startup, as
    torch.backends.cpu.get_cpu_capability(): "AVX512", "AVX2" or "DEFAULT"
    (the baseline the wheel was compiled for: SSE2 on x86-64, NEON on ARM64).
    The RAXIS_CPU_CAPABILITY environment variable ("avx512", "avx2",
    "default") lowers it, as ATEN_CPU_CAPABILITY does for torch."""
    return _C.get_cpu_capability()
