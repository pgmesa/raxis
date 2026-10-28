"""torch.backends.cuda: whether the CUDA backend is there, and the TF32
switch of float32 matmuls."""
import raxis
from raxis import _C


__all__ = ["is_built", "matmul"]


def is_built() -> bool:
    """Whether raxis loaded its CUDA backend (libraxis_cuda, from a source
    build or the raxis-cuda12 / raxis-cuda13 packages). It does not mean
    there is a GPU: that is raxis.cuda.is_available()."""
    return raxis._cuda_library is not None


class _MatmulFlags:
    """torch.backends.cuda.matmul (torch's cuBLASModule). __slots__ makes a
    misspelt flag raise instead of being silently stored."""
    __slots__ = ()

    @property
    def allow_tf32(self) -> bool:
        """Whether float32 matmuls on the GPU (mm, addmm, bmm, matmul, linear)
        may run on the tensor cores in TF32: inputs rounded to a 10-bit
        mantissa (about 1e-3 relative error), products accumulated in float32.

        Off by default, as in torch since 1.12: a float32 matmul gives float32
        precision unless asked otherwise. Same effect as
        raxis.set_float32_matmul_precision("high").
        """
        return _C._get_allow_tf32_matmul()

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        _C._set_allow_tf32_matmul(bool(value))


matmul = _MatmulFlags()
