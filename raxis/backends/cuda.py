"""torch.backends.cuda: the TF32 switch of float32 matmuls."""
from raxis import _C


__all__ = ["matmul"]


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
