"""torch.backends.cudnn: the TF32 switch of float32 convolutions.

raxis has no cuDNN: its GPU convolutions are im2col + cuBLAS. The flag
keeps torch's name and meaning (it governs convolutions, not matmuls), so
`torch.backends.cudnn.allow_tf32 = ...` in torch code does the same here.
"""
import sys
import types

from raxis import _C


class _CudnnModule(types.ModuleType):
    # A module cannot have properties; its class can (torch replaces the
    # module object for the same reason)

    @property
    def allow_tf32(self) -> bool:
        """Whether the gemms of float32 convolutions on the GPU may run in
        TF32 (see raxis.backends.cuda.matmul.allow_tf32). On by default, as
        in torch: the rounding is far below the noise of training."""
        return _C._get_allow_tf32_conv()

    @allow_tf32.setter
    def allow_tf32(self, value: bool) -> None:
        _C._set_allow_tf32_conv(bool(value))


sys.modules[__name__].__class__ = _CudnnModule
