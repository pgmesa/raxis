from raxis import Tensor
from raxis.nn import functional as F
from raxis.nn.modules import Module


# From: torch/nn/modules/pooling.py
class MaxPool2d(Module):
    """Max over each kernel_size window. stride defaults to kernel_size.
    With return_indices=True the forward also returns the int64 position of
    each max inside its input plane."""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices: bool = False, ceil_mode: bool = False) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor):
        return F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation,
                            ceil_mode=self.ceil_mode, return_indices=self.return_indices)

    def extra_repr(self) -> str:
        return (f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
                f"dilation={self.dilation}, ceil_mode={self.ceil_mode}")


class AvgPool2d(Module):
    """Mean over each kernel_size window. stride defaults to kernel_size."""

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode: bool = False,
                 count_include_pad: bool = True, divisor_override: int | None = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, input: Tensor) -> Tensor:
        return F.avg_pool2d(input, self.kernel_size, self.stride, self.padding, self.ceil_mode,
                            self.count_include_pad, self.divisor_override)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"
