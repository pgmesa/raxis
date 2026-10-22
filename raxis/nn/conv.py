import math

import raxis
from raxis import Tensor
from raxis.nn import functional as F, init
from raxis.nn.modules import Module, Parameter


def _ntuple(n: int):
    def parse(x) -> tuple:
        return tuple(x) if isinstance(x, (tuple, list)) else (x,) * n
    return parse


_single = _ntuple(1)
_pair = _ntuple(2)


# From: torch/nn/modules/conv.py
class _ConvNd(Module):
    """Base of Conv1d and Conv2d.

    weight: (out_channels, in_channels // groups, *kernel_size), bias:
    (out_channels,). Initialized as in PyTorch: kaiming_uniform_ with
    a = sqrt(5) for the weight, U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for the
    bias. Only zero padding (padding_mode='zeros').
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: tuple,
                 padding: tuple | str, dilation: tuple, groups: int, bias: bool,
                 padding_mode: str) -> None:
        super().__init__()
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if isinstance(padding, str):
            if padding not in ("same", "valid"):
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {{'same', 'valid'}}")
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")
        if padding_mode != "zeros":
            raise NotImplementedError(f"padding_mode={padding_mode!r} is not supported, only 'zeros'")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = Parameter(raxis.empty((out_channels, in_channels // groups, *kernel_size)))
        if bias:
            self.bias = Parameter(raxis.empty((out_channels,)))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.kernel_size):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


class Conv1d(_ConvNd):
    """1-D convolution over (N, C_in, L) or (C_in, L) inputs."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding=0, dilation=1, groups: int = 1, bias: bool = True,
                 padding_mode: str = "zeros") -> None:
        super().__init__(
            in_channels, out_channels, _single(kernel_size), _single(stride),
            padding if isinstance(padding, str) else _single(padding), _single(dilation),
            groups, bias, padding_mode)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv1d(input, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class Conv2d(_ConvNd):
    """2-D convolution over (N, C_in, H, W) or (C_in, H, W) inputs."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1,
                 padding=0, dilation=1, groups: int = 1, bias: bool = True,
                 padding_mode: str = "zeros") -> None:
        super().__init__(
            in_channels, out_channels, _pair(kernel_size), _pair(stride),
            padding if isinstance(padding, str) else _pair(padding), _pair(dilation),
            groups, bias, padding_mode)

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
