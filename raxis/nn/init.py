"""Parameter initialization, mirroring torch.nn.init.

Every function fills the tensor in place under no_grad and returns it.
Reference: https://pytorch.org/docs/stable/nn.init.html
"""

import math

import raxis
from raxis import Tensor


def calculate_gain(nonlinearity: str, param: float | None = None) -> float:
    """Recommended gain for a nonlinearity:

    linear, conv, sigmoid: 1, tanh: 5/3, relu: sqrt(2),
    leaky_relu: sqrt(2 / (1 + negative_slope^2)), selu: 3/4
    """
    if nonlinearity in ("linear", "conv1d", "conv2d", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3
    if nonlinearity == "relu":
        return math.sqrt(2.0)
    if nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif isinstance(param, (int, float)) and not isinstance(param, bool):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    if nonlinearity == "selu":
        # Found empirically (https://github.com/pytorch/pytorch/pull/50664)
        return 3.0 / 4
    raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple[int, int]:
    if tensor.dim() < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    receptive_field_size = math.prod(tensor.shape[2:])
    fan_in = tensor.shape[1] * receptive_field_size
    fan_out = tensor.shape[0] * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"Mode {mode} not supported, please use one of fan_in, fan_out")
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fills with values from U(a, b)."""
    with raxis.no_grad():
        return tensor.uniform_(a, b)


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fills with values from N(mean, std^2)."""
    with raxis.no_grad():
        return tensor.normal_(mean, std)


def constant_(tensor: Tensor, val: float) -> Tensor:
    with raxis.no_grad():
        return tensor.fill_(val)


def ones_(tensor: Tensor) -> Tensor:
    return constant_(tensor, 1.0)


def zeros_(tensor: Tensor) -> Tensor:
    with raxis.no_grad():
        return tensor.zero_()


def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Glorot & Bengio (2010): U(-a, a) with a = gain * sqrt(6 / (fan_in + fan_out))."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return uniform_(tensor, -a, a)


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Glorot & Bengio (2010): N(0, std^2) with std = gain * sqrt(2 / (fan_in + fan_out))."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(tensor: Tensor, a: float = 0, mode: str = "fan_in",
                     nonlinearity: str = "leaky_relu") -> Tensor:
    """He et al. (2015): U(-bound, bound) with bound = gain * sqrt(3 / fan)."""
    fan = _calculate_correct_fan(tensor, mode)
    bound = calculate_gain(nonlinearity, a) * math.sqrt(3.0 / fan)
    return uniform_(tensor, -bound, bound)


def kaiming_normal_(tensor: Tensor, a: float = 0, mode: str = "fan_in",
                    nonlinearity: str = "leaky_relu") -> Tensor:
    """He et al. (2015): N(0, std^2) with std = gain / sqrt(fan)."""
    fan = _calculate_correct_fan(tensor, mode)
    std = calculate_gain(nonlinearity, a) / math.sqrt(fan)
    return normal_(tensor, 0.0, std)
