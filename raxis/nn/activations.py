from raxis import Tensor
from raxis.nn import functional as F
from raxis.nn.modules import Module


class ReLU(Module):
    """max(0, x)"""

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class LeakyReLU(Module):
    """max(0, x) + negative_slope * min(0, x)"""

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(x, self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class SELU(Module):
    """scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))"""

    def forward(self, x: Tensor) -> Tensor:
        return F.selu(x)


class Sigmoid(Module):
    """1 / (1 + exp(-x))"""

    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)


class Tanh(Module):

    def forward(self, x: Tensor) -> Tensor:
        return F.tanh(x)


class Softmax(Module):
    """exp(x) / sum(exp(x)) along dim"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class LogSoftmax(Module):
    """log(softmax(x)) along dim, computed with the log-sum-exp trick"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.log_softmax(x, self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
