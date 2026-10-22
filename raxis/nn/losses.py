from raxis import Tensor
from raxis.nn import functional as F
from raxis.nn.modules import Module


class _Loss(Module):

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def extra_repr(self) -> str:
        return f"reduction={self.reduction!r}" if self.reduction != "mean" else ""


class MSELoss(_Loss):
    """(input - target) ** 2"""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction)


class _WeightedLoss(_Loss):
    """A loss with an optional weight per class, kept as a buffer so that it
    follows the module in state_dict."""

    def __init__(self, weight: Tensor | None = None, reduction: str = "mean") -> None:
        super().__init__(reduction)
        self.register_buffer("weight", weight)


class NLLLoss(_WeightedLoss):
    """-weight[target[i]] * input[i, target[i]]. input holds log-probabilities
    (e.g. from LogSoftmax), target holds class indices (int64), not one-hot
    vectors. With weight, the mean divides by the sum of the target weights."""

    def __init__(self, weight: Tensor | None = None, ignore_index: int = -100,
                 reduction: str = "mean") -> None:
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input, target, self.weight, ignore_index=self.ignore_index,
                          reduction=self.reduction)


class CrossEntropyLoss(_WeightedLoss):
    """NLLLoss(LogSoftmax(dim=1)(input), target): takes raw logits."""

    def __init__(self, weight: Tensor | None = None, ignore_index: int = -100,
                 reduction: str = "mean") -> None:
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(input, target, self.weight, ignore_index=self.ignore_index,
                               reduction=self.reduction)


class BCELoss(_Loss):
    """-(t * log(x) + (1 - t) * log(1 - x)). input holds probabilities."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, reduction=self.reduction)


class BCEWithLogitsLoss(_Loss):
    """BCELoss(sigmoid(input), target), computed in a numerically stable way."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)
