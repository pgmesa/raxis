import numbers

import raxis
from raxis import Tensor
from raxis.nn import functional as F, init
from raxis.nn.modules import Module, Parameter


# From: torch/nn/modules/normalization.py (LayerNorm)
class LayerNorm(Module):
    """Normalizes each sample over its last len(normalized_shape) dims with
    their mean and (biased) variance, then scales and shifts it elementwise
    with the learnable weight and bias (when elementwise_affine). Unlike
    BatchNorm the statistics never depend on the batch, so training and eval
    compute the same thing.
    """

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True,
                 bias: bool = True) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = Parameter(raxis.empty(self.normalized_shape))
            if bias:
                self.bias = Parameter(raxis.empty(self.normalized_shape))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return (f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}, "
                f"bias={self.bias is not None}")
