import math

import raxis
from raxis import Tensor
from raxis.nn import functional as F, init
from raxis.nn.modules import Module, Parameter


class Linear(Module):
    """y = x @ weight.T + bias

    weight: (out_features, in_features), bias: (out_features,). Both are
    initialized from U(-1/sqrt(in_features), 1/sqrt(in_features)), as in
    PyTorch.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(raxis.empty((out_features, in_features)))
        if bias:
            self.bias = Parameter(raxis.empty((out_features,)))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # kaiming_uniform_ with a = sqrt(5) reduces to U(-1/sqrt(fan_in), 1/sqrt(fan_in))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# From: torch/nn/modules/sparse.py
class Embedding(Module):
    """A lookup table: row i of weight (num_embeddings, embedding_dim) is
    the vector of index i. Initialized from N(0, 1); the padding_idx row
    starts at zero and gets no gradient."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None,
                 max_norm: float | None = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Tensor | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < num_embeddings, "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert padding_idx >= -num_embeddings, "Padding_idx must be within num_embeddings"
                padding_idx = num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        if _weight is None:
            self.weight = Parameter(raxis.empty((num_embeddings, embedding_dim)))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                "Shape of weight does not match num_embeddings and embedding_dim"
            self.weight = Parameter(_weight)

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        if self.padding_idx is not None:
            with raxis.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return F.embedding(input, self.weight, self.padding_idx, self.max_norm, self.norm_type,
                           self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2:
            s += f", norm_type={self.norm_type}"
        if self.scale_grad_by_freq:
            s += ", scale_grad_by_freq=True"
        if self.sparse:
            s += ", sparse=True"
        return s


class Flatten(Module):
    """Flattens dims [start_dim, end_dim] into one."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Dropout(Module):
    """Zeroes elements with probability p during training and scales the
    rest by 1 / (1 - p). Identity in eval mode."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return F.dropout(x, self.p, self.training)

    def extra_repr(self) -> str:
        return f"p={self.p}"
