import raxis
from raxis import Tensor
from raxis.nn import functional as F, init
from raxis.nn.modules import Module, Parameter


# From: torch/nn/modules/batchnorm.py (_NormBase, _BatchNorm)
class _BatchNorm(Module):
    """Normalizes each channel (dim 1) with the batch mean and variance in
    training and with running estimates of them in eval, then scales and
    shifts it with the learnable weight and bias (when affine).

    The running estimates are buffers updated in training as
    running = (1 - momentum) * running + momentum * batch_stat, the variance
    with its unbiased estimate. momentum=None uses the cumulative average
    over num_batches_tracked instead.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float | None = 0.1,
                 affine: bool = True, track_running_stats: bool = True, *, bias: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if affine:
            self.weight = Parameter(raxis.empty((num_features,)))
            if bias:
                self.bias = Parameter(raxis.empty((num_features,)))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if track_running_stats:
            self.register_buffer("running_mean", raxis.zeros((num_features,)))
            self.register_buffer("running_var", raxis.ones((num_features,)))
            self.register_buffer("num_batches_tracked", raxis.tensor(0, dtype=raxis.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def _check_input_dim(self, input: Tensor) -> None:
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        if self.training and self.track_running_stats:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()

        # Batch statistics in training, and in eval when there are no running ones
        bn_training = self.training or (self.running_mean is None and self.running_var is None)
        # The running buffers are passed only to be read (eval) or updated
        # (training with track_running_stats)
        use_buffers = not self.training or self.track_running_stats
        return F.batch_norm(
            input,
            self.running_mean if use_buffers else None,
            self.running_var if use_buffers else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

    def extra_repr(self) -> str:
        return (f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
                f"affine={self.affine}, bias={self.bias is not None}, "
                f"track_running_stats={self.track_running_stats}")


class BatchNorm1d(_BatchNorm):
    """BatchNorm over (N, C) or (N, C, L) inputs."""

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(f"expected 2D or 3D input (got {input.dim()}D input)")


class BatchNorm2d(_BatchNorm):
    """BatchNorm over (N, C, H, W) inputs."""

    def _check_input_dim(self, input: Tensor) -> None:
        if input.dim() != 4:
            raise ValueError(f"expected 4D input (got {input.dim()}D input)")
