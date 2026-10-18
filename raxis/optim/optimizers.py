from typing import Iterable

import raxis
from raxis import Tensor


class Optimizer:

    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("optimizer got an empty parameter list")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.lr = lr
        # Per-parameter state (momentum buffers, moments), created lazily
        self.state: list[dict] = [{} for _ in self.params]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum, as torch.optim.SGD:

        g = grad + weight_decay * p
        b = momentum * b + (1 - dampening) * g     (b = g on the first step)
        g = g + momentum * b if nesterov else b
        p = p - lr * g
    """

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3, momentum: float = 0.0,
                 dampening: float = 0.0, weight_decay: float = 0.0, nesterov: bool = False,
                 maximize: bool = False) -> None:
        super().__init__(params, lr)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.maximize = maximize

    def step(self) -> None:
        with raxis.no_grad():
            for p, state in zip(self.params, self.state):
                if p.grad is None:
                    continue
                grad = -p.grad if self.maximize else p.grad
                if self.weight_decay != 0:
                    grad = grad + p * self.weight_decay

                if self.momentum != 0:
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = state["momentum_buffer"] = grad.clone()
                    else:
                        buf.mul_(self.momentum).add_(grad * (1.0 - self.dampening))
                    grad = grad + buf * self.momentum if self.nesterov else buf

                p.sub_(grad * self.lr)


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2014), as torch.optim.Adam (L2 weight decay)."""

    decoupled_weight_decay = False

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, maximize: bool = False) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.maximize = maximize

    def step(self) -> None:
        with raxis.no_grad():
            for p, state in zip(self.params, self.state):
                if p.grad is None:
                    continue
                grad = -p.grad if self.maximize else p.grad

                if self.weight_decay != 0:
                    if self.decoupled_weight_decay:
                        p.mul_(1.0 - self.lr * self.weight_decay)
                    else:
                        grad = grad + p * self.weight_decay

                if not state:
                    state["step"] = 0
                    state["exp_avg"] = raxis.zeros_like(p)
                    state["exp_avg_sq"] = raxis.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # Running averages of the gradient and its square
                exp_avg.mul_(self.beta1).add_(grad * (1.0 - self.beta1))
                exp_avg_sq.mul_(self.beta2).add_(grad * grad * (1.0 - self.beta2))

                # Bias correction: both averages start at 0
                step_size = self.lr / (1.0 - self.beta1 ** t)
                denom = (exp_avg_sq / (1.0 - self.beta2 ** t)).sqrt() + self.eps
                p.sub_(exp_avg / denom * step_size)


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2017):
    p = p * (1 - lr * weight_decay) instead of adding it to the gradient."""

    decoupled_weight_decay = True

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, maximize: bool = False) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, maximize)
