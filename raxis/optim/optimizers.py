from collections import defaultdict
from typing import Any, Iterable

import raxis
from raxis import Tensor


class Optimizer:
    """Base class, with PyTorch's layout:

    - ``param_groups``: list of dicts, each with its ``params`` and its own
      hyperparameters (``lr``, ...), filled in from ``defaults``. Changing
      ``param_groups[i]["lr"]`` changes the learning rate of that group.
    - ``state``: per-parameter state (momentum buffers, moments), keyed by the
      parameter and created lazily on the first step.

    ``params`` is an iterable of tensors, or of dicts to give each group its
    own hyperparameters: ``[{"params": body.parameters()}, {"params": head.parameters(), "lr": 1e-2}]``.
    """

    def __init__(self, params: Iterable[Tensor] | Iterable[dict], defaults: dict[str, Any]) -> None:
        self.defaults = defaults
        self.state: defaultdict[Tensor, dict] = defaultdict(dict)
        self.param_groups: list[dict[str, Any]] = []

        param_groups = list(params)
        if not param_groups:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
        for group in param_groups:
            self.add_param_group(group)

    # From: torch/optim/optimizer.py (Optimizer.add_param_group)
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        params = param_group["params"]
        param_group["params"] = [params] if isinstance(params, Tensor) else list(params)
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        seen = {id(p) for group in self.param_groups for p in group["params"]}
        if any(id(p) in seen for p in param_group["params"]):
            raise ValueError("some parameters appear in more than one parameter group")
        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for group in self.param_groups:
            for p in group["params"]:
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

    def __init__(self, params: Iterable[Tensor] | Iterable[dict], lr: float = 1e-3,
                 momentum: float = 0.0, dampening: float = 0.0, weight_decay: float = 0.0,
                 nesterov: bool = False, maximize: bool = False) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        super().__init__(params, defaults)

    def step(self) -> None:
        with raxis.no_grad():
            for group in self.param_groups:
                lr, momentum = group["lr"], group["momentum"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = -p.grad if group["maximize"] else p.grad
                    if group["weight_decay"] != 0:
                        grad = grad + p * group["weight_decay"]

                    if momentum != 0:
                        state = self.state[p]
                        buf = state.get("momentum_buffer")
                        if buf is None:
                            buf = state["momentum_buffer"] = grad.clone()
                        else:
                            buf.mul_(momentum).add_(grad * (1.0 - group["dampening"]))
                        grad = grad + buf * momentum if group["nesterov"] else buf

                    p.sub_(grad * lr)


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2014), as torch.optim.Adam (L2 weight decay)."""

    decoupled_weight_decay = False

    def __init__(self, params: Iterable[Tensor] | Iterable[dict], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, maximize: bool = False) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, maximize=maximize)
        super().__init__(params, defaults)

    def step(self) -> None:
        with raxis.no_grad():
            for group in self.param_groups:
                lr, eps, weight_decay = group["lr"], group["eps"], group["weight_decay"]
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = -p.grad if group["maximize"] else p.grad

                    if weight_decay != 0:
                        if self.decoupled_weight_decay:
                            p.mul_(1.0 - lr * weight_decay)
                        else:
                            grad = grad + p * weight_decay

                    state = self.state[p]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = raxis.zeros_like(p)
                        state["exp_avg_sq"] = raxis.zeros_like(p)
                    state["step"] += 1
                    t = state["step"]
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                    # Running averages of the gradient and its square
                    exp_avg.mul_(beta1).add_(grad * (1.0 - beta1))
                    exp_avg_sq.mul_(beta2).add_(grad * grad * (1.0 - beta2))

                    # Bias correction: both averages start at 0
                    step_size = lr / (1.0 - beta1 ** t)
                    denom = (exp_avg_sq / (1.0 - beta2 ** t)).sqrt() + eps
                    p.sub_(exp_avg / denom * step_size)


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2017):
    p = p * (1 - lr * weight_decay) instead of adding it to the gradient."""

    decoupled_weight_decay = True

    def __init__(self, params: Iterable[Tensor] | Iterable[dict], lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2, maximize: bool = False) -> None:
        super().__init__(params, lr, betas, eps, weight_decay, maximize)
