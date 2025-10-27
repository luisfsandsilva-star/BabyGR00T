from typing import Iterable, Optional, Callable

import torch
from torch import Tensor
from torch.optim import Optimizer


class Lion(Optimizer):
    r"""Implements the Lion optimizer.

    Lion is described in the paper `Symbolic Discovery of Optimization Algorithms`
    (https://arxiv.org/abs/2302.06675).

    This implementation follows the reference optimizer from the lucidrains
    `lion-pytorch` package while avoiding the external dependency so the project
    can run in offline environments.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        """Performs a single optimization step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay: float = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                update = grad
                if weight_decay != 0:
                    update = update.add(p, alpha=weight_decay)

                exp_avg.mul_(beta2).add_(update, alpha=1 - beta2)

                p.add_(update.sign(), alpha=-lr * beta1)
                p.add_(exp_avg.sign(), alpha=-lr * (1 - beta1))

        return loss
