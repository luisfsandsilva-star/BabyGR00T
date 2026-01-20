import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion optimizer (Chen et al., 2023).
    Minimal, dependency-free PyTorch implementation with decoupled weight decay.
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0) or not (0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                # Decoupled weight decay
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]

                # First update using sign of momentum with beta1
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                p.add_(m.sign(), alpha=-lr)

                # Update momentum with beta2 for next step
                m.mul_(beta2).add_(g, alpha=1 - beta2)

        return loss


