"""MuSGD + LARS: Muon-style Newton-Schulz orthogonalization for 2D weights,
plain SGD-momentum for 1D (biases, norms), and a per-parameter LARS trust
ratio over both. lr is the only global lever — there is no scalar coefficient
on the LARS trust.
"""
import torch
from torch.optim import Optimizer


def _newton_schulz(G, steps=5, eps=1e-7,
                   abc=(3.4445, -4.7750, 2.0315)):
    """Newton-Schulz iteration that approximates the orthogonal component of G.
    G: 2D tensor. Returns X ≈ U V^T (the nearest orthogonal matrix, up to scale).
    """
    a, b, c = abc
    X = G / (G.norm() + eps)
    if G.shape[0] > G.shape[1]:
        X = X.T
        transposed = True
    else:
        transposed = False
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class MuSGD_LARS(Optimizer):
    """Muon-style NS orth (2D) + SGDM (1D) + LARS trust ratio (all params).

    Update for 2D params:
      buf  = momentum·buf + grad
      upd  = grad + momentum·buf            (Nesterov)
      upd  = NS(upd)                         (orthogonalization)
      trust = ‖w‖ / ‖upd‖                    (LARS, no coefficient)
      w   ← w·(1 - lr·wd) - lr·trust·upd
    """
    def __init__(self, params, lr=0.01, momentum=0.95, weight_decay=1e-4,
                 nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr        = group['lr']
            momentum  = group['momentum']
            wd        = group['weight_decay']
            nesterov  = group['nesterov']
            ns_steps  = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad  = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)

                update = (grad + momentum * buf) if nesterov else buf
                if p.dim() == 2:
                    update = _newton_schulz(update, steps=ns_steps)

                w_norm = p.data.norm()
                u_norm = update.norm()
                trust = (w_norm / u_norm) if (w_norm > 0 and u_norm > 0) else 1.0

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(update, alpha=-lr * trust)

        return loss
