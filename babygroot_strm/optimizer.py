"""MuSGD + LARS: Muon-style Newton-Schulz orthogonalization for 2D weights,
LARS trust-ratio scaling for ALL params (2D and 1D), weight decay 2D-only.

Per-param-class treatment:
  - 2D matrices:    upd ← NS(grad + m·buf);  LARS trust;  WD via shrink-then-step.
  - 1D / 4D / etc.: upd ← grad + m·buf;       LARS trust;  no WD.

Rationale:
  - LARS trust = (||w||+ε)/(||upd||+ε) IS NEEDED for 1D params too. Without it,
    tiny-norm biases (e.g. q_norm bias with ||w||=1e-4) get raw-LR × grad
    updates that swamp their value in a single step — the steady-state
    momentum-saturated relative update reaches >20%/step (see
    scripts/diag_lars_updates.py). With trust, the update is normalized to
    ≈ LR per step, same as 2D matrices.
  - NS (Newton-Schulz orthogonalization) is a 2D-matrix operation — skip for
    everything else.
  - WD on biases/norms hurts (standard transformer practice — BERT, GPT, ViT).
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
      trust = (‖w‖ + ε) / (‖upd‖ + ε)        (LARS w/ symmetric eps so trust never collapses
                                              for tiny w; for ‖w‖→0, trust → ε/(‖upd‖+ε) ≠ 0)
      w   ← w·(1 - lr·wd) - lr·trust·upd
    """
    def __init__(self, params, lr=0.01, momentum=0.95, weight_decay=1e-4,
                 nesterov=True, ns_steps=5, trust_eps=0.1, trust_max=2.0, use_lars=True):
        # trust_max bounds the per-step ratio (||W||+ε)/(||u||+ε). Without it,
        # weights grow exponentially: NS normalizes update direction to
        # Frobenius ≈ √min(m,n), so trust = ||W||/√min(m,n) grows linearly with
        # ||W||, giving Δw ∝ ||W|| every step — positive feedback loop.
        # Default trust_max=2.0 = "max relative update ≈ 2·lr per step at
        # ||W||=NS_norm" — matches bitsandbytes' max_unorm=0.02 (lr=0.01).
        # NVIDIA Apex LARC uses ≈1.0 (more conservative). At standard
        # xavier/kaiming init, ||W|| ≈ NS_norm so trust starts ≈ 1.0 and the
        # clamp only kicks in once ||W|| would otherwise grow past NS_norm.
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps,
                        trust_eps=trust_eps, trust_max=trust_max, use_lars=use_lars)
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
            trust_eps = group.get('trust_eps', 0.1)
            trust_max = group.get('trust_max', 1.0)
            use_lars  = group.get('use_lars', True)

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

                # NS orthogonalization: 2D matrices only.
                if p.dim() == 2:
                    update = _newton_schulz(update, steps=ns_steps)

                # LARS trust ratio: applied to ALL params (including 1D — see
                # docstring for why). Symmetric ε so trust is well-defined for
                # any w/upd magnitude (including tiny-norm biases like q_norm).
                # Clamp trust ≤ trust_max to prevent runaway growth (see __init__).
                # `use_lars=False` ⇒ pure MuSGD (NS-orthogonalized SGD + momentum, no LARS).
                if use_lars:
                    w_norm = p.data.norm().item()
                    u_norm = update.norm().item()
                    trust = min(trust_max, (w_norm + trust_eps) / (u_norm + trust_eps))
                else:
                    trust = 1.0

                # Weight decay: 2D matrices only. Skipping WD on 1D biases /
                # LayerNorm gains / embeddings is standard transformer practice.
                if wd > 0 and p.dim() == 2:
                    p.data.mul_(1 - lr * wd)

                p.data.add_(update, alpha=-lr * trust)

        return loss
