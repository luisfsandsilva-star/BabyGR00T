#!/usr/bin/env python3
"""Synthetic stress test: does moment-form Bayesian fusion blow up, and does
information form fix it? Mirror the inner+outer iteration structure of the real
policy on a tiny problem so we can iterate fast.

Two fusion variants, identical math (Gaussian product), different parametrization:

  moment form (current STRMPolicyVAE):
    inputs: (mu, log_tau)
    fuse:   tau_post = tau_p + tau_g
            mu_post  = (tau_p·mu_p + tau_g·mu_g) / tau_post
    issues: exp(logprec) amplifier, division in iteration, gradient compounds

  information form (proposed):
    inputs: (eta = tau·mu, lambda = tau)  with lambda = softplus(raw)
    fuse:   eta_post = eta_p + eta_g    (pure addition, Jacobian = I)
            lam_post = lam_p + lam_g    (pure addition)
    convert to (mu, sigma) ONLY at the head.

Mini network mirrors STRMPolicyVAE:
  - g: shared transformer-like Linear that produces evidence each step
  - inner loop L times
  - outer loop H times
  - head: latent_dim → K logits → CE against synthetic targets

We then measure: training stability (NaN onset, loss trajectory, max gradient norm)
on a deliberately stressy synthetic task (codes drawn so that the model must push
mu hard, exercising the precision amplifier).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ──────────────────────────────────────────────────────────────────────
# Fusion variants
# ──────────────────────────────────────────────────────────────────────

def moment_fuse(z_p, z_g, D, lp_clamp=5.0):
    """Current Bayesian fusion in moment form (mu, log tau)."""
    mu_p, lp_p = z_p[..., :D], z_p[..., D:]
    mu_g, lp_g = z_g[..., :D], z_g[..., D:]
    lp_p = lp_p.clamp(-lp_clamp, lp_clamp)
    lp_g = lp_g.clamp(-lp_clamp, lp_clamp)
    tau_p, tau_g = lp_p.exp(), lp_g.exp()
    tau_post = tau_p + tau_g
    mu_post = (tau_p * mu_p + tau_g * mu_g) / tau_post
    lp_post = tau_post.log().clamp(-lp_clamp, lp_clamp)
    return torch.cat([mu_post, lp_post], dim=-1)


def info_fuse(z_p, z_g, D):
    """Information form: (eta, lambda) — pure addition, no exp() in iteration.
    Stored representation: [eta | softplus_inverse(lambda)] so successive calls
    re-apply softplus and get the same lambda back. Equivalent: store [eta | raw_lam]
    where lam = softplus(raw_lam). After fusion lam_post = softplus(raw_p) + softplus(raw_g)
    is positive; we convert back to raw via inverse softplus for storage.
    """
    eta_p, raw_lam_p = z_p[..., :D], z_p[..., D:]
    eta_g, raw_lam_g = z_g[..., :D], z_g[..., D:]
    lam_p = F.softplus(raw_lam_p)
    lam_g = F.softplus(raw_lam_g)
    eta_post = eta_p + eta_g
    lam_post = lam_p + lam_g
    # inverse softplus: x such that softplus(x) = lam_post → x = lam_post + log(1 - exp(-lam_post))
    raw_lam_post = lam_post + torch.log(-torch.expm1(-lam_post.clamp(min=1e-6)))
    return torch.cat([eta_post, raw_lam_post], dim=-1)


def head_from_moment(z, head, D):
    mu = z[..., :D]
    return head(mu)


def head_from_info(z, head, D):
    eta = z[..., :D]
    lam = F.softplus(z[..., D:])
    mu = eta / lam.clamp(min=1e-6)
    return head(mu)


# ──────────────────────────────────────────────────────────────────────
# Tiny end-to-end policy mirror
# ──────────────────────────────────────────────────────────────────────

class TinyBayesPolicy(nn.Module):
    def __init__(self, D=32, K=128, L_inner=5, H_outer=2, fuse_mode='moment'):
        super().__init__()
        self.D = D; self.L = L_inner; self.H = H_outer
        self.fuse_mode = fuse_mode
        # The "g network" produces evidence (mu, logprec) or (eta, raw_lam) from current z + context c
        # 2D input (current z full state) → 2D output (new evidence in same param form)
        self.g = nn.Sequential(
            nn.Linear(2*D, 4*D), nn.SiLU(), nn.Linear(4*D, 2*D))
        self.head = nn.Linear(D, K)

    def fuse(self, z_p, z_g):
        if self.fuse_mode == 'moment':
            return moment_fuse(z_p, z_g, self.D)
        return info_fuse(z_p, z_g, self.D)

    def heads(self, z):
        if self.fuse_mode == 'moment':
            return head_from_moment(z, self.head, self.D)
        return head_from_info(z, self.head, self.D)

    def forward(self, c):
        """c is a context vector (B, 2D); we add it to z at every g call (mimics
        z_L + z_H + y in real policy)."""
        z_H = torch.zeros_like(c)
        for h in range(self.H):
            z_L = torch.zeros_like(c)
            for t in range(self.L):
                g_out = self.g(z_L + z_H + c)
                z_L = self.fuse(z_L, g_out)
            g_out_H = self.g(z_H + z_L + c)
            z_H = self.fuse(z_H, g_out_H)
        return self.heads(z_H)


# ──────────────────────────────────────────────────────────────────────
# Stress task: synthetic codes from a Gaussian context
# ──────────────────────────────────────────────────────────────────────

def make_task(B=64, D=32, K=128, seed=0):
    g = torch.Generator(device=_DEV).manual_seed(seed)
    c = torch.randn(B, 2*D, generator=g, device=_DEV) * 3.0          # contexts (deliberately wide)
    proj = torch.randn(2*D, K, generator=g, device=_DEV)
    targets = (c @ proj).argmax(-1)
    return c, targets


def train_and_measure(mode, steps=1000, lr=3.8e-3, B=64, D=32, K=128, L=5, H=2,
                      seed=0, batches_per_step=4):
    torch.manual_seed(seed)
    model = TinyBayesPolicy(D=D, K=K, L_inner=L, H_outer=H, fuse_mode=mode).to(_DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses, gradnorms = [], []
    first_nan_step = None
    for step in range(steps):
        opt.zero_grad()
        agg_loss = 0
        for _ in range(batches_per_step):
            c, targets = make_task(B=B, D=D, K=K, seed=hash((mode, step, seed, _)) & 0xFFFFFFFF)
            logits = model(c)
            agg_loss = agg_loss + F.cross_entropy(logits, targets)
        loss = agg_loss / batches_per_step
        loss.backward()
        # total gradient norm BEFORE any clipping
        gnorm = math.sqrt(sum((p.grad.norm().item() ** 2) for p in model.parameters() if p.grad is not None))
        opt.step()
        losses.append(loss.item())
        gradnorms.append(gnorm)
        if not math.isfinite(loss.item()) and first_nan_step is None:
            first_nan_step = step
        if first_nan_step is not None and step > first_nan_step + 10:
            break
    return losses, gradnorms, first_nan_step


# ──────────────────────────────────────────────────────────────────────
# Run experiments
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("=== Test 1: moment form (current STRMPolicyVAE) ===")
    losses_m, gnorms_m, nan_m = train_and_measure('moment', steps=1500, seed=0)
    print(f"  first NaN @ step {nan_m}, max loss = {max(l for l in losses_m if math.isfinite(l)):.2e}, "
          f"max grad norm = {max(g for g in gnorms_m if math.isfinite(g)):.2e}")

    print("\n=== Test 2: information form (proposed fix) ===")
    losses_i, gnorms_i, nan_i = train_and_measure('info', steps=1500, seed=0)
    print(f"  first NaN @ step {nan_i}, max loss = {max(l for l in losses_i if math.isfinite(l)):.2e}, "
          f"max grad norm = {max(g for g in gnorms_i if math.isfinite(g)):.2e}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.semilogy(losses_m, 'r-', label=f'moment form (NaN @ {nan_m})', linewidth=1, alpha=0.7)
    ax.semilogy(losses_i, 'b-', label=f'info form (NaN @ {nan_i})', linewidth=1, alpha=0.7)
    ax.set_xlabel('step'); ax.set_ylabel('CE loss (log)'); ax.set_title('Synthetic stress task — loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.semilogy(gnorms_m, 'r-', label='moment form ||grad||', linewidth=1, alpha=0.7)
    ax.semilogy(gnorms_i, 'b-', label='info form ||grad||', linewidth=1, alpha=0.7)
    ax.set_xlabel('step'); ax.set_ylabel('total grad norm (log)'); ax.set_title('Gradient norm over training')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle(f'Bayesian fusion stability — synthetic stress (D=32, K=128, L=5, H=2)', fontsize=12)
    plt.tight_layout()
    plt.savefig('/tmp/bayes_stability.png', dpi=110, bbox_inches='tight')
    print(f"\nplot saved /tmp/bayes_stability.png")
    print(f"max grad ratio (moment / info): {max(gnorms_m)/max(gnorms_i):.1e}x")
