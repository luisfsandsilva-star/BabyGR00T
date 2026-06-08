#!/usr/bin/env python3
"""Mechanistic reproduction of the Bayesian-fusion runaway.

Hypothesis: in moment form, when logprec hits the clamp boundary (±5 or ±10),
the exp() amplifier gives the chain-of-fusions Jacobian eigenvalues > 1 in
some directions. Combined with bf16 precision losses, a single bad batch can
push logprec to the boundary, and from there the model rapidly diverges.

We test by:
  1. Initializing logprec deliberately near the boundary (±5)
  2. Applying ONE optimizer step
  3. Measuring how far logprec moved
  4. Iterating: does logprec stay at boundary, escape, or runaway?

Also tests:
  - bf16 vs fp32 (precision-loss contribution)
  - moment vs info form (parameterization contribution)
  - effect of L_inner depth on runaway speed
"""
import math, torch
import torch.nn as nn
import torch.nn.functional as F

_DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ──────────────────────────────────────────────────────────────────────
# Fusion variants
# ──────────────────────────────────────────────────────────────────────

def moment_fuse(z_p, z_g, D, lp_clamp=5.0):
    mu_p, lp_p = z_p[..., :D], z_p[..., D:]
    mu_g, lp_g = z_g[..., :D], z_g[..., D:]
    lp_p = lp_p.clamp(-lp_clamp, lp_clamp)
    lp_g = lp_g.clamp(-lp_clamp, lp_clamp)
    tau_p, tau_g = lp_p.exp(), lp_g.exp()
    tau_post = tau_p + tau_g
    mu_post = (tau_p * mu_p + tau_g * mu_g) / tau_post
    lp_post = tau_post.log().clamp(-lp_clamp, lp_clamp)
    return torch.cat([mu_post, lp_post], dim=-1)


class TinyPolicy(nn.Module):
    """Mimics STRMPolicyVAE structure: g is a small attention-like block, plus
    a head reading mu. Critically uses the SAME inner-loop iteration as real policy."""
    def __init__(self, D=384, K=128, L_inner=5, H_outer=2, heads=8):
        super().__init__()
        self.D = D; self.L = L_inner; self.H = H_outer
        # g: cross-attn style — takes (z + c) as input, attends over a context kv
        self.proj_in = nn.Linear(2*D, 2*D)
        self.attn = nn.MultiheadAttention(2*D, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(2*D, 4*D), nn.SiLU(), nn.Linear(4*D, 2*D))
        self.norm1 = nn.LayerNorm(2*D)
        self.norm2 = nn.LayerNorm(2*D)
        self.head = nn.Linear(D, K)

    def g(self, z, kv):
        x = self.proj_in(z)
        y, _ = self.attn(x, kv, kv)
        x = self.norm1(x + y)
        x = self.norm2(x + self.ff(x))
        return x

    def forward(self, c, kv):
        """c: (B, T, 2D) bias added each step; kv: (B, N_kv, 2D) attention context."""
        z_H = torch.zeros_like(c)
        for h in range(self.H):
            z_L = torch.zeros_like(c)
            for t in range(self.L):
                g_out = self.g(z_L + z_H + c, kv)
                z_L = moment_fuse(z_L, g_out, self.D)
            g_out_H = self.g(z_H + z_L + c, kv)
            z_H = moment_fuse(z_H, g_out_H, self.D)
        mu = z_H[..., :self.D]
        return self.head(mu)


def measure_runaway(precision='fp32', steps=500, lr=2e-3, B=16, D=384, K=128, L=5, H=2,
                    bad_init=False, seed=0, log_every=50):
    """Returns trajectory of (loss, grad_norm, max_logprec_in_state) over `steps`."""
    torch.manual_seed(seed)
    autocast_dtype = torch.bfloat16 if precision == 'bf16' else None

    model = TinyPolicy(D=D, K=K, L_inner=L, H_outer=H).to(_DEV)
    # optionally initialize the head's logprec-output bias near the clamp boundary
    if bad_init:
        # bias the network to output logprec near +5 (clamp boundary) initially
        # add a positive bias to the latter half (logprec channels) of the FF output
        with torch.no_grad():
            model.ff[-1].bias[D:].fill_(4.0)         # logprec channels get +4 bias → exp(4)=55, near boundary

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # synthetic stress task
    g = torch.Generator(device=_DEV).manual_seed(seed)
    c = torch.randn(B, 4, 2*D, generator=g, device=_DEV) * 2.0
    kv = torch.randn(B, 50, 2*D, generator=g, device=_DEV) * 2.0
    target_proj = torch.randn(D, K, generator=g, device=_DEV)
    # targets via deterministic projection
    targets = (c[:, :, :D] @ target_proj).argmax(-1).flatten()

    trajectory = []
    first_nan_step = None
    for step in range(steps):
        opt.zero_grad()
        ctx = torch.amp.autocast('cuda', dtype=autocast_dtype) if (autocast_dtype is not None and _DEV == 'cuda') \
              else torch.amp.autocast('cuda', enabled=False)
        with ctx:
            logits = model(c, kv).flatten(0, 1)
            loss = F.cross_entropy(logits, targets)
        loss.backward()
        gnorm = math.sqrt(sum((p.grad.norm().item() ** 2) for p in model.parameters() if p.grad is not None))
        # peek at the actual z_H logprec the model produces
        with torch.no_grad():
            with ctx:
                z_H = torch.zeros_like(c)
                z_L = torch.zeros_like(c)
                for t in range(L):
                    z_L = moment_fuse(z_L, model.g(z_L + z_H + c, kv), D)
                max_lp = z_L[..., D:].abs().max().item()
        opt.step()
        trajectory.append({
            'step': step, 'loss': loss.item(), 'gnorm': gnorm, 'max_lp': max_lp,
        })
        if not math.isfinite(loss.item()) and first_nan_step is None:
            first_nan_step = step
        if step % log_every == 0 or step in (0, 1, 5, 10, 20):
            print(f"  step {step:>4d}  loss={loss.item():>10.4e}  ||g||={gnorm:>10.2e}  max|lp|={max_lp:>6.2f}")
        if first_nan_step is not None and step > first_nan_step + 5:
            break
    return trajectory, first_nan_step


if __name__ == '__main__':
    print("=" * 60)
    print("(A) fp32 + benign init (control)")
    print("=" * 60)
    traj_a, nan_a = measure_runaway(precision='fp32', bad_init=False, steps=300)
    print(f"  → first NaN @ {nan_a}, final loss={traj_a[-1]['loss']:.3e}")

    print("\n" + "=" * 60)
    print("(B) bf16 + benign init")
    print("=" * 60)
    traj_b, nan_b = measure_runaway(precision='bf16', bad_init=False, steps=300)
    print(f"  → first NaN @ {nan_b}, final loss={traj_b[-1]['loss']:.3e}")

    print("\n" + "=" * 60)
    print("(C) fp32 + BAD init (logprec bias=4, near clamp boundary)")
    print("=" * 60)
    traj_c, nan_c = measure_runaway(precision='fp32', bad_init=True, steps=300)
    print(f"  → first NaN @ {nan_c}, final loss={traj_c[-1]['loss']:.3e}")

    print("\n" + "=" * 60)
    print("(D) bf16 + BAD init  ← the suspected real-world scenario")
    print("=" * 60)
    traj_d, nan_d = measure_runaway(precision='bf16', bad_init=True, steps=300)
    print(f"  → first NaN @ {nan_d}, final loss={traj_d[-1]['loss']:.3e}")

    print("\n" + "=" * 60)
    print("(E) bf16 + BAD init + L_inner=10 (more compounding)")
    print("=" * 60)
    traj_e, nan_e = measure_runaway(precision='bf16', bad_init=True, L=10, steps=300)
    print(f"  → first NaN @ {nan_e}, final loss={traj_e[-1]['loss']:.3e}")

    # Plot loss + grad-norm + max logprec
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for label, traj in [('fp32 benign', traj_a), ('bf16 benign', traj_b),
                        ('fp32 bad init', traj_c), ('bf16 bad init', traj_d),
                        ('bf16 bad init L=10', traj_e)]:
        steps = [r['step'] for r in traj]
        losses = [max(r['loss'], 1e-8) if math.isfinite(r['loss']) else None for r in traj]
        gnorms = [r['gnorm'] for r in traj]
        max_lps = [r['max_lp'] for r in traj]
        axes[0].semilogy(steps, [l if l is not None else float('nan') for l in losses], label=label, linewidth=1.2)
        axes[1].semilogy(steps, gnorms, label=label, linewidth=1.2)
        axes[2].plot(steps, max_lps, label=label, linewidth=1.2)
    for ax, ttl in zip(axes, ['loss (log)', '||grad|| (log)', 'max |logprec|']):
        ax.set_xlabel('step'); ax.set_ylabel(ttl); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.suptitle('Mechanistic Bayesian runaway test: bf16 + bad init triggers explosion', fontsize=12)
    plt.tight_layout()
    plt.savefig('/tmp/bayes_runaway.png', dpi=110, bbox_inches='tight')
    print("\nsaved /tmp/bayes_runaway.png")
