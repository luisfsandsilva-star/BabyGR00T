#!/usr/bin/env python3
"""Compare regularizer stability between KL and various closed-form Gaussian
distribution distances. For diagonal posterior N(μ, σ²I) vs prior N(0, I):

  KL                : ½ (σ² + μ² - 1 - log σ²)            — log σ² singular at σ→0
  W2 (Wasserstein²) : ||μ||² + Σ (σ - 1)²                 — no log; cleanest
  Frobenius         : ||μ||² + ||σ² - 1||²                — variance² difference
  MMD (RBF, closed) : k_pp + k_qq - 2 k_pq    (per below) — bounded but underflows in high-d

Closed-form MMD with RBF kernel k(x,y) = exp(-||x-y||²/(2σ_k²)) between
two diagonal Gaussians P = N(μ_p, Σ_p), Q = N(μ_q, Σ_q):

  k_pp = ∏_d (1 + 2σ_p,d²/σ_k²)^(-1/2)
  k_qq = ∏_d (1 + 2σ_q,d²/σ_k²)^(-1/2)
  k_pq = ∏_d (1 + (σ_p,d² + σ_q,d²)/σ_k²)^(-1/2) · exp(-||μ_p - μ_q||²/(2(σ_p² + σ_q² + σ_k²)))   (per-dim)

In high-d, ∏_d (1 + small)^(-1/2) → e^(-d/2 · small) which underflows quickly.
We test fp32 and bf16 to characterize the underflow regime.
"""
import math
import torch
import torch.nn.functional as F

D = 384                                              # latent_dim in our policy (dim=768/2)


def kl_gaussian(mu, log_var):
    """KL(N(μ, σ²) || N(0, 1)), per-dim then summed."""
    sig2 = log_var.exp()
    return 0.5 * (sig2 + mu * mu - 1.0 - log_var)


def w2_gaussian(mu, log_var):
    """W₂² between N(μ, σ²I) and N(0, I) (diagonal)."""
    sig = (log_var * 0.5).exp()
    return mu * mu + (sig - 1.0) ** 2


def frob_gaussian(mu, log_var):
    """Frobenius distance: ||μ||² + ||σ² - 1||²."""
    sig2 = log_var.exp()
    return mu * mu + (sig2 - 1.0) ** 2


def mmd_gaussian_rbf(mu_q, log_var_q, mu_p=None, log_var_p=None, sigma_k=1.0):
    """Closed-form MMD² with RBF kernel σ_k between two diagonal Gaussians.

    Computes log MMD² for numerical stability (RBF MMD underflows hard in high-d).
    Inputs are (D,) per-dim. Prior defaults to N(0, I).
    Returns the SCALAR log MMD² (or NaN if MMD² ≤ 0 due to numerical error).
    """
    if mu_p is None: mu_p = torch.zeros_like(mu_q)
    if log_var_p is None: log_var_p = torch.zeros_like(log_var_q)
    sk2 = sigma_k ** 2
    var_p, var_q = log_var_p.exp(), log_var_q.exp()
    # k_pp, k_qq, k_pq products in log space (sum of logs)
    log_kpp = (-0.5 * (1.0 + 2 * var_p / sk2).log()).sum()
    log_kqq = (-0.5 * (1.0 + 2 * var_q / sk2).log()).sum()
    log_norm_pq = (-0.5 * (1.0 + (var_p + var_q) / sk2).log()).sum()
    # exp_pq is per-dim with shared denominator (sum then exp_per_dim doesn't factor cleanly; use sum approx)
    # for diagonal Gaussians, the exp part factorizes per-dim too:
    exp_pq_arg = (-(mu_p - mu_q) ** 2 / (2 * (var_p + var_q + sk2))).sum()
    log_kpq = log_norm_pq + exp_pq_arg
    # MMD² = k_pp + k_qq - 2 k_pq.  Compute via logsumexp tricks.
    # Note MMD² ≥ 0 in theory but rounding can give tiny negatives → log undef.
    kpp, kqq, kpq = log_kpp.exp(), log_kqq.exp(), log_kpq.exp()
    mmd2 = kpp + kqq - 2.0 * kpq
    return {
        'log_kpp': log_kpp.item(), 'log_kqq': log_kqq.item(), 'log_kpq': log_kpq.item(),
        'mmd2': mmd2.item(), 'kpp': kpp.item(), 'kqq': kqq.item(), 'kpq': kpq.item()
    }


# ──────────────────────────────────────────────────────────────────────
# Scenarios stressing each regularizer
# ──────────────────────────────────────────────────────────────────────
def test_scenario(name, mu, log_var, sigma_k=1.0):
    print(f"\n=== {name}: μ_mean={mu.mean().item():.3f} σ²_range=[{log_var.exp().min().item():.2e}, "
          f"{log_var.exp().max().item():.2e}] ===")
    print(f"  KL per-dim sum:   {kl_gaussian(mu, log_var).sum().item():>14.4e}")
    print(f"  W2² per-dim sum:  {w2_gaussian(mu, log_var).sum().item():>14.4e}")
    print(f"  Frob² per-dim:    {frob_gaussian(mu, log_var).sum().item():>14.4e}")
    mmd = mmd_gaussian_rbf(mu, log_var, sigma_k=sigma_k)
    print(f"  RBF-MMD² (σ_k={sigma_k}): mmd²={mmd['mmd2']:>10.4e}  "
          f"log(k_pp)={mmd['log_kpp']:>8.2f}  log(k_pq)={mmd['log_kpq']:>8.2f}")
    if mmd['kpp'] == 0 or mmd['kqq'] == 0 or mmd['kpq'] == 0:
        print(f"  → ⚠️ underflow: kpp={mmd['kpp']:.2e}, kqq={mmd['kqq']:.2e}, kpq={mmd['kpq']:.2e}")


# (a) Normal regime: μ=0, σ²=1 (matches prior exactly → all distances should be 0)
mu0 = torch.zeros(D); lv0 = torch.zeros(D)
test_scenario("matches prior exactly", mu0, lv0)

# (b) σ² → 0 (posterior collapse): KL log term diverges
mu0_sigsmall = torch.zeros(D); lv_neg = torch.full((D,), -10.0)         # σ² = e^-10 ≈ 4.5e-5
test_scenario("σ² very small (posterior collapse)", mu0_sigsmall, lv_neg)

# (c) σ² → ∞ (wide posterior): KL σ² term explodes
mu0_sigbig = torch.zeros(D); lv_pos = torch.full((D,), +10.0)            # σ² = e^10 ≈ 22000
test_scenario("σ² very large (uninformative posterior)", mu0_sigbig, lv_pos)

# (d) μ shifted away
mu_far = torch.full((D,), 5.0); lv1 = torch.zeros(D)
test_scenario("μ shifted to (5, 5, ...)", mu_far, lv1)

# (e) realistic mid-training: μ ~ N(0, .5), σ² ~ exp(N(-1, 1))
torch.manual_seed(0)
mu_real = torch.randn(D) * 0.5
lv_real = torch.randn(D) - 1.0
test_scenario("realistic mid-training", mu_real, lv_real)

# (f) bf16 stress: cast to bfloat16 and recompute
print("\n=== bf16 cast — does anything explode? ===")
for name, mu, lv in [('matches prior', mu0, lv0),
                     ('σ² small', mu0, lv_neg),
                     ('σ² large', mu0, lv_pos),
                     ('μ shifted', mu_far, lv1)]:
    mu_b, lv_b = mu.bfloat16(), lv.bfloat16()
    kl_b = kl_gaussian(mu_b, lv_b).sum().item()
    w2_b = w2_gaussian(mu_b, lv_b).sum().item()
    frob_b = frob_gaussian(mu_b, lv_b).sum().item()
    mmd_b = mmd_gaussian_rbf(mu_b, lv_b)
    print(f"  {name:<30s} | KL={kl_b:>10.2e}  W²={w2_b:>10.2e}  Frob²={frob_b:>10.2e}  "
          f"MMD²={mmd_b['mmd2']:>10.2e}  (kpp={mmd_b['kpp']:>10.2e})")


# (g) gradient stability — backward through each
print("\n=== gradient stability: ||∂loss/∂μ|| + ||∂loss/∂log_var|| ===")
for name, mu, lv in [('σ² small', mu0.clone(), lv_neg.clone()),
                     ('σ² large', mu0.clone(), lv_pos.clone()),
                     ('realistic', mu_real.clone(), lv_real.clone())]:
    mu.requires_grad_(True); lv.requires_grad_(True)
    for reg_name, reg_fn in [('KL', kl_gaussian), ('W2', w2_gaussian), ('Frob', frob_gaussian)]:
        if mu.grad is not None: mu.grad.zero_()
        if lv.grad is not None: lv.grad.zero_()
        reg_fn(mu, lv).sum().backward()
        gm = mu.grad.norm().item(); gv = lv.grad.norm().item()
        print(f"  {name:<15s} | {reg_name:<5s} | ||∂μ||={gm:>10.3e}  ||∂log_var||={gv:>10.3e}")
