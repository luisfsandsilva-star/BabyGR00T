"""TRM policy with additive (geometrically-decayed) latent updates.

Vanilla TRM in every respect — two latents `z_H` (running solution) and
`z_L` (re-derived reasoning), one shared tiny net `g`, L inner / H outer
recursions, deep supervision — except the latent update is **additive with
a decaying weight** instead of a replacement:

    z ← z + a_t · g(z + context + y,  kv)

A standard residual block is already `z ← z + g(z)`; we only put the weight
`a_t` on the transform `g`. The weights are a **closed form in the loop
length** — a total decay `ρ` spread across the `n` steps of the loop:

    a_t = ρ^{ t / (n-1) } = ρ^{linspace(0,1,n)_t} ,   t = 0,…,n-1,
          n = L (inner) or H (outer)

`a_0 = 1` (full-weight first step), `a_{n-1} = ρ` exactly (last step), and the
*profile* in loop-fraction `t/(n-1)` is identical for any `n` (the schedule is
the same curve sampled at `n` points). So each loop completes the same
refinement over whatever step budget it is given — "converges exactly,
independent of #steps." `ρ_L`, `ρ_H` are a single learnable scalar per loop
(sigmoid → (0,1)); this only sets the decay *rate*, the decaying/bounded
structure holds for every ρ ∈ (0,1) so the guarantee is unaffected.

The weights are *not* normalized: the accumulated latent grows with more
steps, so predictions sharpen with more compute. No contraction / spectral
norm / Lipschitz bound is needed — within any finite loop the update is a
bounded weighted sum of the bounded transforms `g(·)`.

`g` returns a pure transform (sum of the sub-layer outputs, no identity
pass-through) — the accumulation is the residual stream. `g(z, ctx, kv)`
takes the iterate `z` and the conditioning context `ctx = z_other + y`
**concatenated** (not summed) and linearly mixed by an input projection,
plus vision/state via cross-attention (`kv`). Concatenation (vs. the old
`g(z + ctx)`) gives a separate learnable z→z path whose spectral norm the
model can drive below 1, so the damped map is contractive and a fixed point
exists — see TRMNet. Attention is L2 (distance) form and every sub-layer is
LayerScale-gated so the per-block Lipschitz constant is learnable below
expansive. No output ScaleNorm and no Tikhonov noise are needed.

The decay lives ONLY in the `z` update rule — the decoding head reads the
raw accumulated latent (no ScaleNorm, no decay), so logit magnitude is free
to grow as the latent sharpens.

MASK is a plain input marker (a learned row in the token table used at
masked positions); the head predicts the K real codes.
"""
import math
import random as _random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt


# ── Building blocks ──

# GQA-aware SDPA with a fallback for torch < 2.5 (no enable_gqa kwarg).
# enable_gqa=True internally repeats KV heads; doing it manually is identical math
# and works on any torch version (e.g. the sim venv's torch 2.4.1).
# Probe support via a tiny call (inspect.signature fails on the C builtin).
def _probe_gqa():
    try:
        _q = torch.zeros(1, 2, 1, 4); _kv = torch.zeros(1, 1, 1, 4)
        F.scaled_dot_product_attention(_q, _kv, _kv, enable_gqa=True)
        return True
    except TypeError:
        return False
    except Exception:
        return False
_SDPA_HAS_GQA = _probe_gqa()


def _sdpa_gqa(q, k, v, dropout_p, H, Hkv, scale=None):
    if H == Hkv:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)
    if _SDPA_HAS_GQA:
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale, enable_gqa=True)
    # manual KV expansion: (B, Hkv, T, Hd) → (B, H, T, Hd)
    rep = H // Hkv
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, scale=scale)


_SQRT2 = math.sqrt(2.0)


def _l2_augment(q, k):
    """Augmentation trick for **L2 (distance) attention**: build q̃, k̃ such that
    q̃·k̃ = −‖q−k‖², so a plain dot-product kernel computes the distance score.

        −‖q−k‖² = 2 q·k − ‖q‖² − ‖k‖²
        q̃ = [√2·q ; −‖q‖² ;   1   ]   (d+2 dims)
        k̃ = [√2·k ;   1    ; −‖k‖² ]

    Unlike dot-product attention (whose logits ‖q‖‖k‖ are unbounded → not
    Lipschitz), distance attention is Lipschitz-bounded (Kim et al. 2021,
    "The Lipschitz Constant of Self-Attention"), so combined with LayerScale the
    model can pull each block's Lipschitz constant below 1 (non-expansive) and a
    fixed point of the damped recursion becomes attainable. Reuses the fused SDPA
    kernel by passing scale=1/√d (the original, un-augmented head_dim)."""
    qn = (q * q).sum(-1, keepdim=True)                       # ‖q‖²
    kn = (k * k).sum(-1, keepdim=True)                       # ‖k‖²
    q_aug = torch.cat([_SQRT2 * q, -qn, torch.ones_like(qn)], dim=-1)
    k_aug = torch.cat([_SQRT2 * k, torch.ones_like(kn), -kn], dim=-1)
    return q_aug, k_aug


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.eps = eps

    def forward(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=self.eps) * self.g


class LayerScale(nn.Module):
    """Learnable per-channel scale on a sub-layer output (CaiT, Touvron 2021).
    Initialized small so each block starts near-identity / non-expansive; the
    gain γ is free to grow, letting the model set each block's Lipschitz
    contribution — the lever that makes the damped recursion contractive."""
    def __init__(self, dim, init=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x):
        return x * self.gamma


class SelfAttention(nn.Module):
    """MHSA with QK-norm; supports GQA (kv_heads < heads). Plain softmax, no sink."""
    def __init__(self, dim, heads, dropout=0.0, kv_heads=None):
        super().__init__()
        assert dim % heads == 0
        kv_heads = heads if kv_heads is None else kv_heads
        assert heads % kv_heads == 0, f"heads ({heads}) must be divisible by kv_heads ({kv_heads})"
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.kv_dim = self.head_dim * kv_heads
        self.drop_p = dropout
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.kv_dim, bias=False)
        self.wv = nn.Linear(dim, self.kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        # QK-norm without bias: modern practice (NormFormer, PaLM, ViT-22B).
        # bias=False removes the offset and keeps γ scale only — eliminates the
        # tiny-norm (||W||~1e-4) bias params that destabilize SGDM at any non-tiny LR.
        self.q_norm = nn.LayerNorm(self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Hkv, Hd = self.heads, self.kv_heads, self.head_dim
        q = self.wq(x).view(B, T, H,   Hd).transpose(1, 2)        # (B, H,   T, Hd)
        k = self.wk(x).view(B, T, Hkv, Hd).transpose(1, 2)        # (B, Hkv, T, Hd)
        v = self.wv(x).view(B, T, Hkv, Hd).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k)
        q, k = _l2_augment(q, k)                                  # L2 (distance) attention
        o = _sdpa_gqa(q, k, v, self.drop_p if self.training else 0.0, H, Hkv,
                      scale=Hd ** -0.5)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    """MHCA with QK-norm; supports GQA (kv_heads < heads). Vision/state KV."""
    def __init__(self, dim, heads, dropout=0.0, kv_heads=None):
        super().__init__()
        assert dim % heads == 0
        kv_heads = heads if kv_heads is None else kv_heads
        assert heads % kv_heads == 0, f"heads ({heads}) must be divisible by kv_heads ({kv_heads})"
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.kv_dim = self.head_dim * kv_heads
        self.drop_p = dropout
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.kv_dim, bias=False)
        self.wv = nn.Linear(dim, self.kv_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        # QK-norm without bias: modern practice (NormFormer, PaLM, ViT-22B).
        # bias=False removes the offset and keeps γ scale only — eliminates the
        # tiny-norm (||W||~1e-4) bias params that destabilize SGDM at any non-tiny LR.
        self.q_norm = nn.LayerNorm(self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, bias=False)

    def forward(self, x, kv):
        B, T, D = x.shape
        _, M, _ = kv.shape
        H, Hkv, Hd = self.heads, self.kv_heads, self.head_dim
        q = self.wq(x ).view(B, T, H,   Hd).transpose(1, 2)
        k = self.wk(kv).view(B, M, Hkv, Hd).transpose(1, 2)
        v = self.wv(kv).view(B, M, Hkv, Hd).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k)
        q, k = _l2_augment(q, k)                                  # L2 (distance) attention
        o = _sdpa_gqa(q, k, v, self.drop_p if self.training else 0.0, H, Hkv,
                      scale=Hd ** -0.5)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class GeGLU(nn.Module):
    """GeGLU FFN (Shazeer 2020)."""
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.gelu(self.w1(x)) * self.w2(x)))


class TRMNet(nn.Module):
    """The shared tiny net `g(z, ctx, kv)`. `depth` pre-norm sub-blocks
    (SelfAttn → CrossAttn → GeGLU). Returns the **sum of the sub-layer
    transforms** — a pure update direction, no identity pass-through (the
    accumulation in the policy is the residual stream).

    **Fixed-point fix (MNIST-characterized):** the iterate `z` and the
    conditioning context `ctx` are **concatenated** and linearly mixed by
    `in_proj`, *not* summed (`g(z + ctx)`). Summing forces the z→z Jacobian
    path through identity, so the map `T(z) = g(z + ctx)` is non-contractive
    in z and no fixed point can exist. Concatenation gives the model a separate
    learnable `W_z` block (the left half of `in_proj`) whose spectral norm it
    can drive below 1 — making `T(z) = g([z, ctx])` contractive so a fixed
    point of the damped recursion exists.

    Pre-norm is used everywhere (ScaleNorm), including on `z` and `ctx` before
    `in_proj`, and every sub-layer output is gated by a learnable **LayerScale**
    — together with L2 attention these let the model set its Lipschitz constant
    below expansive naturally. No ScaleNorm on the recursion output (the head
    reads the raw latent) and no Tikhonov input-noise are needed.
    """
    def __init__(self, dim, heads, ff_hidden, depth=2, dropout=0.0, kv_heads=None,
                 layerscale_init=0.1):
        super().__init__()
        # Pre-norm + concat-project the (latent, context) pair → dim.
        self.in_norm_z = ScaleNorm(dim)
        self.in_norm_c = ScaleNorm(dim)
        self.in_proj = nn.Linear(2 * dim, dim, bias=False)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'sa_norm': ScaleNorm(dim), 'sa': SelfAttention(dim, heads, dropout, kv_heads=kv_heads),
                'ls_sa': LayerScale(dim, layerscale_init),
                'ca_norm': ScaleNorm(dim), 'ca': CrossAttention(dim, heads, dropout, kv_heads=kv_heads),
                'ls_ca': LayerScale(dim, layerscale_init),
                'ff_norm': ScaleNorm(dim), 'ff': GeGLU(dim, ff_hidden, dropout),
                'ls_ff': LayerScale(dim, layerscale_init),
            }) for _ in range(depth)
        ])

    def forward(self, z, ctx, kv):
        # Concatenate latent + context (not sum) → enables a contractive z-path.
        u = self.in_proj(torch.cat([self.in_norm_z(z), self.in_norm_c(ctx)], dim=-1))
        h = u
        out = torch.zeros_like(u)
        for blk in self.blocks:
            s = blk['ls_sa'](blk['sa'](blk['sa_norm'](h)));      h = h + s; out = out + s
            c = blk['ls_ca'](blk['ca'](blk['ca_norm'](h), kv));  h = h + c; out = out + c
            f = blk['ls_ff'](blk['ff'](blk['ff_norm'](h)));      h = h + f; out = out + f
        return out


# ════════════════════════════════════════════════════════════
#  Policy
# ════════════════════════════════════════════════════════════

class STRMPolicy(nn.Module):
    """TRM over [vis | state] (cross-attn) + [L0 | L1 | L2] action tokens,
    with additive geometrically-decayed latent updates.

    seq_lens : per-level token counts (default (4, 8, 16) for the 3-level CQ-VAE).
    Each level has its own codebook input-embedding E_l and output head W_l.

    Recipe knobs:
      depth   : sub-blocks inside the shared net g (TRM stays tiny; default 2)
      L_inner : inner recursions; H_outer : outer recursions (deep supervision)
      rho_L / rho_H : initial total-decay rate for the closed-form weights
                      a_t = ρ^{t/(n-1)} (single learnable scalar per loop,
                      sigmoid-constrained to (0,1); separate for L and H).
      grad_checkpoint : checkpoint each outer cycle (memory O(one cycle)).
    """
    def __init__(self, seq_lens=(4, 8, 16), k_codebook=128,
                 dim=768, heads=8, kv_heads=None, ff_hidden=None, depth=2,
                 L_inner=5, H_outer=4,
                 rho_L=0.1, rho_H=0.1,
                 dropout=0.0, max_prefix=160, state_dim=6,
                 grad_checkpoint=True, weighting='geometric',
                 update_mode='accumulator', alpha_parametrization='clamp',
                 alpha_per_dim=False,
                 n_embodiments=1, per_emb_head=False,
                 g_input_noise=0.0, layerscale_init=0.1, one_step_grad=False,
                 output_scalenorm=False, carry_zl=False, nesterov=False, nesterov_beta=0.7,
                 inner_tol=0.0):
        super().__init__()
        # inner_tol: relative fixed-point residual at which the no_grad inner warmup early-stops
        #   (0 = run all L steps). Speeds up carry_zl warm-started cycles with no value change.
        self.inner_tol = float(inner_tol)
        # carry_zl: persist the reasoning latent z_L across outer cycles (TRM-faithful —
        #   z_L is read from the previous cycle, not reset to 0). The official TRM carries
        #   both z_H and z_L; our original reset z_L each cycle, preventing the reasoning from
        #   accumulating across cycles. damped + one_step only.
        # nesterov: accelerate the inner fixed-point iteration with Nesterov momentum +
        #   adaptive restart (ŷ=z+β(z−z₋); damped step on ŷ; revert to plain damped if the
        #   fp-residual rises). damped mode only.
        self.carry_zl = bool(carry_zl)
        self.nesterov = bool(nesterov)
        self.nesterov_beta = float(nesterov_beta)
        # HRM/DEQ 1-step (Jacobian-free) gradient: reach the fixed point under
        # no_grad, backprop through only the last step. Valid because the damped
        # map is contractive (see fixed_point_diagnostics). damped + training only.
        self.one_step_grad = bool(one_step_grad)
        # Bishop 1995 input-noise Tikhonov: add fresh noise N(0, σ²I) to g's
        # input at EVERY g-call (training only). Equivalent to penalizing
        # σ²·E[‖∂g/∂x‖²] — regularizes g's Jacobian, bounds its Lipschitz
        # constant, helps damped iteration's contractivity guarantee.
        # NOT noise on the target (that's denoising AE, different regularizer).
        # NOT noise on z (that contaminates the recurrent state across iters).
        # Fresh per-call so each of the H·(L+1) g-calls sees independent noise.
        self.g_input_noise = float(g_input_noise)
        self.seq_lens = list(seq_lens)
        self.n_levels = len(self.seq_lens)
        self.dim = dim
        self.k = k_codebook
        self.L_inner = L_inner
        self.H_outer = H_outer
        self.depth = depth
        self.grad_checkpoint = grad_checkpoint
        self.mask_idx = k_codebook                       # MASK input marker
        # weighting scheme for the inner/outer convex combination (accumulator mode):
        #   'geometric': w_t = ρ^(t/(n-1)) normalized — single ρ ∈ (0,1); 'rho_*_raw' = ρ (clamp_direct)
        #   'linear':    w_t = clamp(1 + slope·(t − (n−1)/2), eps, ∞) normalized — slope ∈ ℝ; 'rho_*_raw' = slope
        # update_mode = 'damped':  z = (1-α)·z + α·g(z+c, kv) per step — α = rho_*_raw under
        #                          chosen parametrization. Schedule-INDEPENDENT (converges to fp of g).
        #                          Complementary gate form → sigmoid grad doesn't vanish at extremes
        #                          because loss has a clean signal (higher α = faster fp convergence).
        # alpha_parametrization (damped only):  'clamp' (α=clamp(raw,eps,1-eps)) or 'sigmoid' (α=σ(raw)).
        self.weighting = weighting
        self.update_mode = update_mode
        self.alpha_parametrization = alpha_parametrization
        # alpha_per_dim: if True (damped only), rho_*_raw is shape (dim,) — each channel gates independently.
        # Lets different features have different damping rates.
        self.alpha_per_dim = bool(alpha_per_dim) and update_mode == 'damped'
        if self.alpha_per_dim:
            self.rho_L_raw = nn.Parameter(torch.full((dim,), float(rho_L)))
            self.rho_H_raw = nn.Parameter(torch.full((dim,), float(rho_H)))
        else:
            self.rho_L_raw = nn.Parameter(torch.tensor(float(rho_L)))
            self.rho_H_raw = nn.Parameter(torch.tensor(float(rho_H)))

        if ff_hidden is None:
            ff_hidden = (int(dim * 8 / 3) + 63) // 64 * 64

        # Per-level token embeddings (K codes + 1 MASK marker row).
        self.tok_emb = nn.ModuleList([
            nn.Embedding(k_codebook + 1, dim) for _ in range(self.n_levels)
        ])
        # Additive learned position / level embeddings (ViT-standard init).
        self.level_emb = nn.Parameter(torch.empty(self.n_levels, dim))
        nn.init.trunc_normal_(self.level_emb, std=0.02)
        self.pos_emb = nn.ParameterList([
            nn.Parameter(torch.empty(t, dim)) for t in self.seq_lens
        ])
        for p in self.pos_emb:
            nn.init.trunc_normal_(p, std=0.02)

        # Vision prefix + state → cross-attn KV.
        self.prefix_pos_emb = nn.Parameter(torch.empty(max_prefix, dim))
        nn.init.trunc_normal_(self.prefix_pos_emb, std=0.02)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.state_pos_emb = nn.Parameter(torch.empty(1, dim))
        nn.init.trunc_normal_(self.state_pos_emb, std=0.02)

        # Shared tiny net g.
        self.g = TRMNet(dim, heads, ff_hidden, depth=depth, dropout=dropout, kv_heads=kv_heads,
                        layerscale_init=layerscale_init)

        # Read-out head (plain Linear over the K real codes; deep supervision).
        # NO norm on the head — it reads the raw accumulated latent so logit
        # magnitude is free to grow as the latent sharpens. The decay lives
        # only in the z-update rule.
        self.n_embodiments = int(n_embodiments)
        self.per_emb_head = bool(per_emb_head)
        if self.per_emb_head:
            self.out_head = nn.ModuleList([
                nn.ModuleList([nn.Linear(dim, k_codebook) for _ in range(self.n_levels)])
                for _ in range(self.n_embodiments)
            ])
        else:
            self.out_head = nn.ModuleList([
                nn.Linear(dim, k_codebook) for _ in range(self.n_levels)
            ])
        # Optional ScaleNorm on the recursion OUTPUT (the fixed-point z) before the readout.
        # The redesign removed it ("not needed" on short MNIST runs), but long runs show ‖z*‖
        # drifting to ~70-117 + the fixed point loosening (resid→0.35), which contaminates the
        # readout. Re-adding it bounds what the head sees (projects z to a learnable-radius sphere)
        # — at the cost of discarding magnitude (watch val acc/loss for expressiveness loss).
        self.out_norm = ScaleNorm(dim) if output_scalenorm else None

    # ── Helpers ──

    def _rhos(self):
        # For 'damped' mode, rho_*_raw is α (or "tilt"-from-0.5) per alpha_parametrization.
        # For 'accumulator' mode with 'geometric' weighting, it's ρ in (0, 1).
        # For 'accumulator' mode with 'linear' weighting, it's an unbounded slope.
        if self.update_mode == 'damped':
            if self.alpha_parametrization == 'sigmoid':
                return torch.sigmoid(self.rho_L_raw), torch.sigmoid(self.rho_H_raw)
            elif self.alpha_parametrization == 'tilt':
                # α = clamp(0.5 + tilt, eps, 1-eps); init tilt=0 → α=0.5; unit gradient inside.
                return ((0.5 + self.rho_L_raw).clamp(1e-3, 1 - 1e-3),
                        (0.5 + self.rho_H_raw).clamp(1e-3, 1 - 1e-3))
            else:                                                 # clamp_direct
                return (self.rho_L_raw.clamp(1e-3, 1 - 1e-3),
                        self.rho_H_raw.clamp(1e-3, 1 - 1e-3))
        if self.weighting == 'linear':
            return self.rho_L_raw, self.rho_H_raw
        return (self.rho_L_raw.clamp(1e-3, 1 - 1e-3),
                self.rho_H_raw.clamp(1e-3, 1 - 1e-3))

    def _weights(self, rho, n, device, dtype):
        """Closed-form refinement weights ∝ ρ^{t/(n-1)} = ρ^{linspace(0,1,n)},
        normalized to unit mass (Σ_t a_t = 1) so the accumulator is a convex
        combination → bounded and Cauchy-convergent in the step budget n. The
        decay *ratios* a_t/a_0 = ρ^{t/(n-1)} (hence the budget-invariant profile)
        are preserved; only the overall mass is fixed. n=1 → [1.0]. ρ may be a
        learnable tensor (gradient flows)."""
        if not torch.is_tensor(rho):
            rho = torch.tensor(float(rho), device=device, dtype=dtype)
        rho = rho.to(device=device, dtype=dtype)
        if self.update_mode == 'damped':
            return rho                                              # damped: just pass α through (scalar or per-dim)
        if self.weighting == 'linear':
            t = torch.arange(n, device=device, dtype=dtype)
            w = (1.0 + rho * (t - (n - 1) / 2)).clamp(min=1e-3)
            return w / w.sum()
        # geometric (default)
        expo = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        w = rho ** expo
        # Cauchy-convergent normalization: normalize the closed-form geometric
        # profile to unit mass so the accumulator is a CONVEX combination of the
        # (bounded) transforms. Without this, Σ_t ρ^{t/(n-1)} ≈ 0.44·n grows with
        # the step budget, so ‖z‖ diverges as n→∞; normalized, ‖z‖ ≤ max‖g‖
        # uniformly in n and z_n converges as n→∞ (Cauchy). The decay *shape*
        # (ratios ρ^{t/(n-1)}) — hence the budget-invariant profile — is
        # preserved; only the overall mass is fixed. Applied to both L and H
        # (this is the single shared weight fn), so the two loops stay symmetric.
        return w / w.sum()

    def _build_kv(self, vis, state):
        B, P, _ = vis.shape
        vis_p = vis + self.prefix_pos_emb[:P].unsqueeze(0).to(vis.dtype)
        st = self.state_proj(state).unsqueeze(1) + self.state_pos_emb.unsqueeze(0)
        return torch.cat([vis_p, st], dim=1)

    def _y_embed(self, B, dev, indices_list, mask_list):
        """Task input: MASK marker at masked positions, GT code at unmasked,
        plus level + position embeddings. Static across cycles (the evolving
        prediction lives in the latent z_H, TRM-style)."""
        outs = []
        for l, T_l in enumerate(self.seq_lens):
            if indices_list is not None and mask_list is not None:
                idx = torch.where(mask_list[l],
                                  torch.full_like(indices_list[l], self.mask_idx),
                                  indices_list[l])
            else:
                idx = torch.full((B, T_l), self.mask_idx,
                                 dtype=torch.long, device=dev)
            emb = (self.tok_emb[l](idx)
                   + self.level_emb[l].view(1, 1, -1)
                   + self.pos_emb[l].unsqueeze(0))
            outs.append(emb)
        return torch.cat(outs, dim=1)

    def _heads(self, z, emb_id=None):
        """Raw latent → head. If per_emb_head, route each sample through its embodiment's head."""
        offset = 0
        out = []
        for l, T_l in enumerate(self.seq_lens):
            z_slice = z[:, offset:offset + T_l, :]
            if self.out_norm is not None:                         # bound the readout's input magnitude
                z_slice = self.out_norm(z_slice)
            if self.per_emb_head and emb_id is not None:
                B = z_slice.shape[0]
                # build first sample to discover output dtype (autocast safety)
                first_eid = int(emb_id[0].item())
                if first_eid >= self.n_embodiments: first_eid = 0
                sample = self.out_head[first_eid][l](z_slice[:1])
                logits = torch.zeros(B, T_l, self.k, device=z.device, dtype=sample.dtype)
                logits[0:1] = sample
                for eid in emb_id.unique().tolist():
                    if eid >= self.n_embodiments: continue
                    mask = emb_id == eid
                    if mask[0]: mask[0] = False
                    if mask.any():
                        logits[mask] = self.out_head[eid][l](z_slice[mask])
                out.append(logits)
            else:
                head = self.out_head[l] if not self.per_emb_head else self.out_head[0][l]
                out.append(head(z_slice))
            offset += T_l
        return out


    def _g_noisy(self, z, ctx, kv):
        """g(z, ctx) with optional Bishop-1995 input-noise Tikhonov: fresh noise
        per call, training-only. σ=0 → bypass (no-op; default, since experiments
        showed Tikhonov is not needed once L2-attn + LayerScale bound g)."""
        if self.training and self.g_input_noise > 0:
            z = z + torch.randn_like(z) * self.g_input_noise
            ctx = ctx + torch.randn_like(ctx) * self.g_input_noise
        return self.g(z, ctx, kv)

    def _inner(self, z_H, y, kv, wL, z_L_init=None, n_steps=None):
        """Inner loop. Three modes:
        - 'accumulator' (default): z_L = z_L + w_t · g(z_L+z_H+y, kv) — Cauchy partial sum.
        - 'damped':             z_L = (1-α)·z_L + α·g(z_L+z_H+y, kv) — damped Banach.
        - 'bayesian' (VAE only):  precision-weighted Gaussian fusion. z_L = [μ | log τ].
                                  Each iteration treats g's output as new evidence with its
                                  own (μ_g, log τ_g) and computes the posterior:
                                    τ_post = τ_prior + τ_g    (precisions add)
                                    μ_post = (τ_prior·μ_prior + τ_g·μ_g) / τ_post
                                  No gate (α/ρ) — the evidence's own precision controls
                                  how much it shifts the belief. Information-monotonic.
        """
        z_L = torch.zeros_like(y) if z_L_init is None else z_L_init   # carry_zl: warm-start from prev cycle
        if self.update_mode == 'bayesian':
            assert hasattr(self, 'latent_dim'), "bayesian mode requires VAE belief split"
            L = self.latent_dim
            for t in range(self.L_inner):
                g_out = self._g_noisy(z_L, z_H + y, kv)
                z_L = self._bayes_fuse(z_L, g_out, L)
        elif self.update_mode == 'damped':
            alpha = wL                                                # (scalar) or (D,) for per-dim
            n = self.L_inner if n_steps is None else int(n_steps)
            # Early-stop the no_grad warmup once the fixed point is reached (carry_zl warm-starts
            # later cycles, so they converge in 1-2 steps — running all n is wasted compute). Gated
            # on no_grad so a gradient path is NEVER shortened; the fixed point (hence value) is unchanged.
            es = (not torch.is_grad_enabled()) and self.inner_tol > 0
            if self.nesterov:                                         # Nesterov + adaptive restart
                ctx = z_H + y
                z_prev = z_L; prev_resid = None; beta = self.nesterov_beta
                for t in range(n):
                    yj = z_L + beta * (z_L - z_prev)
                    g_y = self._g_noisy(yj, ctx, kv)
                    resid = (g_y - yj).flatten(1).norm(dim=1).mean()
                    if prev_resid is not None and resid > prev_resid:  # momentum overshoot → restart (plain damped)
                        g_z = self._g_noisy(z_L, ctx, kv)
                        z_new = (1 - alpha) * z_L + alpha * g_z
                        resid = (g_z - z_L).flatten(1).norm(dim=1).mean()
                    else:
                        z_new = (1 - alpha) * yj + alpha * g_y
                    prev_resid = resid; z_prev, z_L = z_L, z_new
                    if es and resid < self.inner_tol * z_L.flatten(1).norm(dim=1).mean().clamp(min=1e-6):
                        break                                          # converged (warm-start)
            else:
                for t in range(n):
                    g_z = self._g_noisy(z_L, z_H + y, kv)
                    z_new = (1 - alpha) * z_L + alpha * g_z
                    z_L = z_new
                    if es and (g_z - z_new).flatten(1).norm(dim=1).mean() < self.inner_tol * z_L.flatten(1).norm(dim=1).mean().clamp(min=1e-6):
                        break
        else:
            for t in range(wL.shape[0]):
                z_L = z_L + wL[t] * self._g_noisy(z_L, z_H + y, kv)
        return z_L

    @staticmethod
    def _bayes_fuse(z_prior, g_evidence, latent_dim, lp_clamp=5.0):
        """Bayesian Gaussian fusion: z_post = fuse(z_prior, g_evidence) where both are
        [μ | log τ] tensors split on the last dim. Returns cat([μ_post, log τ_post]).
        lp_clamp keeps log-precision in [-lp_clamp, lp_clamp] for numerical stability."""
        mu_p, lp_p = z_prior[..., :latent_dim], z_prior[..., latent_dim:]
        mu_g, lp_g = g_evidence[..., :latent_dim], g_evidence[..., latent_dim:]
        lp_p = lp_p.clamp(-lp_clamp, lp_clamp)
        lp_g = lp_g.clamp(-lp_clamp, lp_clamp)
        tau_p, tau_g = lp_p.exp(), lp_g.exp()
        tau_post = tau_p + tau_g
        mu_post = (tau_p * mu_p + tau_g * mu_g) / tau_post
        lp_post = tau_post.log().clamp(-lp_clamp, lp_clamp)
        return torch.cat([mu_post, lp_post], dim=-1)

    # ── Fixed-point / Lipschitz diagnostics (damped mode) ──

    def _layerscale_stats(self):
        """Mean |γ| of each LayerScale (the learnable per-block Lipschitz lever)."""
        out = []
        for bi, blk in enumerate(self.g.blocks):
            for key in ('ls_sa', 'ls_ca', 'ls_ff'):
                out.append(blk[key].gamma.detach().abs().mean().item())
        return out

    def _spectral_norm_jz(self, z, ctx, kv, iters=8, eps=1e-3):
        """σ_max(∂g(z,ctx,kv)/∂z) via power iteration. J·v is estimated by finite
        differences (forward-only — robust through SDPA), Jᵀ·(Jv) by a single
        autograd backward (also SDPA-safe; avoids fragile double-backward)."""
        z = z.detach().float()
        ctx = ctx.detach().float(); kv = kv.detach().float()
        was_train = self.training
        self.eval()
        f = lambda zz: self.g(zz, ctx, kv)
        with torch.no_grad():
            f0 = f(z)
        v = torch.randn_like(z); v = v / (v.norm() + 1e-12)
        sigma = 0.0
        for _ in range(iters):
            with torch.no_grad():
                Jv = (f(z + eps * v) - f0) / eps                  # ≈ J v
            zr = z.clone().requires_grad_(True)
            with torch.enable_grad():
                out = f(zr)
                JtJv = torch.autograd.grad(out, zr, grad_outputs=Jv)[0]   # Jᵀ(Jv)
            sigma = Jv.norm().item()                              # ‖Jv‖, ‖v‖=1 → Rayleigh est
            nrm = JtJv.norm()
            if nrm < 1e-12:
                break
            v = (JtJv / nrm).detach()
        if was_train:
            self.train()
        return sigma

    def _dominant_eig_jz(self, z, ctx, kv, iters=40, eps=1e-3):
        """Dominant EIGENVALUE of J=∂g/∂z (by magnitude) via power iteration on J directly
        (not JᵀJ). Returns (|λ_dom|, Re(λ_dom)≈Rayleigh v·Jv). Compare to the spectral NORM σ_g:
        |λ_dom| ≪ σ_g ⇒ highly non-normal (norm overstates convergence-relevant eigenvalue);
        |λ_dom| ≈ σ_g ⇒ near-normal. Re<0 large-magnitude ⇒ oscillatory/dampable; Re≥1 ⇒ undampable.
        Caveat: for a complex-dominant pair, |λ_dom| still converges but v·Jv (Re) is approximate."""
        z = z.detach().float(); ctx = ctx.detach().float(); kv = kv.detach().float()
        was = self.training; self.eval()
        f = lambda zz: self.g(zz, ctx, kv)
        with torch.no_grad():
            f0 = f(z)
            v = torch.randn_like(z); v = v / (v.norm() + 1e-12)
            absl = rq = 0.0
            for _ in range(iters):
                Jv = (f(z + eps * v) - f0) / eps
                absl = Jv.norm().item()                              # ‖Jv‖, ‖v‖=1 → |λ_dom|
                rq = (v.flatten() * Jv.flatten()).sum().item()       # v·Jv → Re(λ_dom)
                n = Jv.norm()
                if n < 1e-12: break
                v = Jv / n
        if was: self.train()
        return absl, rq

    @torch.no_grad()
    def fixed_point_diagnostics(self, vis, state, indices_list=None, mask_list=None,
                                emb_id=None, n_iter=24, power_iters=8):
        """Damped-mode probe. Runs the inner damped iteration z←(1-α)z+α·g(z,ctx)
        with z_H=0 / fixed ctx=y for n_iter steps (beyond the train budget so we can
        see whether it actually converges), then estimates the Lipschitz constants.
        Returns: resid (‖Δz‖ per step), contraction (geo-mean residual ratio = empirical
        Lipschitz of the damped map T), final_resid, sigma_g (σ_max ∂g/∂z), lip_T_bound
        ((1-α)+α·σ_g; <1 ⇒ guaranteed contraction), alpha, layerscale."""
        assert self.update_mode == 'damped', "fixed_point_diagnostics requires update_mode='damped'"
        B = vis.shape[0]; N = sum(self.seq_lens); dev = vis.device
        if mask_list is None:
            mask_list = [torch.ones(B, T_l, dtype=torch.bool, device=dev) for T_l in self.seq_lens]
        kv = self._build_kv(vis, state)
        y = self._y_embed(B, dev, indices_list, mask_list)
        alpha, _ = self._rhos()                                  # damped α (scalar or per-dim)
        z_H = torch.zeros(B, N, self.dim, device=dev, dtype=y.dtype)
        ctx = z_H + y
        z = torch.zeros_like(y); resid = []
        d0_pe = delta_pe = None
        for t in range(n_iter):
            z_new = (1 - alpha) * z + alpha * self.g(z, ctx, kv)
            delta_pe = (z_new - z).flatten(1).norm(dim=1)            # (B,) per-example ‖Δz‖
            resid.append(delta_pe.norm().item())                    # batch Frobenius (back-compat)
            if t == 0:
                d0_pe = delta_pe.clamp(min=1e-9)
            z = z_new
        ratios = [resid[i + 1] / resid[i] for i in range(len(resid) - 1) if resid[i] > 1e-9]
        contraction = math.exp(sum(math.log(max(r, 1e-9)) for r in ratios) / len(ratios)) if ratios else float('nan')
        a_mean = float(alpha.mean()) if torch.is_tensor(alpha) else float(alpha)
        sigma_g = self._spectral_norm_jz(z, ctx, kv, iters=power_iters)
        # Per-example fixed-point stats: each z[i] IS the fixed point for example i.
        # frac_converged = fraction whose residual dropped ≥100× (reached its fixed point);
        # z_norm spread confirms the fixed points are example-specific (distinct), not collapsed.
        z_norm_pe = z.flatten(1).norm(dim=1)
        return dict(resid=resid, contraction=contraction, final_resid=resid[-1],
                    sigma_g=sigma_g, lip_T_bound=(1 - a_mean) + a_mean * sigma_g,
                    alpha=a_mean, layerscale=self._layerscale_stats(),
                    resid_pe=dict(median=delta_pe.median().item(), p90=delta_pe.quantile(0.9).item(),
                                  max=delta_pe.max().item()),
                    z_norm=dict(min=z_norm_pe.min().item(), median=z_norm_pe.median().item(),
                                max=z_norm_pe.max().item()),
                    frac_converged=(delta_pe / d0_pe < 1e-2).float().mean().item())

    @torch.no_grad()
    def compare_iteration_schemes(self, vis, state, indices_list=None, mask_list=None,
                                  emb_id=None, n_iter=40, beta=0.7, accum_rho=0.5):
        """Run the SAME trained g through 3 fixed-point schemes from z=0 and report how each
        behaves. ctx is held at y (z_H=0), exactly like fixed_point_diagnostics, so all three
        drive the identical map g(·, y, kv).

          'accumulator'  z ← z + w_t·g(z)         (w_t = normalized geometric, ratio accum_rho — "undamped, decaying step")
          'damped'       z ← (1-α)z + α·g(z)       (current; α = learned inner ρ_L)
          'nesterov'     ŷ = z + β(z-z₋); z ← (1-α)ŷ + α·g(ŷ)   (+ adaptive restart on residual increase)

        For each, per step we log:
          step_resid = ‖z_{t+1}-z_t‖   (how big the move is)
          fp_resid   = ‖g(z_t)-z_t‖    (TRUE distance from a g-fixed-point — the honest metric;
                                        accumulator can drive step_resid→0 while fp_resid stays large)
          z_norm     = ‖z_t‖
        and finally the readout masked-CE/acc at the converged z (does the readout still work?).
        Returns {scheme: {step_resid:[...], fp_resid:[...], z_norm:[...], ce, acc}}.
        """
        was_train = self.training; self.eval()
        B = vis.shape[0]; N = sum(self.seq_lens); dev = vis.device
        if mask_list is None:
            mask_list = [torch.ones(B, T_l, dtype=torch.bool, device=dev) for T_l in self.seq_lens]
        kv = self._build_kv(vis, state)
        y = self._y_embed(B, dev, indices_list, mask_list)
        ctx = y                                                  # z_H = 0
        alpha, _ = self._rhos()
        # geometric decaying weights for the accumulator (normalized → convex, Cauchy)
        expo = torch.linspace(0.0, 1.0, n_iter, device=dev, dtype=y.dtype)
        w = (accum_rho ** expo); w = w / w.sum()

        def fp_resid(z):                                         # ‖g(z)-z‖ per-example, batch-mean
            return (self.g(z, ctx, kv) - z).flatten(1).norm(dim=1).mean().item()
        def znorm(z): return z.flatten(1).norm(dim=1).mean().item()

        out = {}
        for scheme in ('accumulator', 'damped', 'nesterov'):
            z = torch.zeros_like(y); z_prev = z.clone()
            sr, fr, zn = [], [], []
            for t in range(n_iter):
                fr.append(fp_resid(z)); zn.append(znorm(z))
                if scheme == 'accumulator':
                    z_new = z + w[t] * self.g(z, ctx, kv)
                elif scheme == 'damped':
                    z_new = (1 - alpha) * z + alpha * self.g(z, ctx, kv)
                else:  # nesterov + adaptive restart
                    yj = z + beta * (z - z_prev)
                    z_new = (1 - alpha) * yj + alpha * self.g(yj, ctx, kv)
                    if t > 0 and fp_resid(z_new) > fr[-1]:      # residual went UP → kill momentum
                        z_new = (1 - alpha) * z + alpha * self.g(z, ctx, kv)
                sr.append((z_new - z).flatten(1).norm(dim=1).mean().item())
                z_prev, z = z, z_new
            # readout at the converged z
            logits = self._heads(z, emb_id)
            ce = acc = float('nan')
            if indices_list is not None:
                tot_ce = tot_c = tot_n = 0.0
                for l, T_l in enumerate(self.seq_lens):
                    m = mask_list[l]
                    lg = logits[l][..., :self.k]
                    tot_ce += F.cross_entropy(lg[m], indices_list[l][m], reduction='sum').item()
                    tot_c += (lg[m].argmax(-1) == indices_list[l][m]).float().sum().item()
                    tot_n += m.sum().item()
                ce = tot_ce / max(tot_n, 1); acc = tot_c / max(tot_n, 1)
            out[scheme] = dict(step_resid=sr, fp_resid=fr, z_norm=zn, ce=ce, acc=acc,
                               final_fp_resid=fr[-1], alpha=float(alpha.mean()) if torch.is_tensor(alpha) else float(alpha))
        if was_train: self.train()
        return out

    @torch.no_grad()
    def diagnose_depth_usage(self, vis, state, indices_list, mask_list, emb_id=None, n_iter=20):
        """WHY don't more iterations help? Iterate the inner map z←(1-α)z+α·g(z,y,kv) and at EACH
        step k record: readout CE/acc(z_k), cosine(z_k, z_final) [direction — what ScaleNorm reads],
        ‖z_k‖, and ‖g(z_k)-z_k‖/‖z_k‖ [how far g is from identity]. If acc + cosine saturate at k=1-2,
        the readout-relevant signal (z direction) is set in ~1 pass ⇒ recursion is ~feedforward."""
        was = self.training; self.eval()
        B = vis.shape[0]; dev = vis.device
        kv = self._build_kv(vis, state); y = self._y_embed(B, dev, indices_list, mask_list); ctx = y
        alpha, _ = self._rhos()
        zs = []; z = torch.zeros_like(y)
        for _ in range(n_iter):
            z = (1 - alpha) * z + alpha * self.g(z, ctx, kv); zs.append(z.clone())
        zf = zs[-1]
        def ce_acc(zin):
            logits = self._heads(zin, emb_id); tc=tn=tce=0.0
            for l in range(len(self.seq_lens)):
                m = mask_list[l]; lg = logits[l][..., :self.k]
                tce += F.cross_entropy(lg[m], indices_list[l][m], reduction='sum').item()
                tc += (lg[m].argmax(-1)==indices_list[l][m]).float().sum().item(); tn += m.sum().item()
            return tce/max(tn,1), tc/max(tn,1)*100
        def stable_rank(M):                                   # ‖M‖_F² / σ_max²  (over-smoothing ⇒ →1)
            M = M - M.mean(0, keepdim=True)
            s = torch.linalg.svdvals(M.float())
            return (s.pow(2).sum() / s[0].pow(2).clamp(min=1e-12)).item()
        rows = []
        for k, zk in enumerate(zs, 1):
            ce, ac = ce_acc(zk)
            cos = F.cosine_similarity(zk.flatten(1), zf.flatten(1), dim=1).mean().item()
            flat = zk.flatten(1)
            # cross-EXAMPLE mean |cosine| (rises ⇒ examples collapse together = over-smoothing)
            fn = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-9)
            B = fn.shape[0]; xcos = (fn @ fn.t()).abs()
            xcos = (xcos.sum() - B) / (B*(B-1))
            srank = stable_rank(flat)                          # effective dimensionality of the z's
            gres = ((self.g(zk, ctx, kv)-zk).flatten(1).norm(dim=1) / flat.norm(dim=1).clamp(min=1e-9)).mean().item()
            rows.append((k, ce, ac, cos, flat.norm(dim=1).mean().item(), gres, xcos.item(), srank))
        if was: self.train()
        return rows

    @torch.no_grad()
    def diagnose_iteration_invariance(self, vis, state, indices_list, mask_list, emb_id=None, n_iter=40):
        """Stress-test WHY the readout looked invariant to the iteration scheme. For the given mask,
        run damped to z*, then measure readout CE/acc when the latent fed to the head is:
          z*  (converged) | 0 | random·‖z*‖ | 2·z* (scaled) | z_accum (non-converged accumulator).
        Reports each WITH the output ScaleNorm (as deployed) AND with it BYPASSED (raw z), plus how
        different z_accum actually is from z* (rel-dist + cosine). If 0/random match z* → readout
        ignores z at this mask (→ z is low-impact here, not a real invariance). Returns a dict.
        """
        was_train = self.training; self.eval()
        B = vis.shape[0]; dev = vis.device
        kv = self._build_kv(vis, state); y = self._y_embed(B, dev, indices_list, mask_list); ctx = y
        alpha, _ = self._rhos()
        z = torch.zeros_like(y)
        for _ in range(n_iter): z = (1 - alpha) * z + alpha * self.g(z, ctx, kv)
        zstar = z
        za = torch.zeros_like(y)
        wexpo = torch.linspace(0., 1., n_iter, device=dev, dtype=y.dtype); w = (0.5 ** wexpo); w = w / w.sum()
        for t in range(n_iter): za = za + w[t] * self.g(za, ctx, kv)
        znorm = zstar.flatten(1).norm(dim=1).mean()                       # scalar mean ‖z*‖
        zr = torch.randn_like(zstar)
        zr = zr / zr.flatten(1).norm(dim=1).view(-1, 1, 1).clamp(min=1e-9) * znorm

        def ce_acc(zin, use_outnorm):
            saved = self.out_norm
            if not use_outnorm: self.out_norm = None
            logits = self._heads(zin, emb_id)
            self.out_norm = saved
            tc = tn = tce = 0.0
            for l in range(len(self.seq_lens)):
                m = mask_list[l]; lg = logits[l][..., :self.k]
                tce += F.cross_entropy(lg[m], indices_list[l][m], reduction='sum').item()
                tc += (lg[m].argmax(-1) == indices_list[l][m]).float().sum().item(); tn += m.sum().item()
            return tce / max(tn, 1), tc / max(tn, 1)

        cands = {'z*': zstar, 'zero': torch.zeros_like(zstar), 'random': zr, '2·z*': 2 * zstar, 'z_accum': za}
        out = {'with_outnorm': {}, 'raw_z': {}}
        for nm, zc in cands.items():
            out['with_outnorm'][nm] = ce_acc(zc, True)
            out['raw_z'][nm] = ce_acc(zc, False)
        # how far is z_accum from z* really?
        rel = ((za - zstar).flatten(1).norm(dim=1) / zstar.flatten(1).norm(dim=1).clamp(min=1e-9)).mean().item()
        cos = F.cosine_similarity(za.flatten(1), zstar.flatten(1), dim=1).mean().item()
        out['z_accum_vs_zstar'] = dict(rel_dist=rel, cosine=cos)
        out['mask_frac'] = float(torch.stack([m.float().mean() for m in mask_list]).mean())
        if was_train: self.train()
        return out

    # ── Inference: MaskGIT iterative decode ──

    @torch.no_grad()
    def generate(self, vis, state, emb_id=None, n_steps=8, temperature=0.0,
                 n_outer=None, n_inner=None, conf_measure='maxprob', sample=False, choice_temp=0.0):
        """MaskGIT iterative decode (Chang+ 2022, Algorithm 1) — the correct inference.

        Start every code MASKed. For each of n_steps (r=(t+1)/n_steps):
          1. forward the partially-decoded sequence → per-position logits;
          2. for each MASKed position, take the predicted code and its softmax
             confidence p(argmax); already-committed positions get +inf (frozen);
          3. keep MASKed the `ceil(γ(r)·T)` LEAST-confident positions and commit the
             rest, where γ(r)=cos(π/2·r) is the cosine mask schedule.
        γ(1)=0 ⇒ the final step commits everything, so the schedule terminates on its
        own (no forced completion). temperature>0 adds annealed Gumbel noise to the
        confidence (stochastic MaskGIT sampling). Returns per-level indices [(B,T_l),…].

        Contrast with the old inference (single all-masked forward + argmax): that hands
        the model 0% visible context and one-shots every code — its hardest, least-trained
        regime, never using the context-conditioning MaskGIT is trained for."""
        was_training = self.training
        self.eval()
        B = vis.shape[0]; dev = vis.device
        cur = [torch.full((B, T_l), self.mask_idx, dtype=torch.long, device=dev) for T_l in self.seq_lens]
        masked = [torch.ones(B, T_l, dtype=torch.bool, device=dev) for T_l in self.seq_lens]
        for t in range(n_steps):
            logits = self.forward(cur, vis, state, mask_list=masked,
                                  n_outer=n_outer, n_inner=n_inner, emb_id=emb_id)[-1]
            r = (t + 1) / n_steps
            for l, T_l in enumerate(self.seq_lens):
                probs = F.softmax(logits[l][..., :self.k].float(), dim=-1)
                if sample:                                                 # faithful MaskGIT: sample the committed token
                    tau = max(temperature, 1e-6)
                    tprobs = F.softmax(logits[l][..., :self.k].float() / tau, dim=-1)
                    pred = torch.multinomial(tprobs.reshape(-1, self.k), 1).reshape(probs.shape[:-1])
                    p_chosen = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1)   # p(sampled) under model probs
                else:                                                      # greedy: chosen token = argmax
                    p_chosen, pred = probs.max(dim=-1)
                if conf_measure == 'entropy':                              # confidence = negentropy (peaked = confident)
                    ent = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
                    conf = -ent
                else:                                                      # 'maxprob' = paper: prob of the chosen token
                    conf = p_chosen
                score = torch.where(masked[l], conf, torch.full_like(conf, float('inf')))  # committed=frozen
                if choice_temp > 0:                                       # annealed Gumbel selection diversity (MaskGIT §3.2)
                    gum = -torch.log(-torch.log(torch.rand_like(score).clamp_min(1e-9)).clamp_min(1e-9))
                    score = score + choice_temp * (1.0 - r) * gum
                gamma = math.cos(math.pi / 2 * r)        # γ(1)=0 mathematically; cos(π/2)≈6e-17 in FP
                if gamma < 1e-6: gamma = 0.0             # snap so the last step commits ALL (no FP leftover)
                n_commit = T_l - int(math.ceil(T_l * gamma))   # = T - γ(r)·T total committed
                if n_commit <= 0:
                    continue                                             # nothing confident enough yet this step
                kth = score.topk(min(n_commit, T_l), dim=-1).values[:, -1:]
                keep_unmasked = score >= kth                             # most-confident (+ frozen committed)
                newly = keep_unmasked & masked[l]
                cur[l] = torch.where(newly, pred, cur[l])
                masked[l] = ~keep_unmasked
        if was_training:
            self.train()
        return cur

    # ── Forward ──

    def _outer_one_grad(self, z_H, y, kv, wL, wH, z_L_carry=None, n_inner=None):
        """One outer cycle with the HRM 1-step gradient (damped only). Reach the
        inner fixed point under no_grad from a DETACHED z_H, then take one
        grad-tracked inner step and one grad-tracked outer step. Gradient flows
        through 2 g-calls instead of L+1, and z_H is detached between cycles, so
        peak graph memory is O(1) in the recursion depth. Forward value ≈ the
        fixed point (z*≈g(z*)), so accuracy is unchanged; only the gradient is
        the Jacobian-free approximation, justified by the measured contraction<1.

        carry_zl: warm-start the inner solve from the previous cycle's z_L (TRM-faithful);
        returns (z_H_new, z_L) so the caller can carry z_L to the next cycle."""
        z_H_in = z_H.detach()
        z_L_init = z_L_carry.detach() if (self.carry_zl and z_L_carry is not None) else None
        with torch.no_grad():
            z_L = self._inner(z_H_in, y, kv, wL, z_L_init=z_L_init, n_steps=n_inner)  # inner fp, no graph
        z_L = (1 - wL) * z_L + wL * self._g_noisy(z_L, z_H_in + y, kv)  # 1 grad inner step
        g_out_H = self._g_noisy(z_H_in, z_L + y, kv)
        z_H_new = (1 - wH) * z_H_in + wH * g_out_H                     # 1 grad outer step
        return z_H_new, z_L

    def forward(self, indices_list, vis, state, mask_list=None,
                n_outer=None, n_inner=None, emb_id=None):
        """Run H outer cycles, each with L inner recursions. Returns a list of
        length H, each a list of per-level (B, T_l, K) logits.

        emb_id (B,): routes per-emb action heads when per_emb_head is True."""
        B = vis.shape[0]
        N = sum(self.seq_lens)
        H = n_outer if n_outer is not None else self.H_outer
        L = n_inner if n_inner is not None else self.L_inner
        dev = vis.device

        if mask_list is None:
            mask_list = [torch.ones(B, T_l, dtype=torch.bool, device=dev)
                         for T_l in self.seq_lens]

        kv = self._build_kv(vis, state)
        y = self._y_embed(B, dev, indices_list, mask_list)
        rL, rH = self._rhos()                            # learnable scalars in (0,1)
        wL = self._weights(rL, L, dev, y.dtype)          # (L,) closed form
        wH = self._weights(rH, H, dev, y.dtype)          # (H,) closed form

        one_step = self.one_step_grad and self.training and self.update_mode == 'damped'
        z_H = torch.zeros(B, N, self.dim, device=dev, dtype=y.dtype)
        z_L_carry = None
        all_logits = []
        for h in range(H):
            if one_step:
                z_H, z_L_carry = self._outer_one_grad(z_H, y, kv, wL, wH, z_L_carry=z_L_carry, n_inner=L)
            else:
                if self.grad_checkpoint and self.training:
                    z_L = _ckpt(self._inner, z_H, y, kv, wL, use_reentrant=False)
                else:
                    z_L = self._inner(z_H, y, kv, wL)
                g_out_H = self._g_noisy(z_H, z_L + y, kv)
                if self.update_mode == 'damped':
                    alpha_H = wH
                    z_H = (1 - alpha_H) * z_H + alpha_H * g_out_H
                elif self.update_mode == 'bayesian':
                    raise ValueError("update_mode='bayesian' requires STRMPolicyVAE (μ|logprec split); "
                                     "got STRMPolicy (no belief structure). Use --no-vae=False.")
                else:
                    z_H = z_H + wH[h] * g_out_H
            all_logits.append(self._heads(z_H, emb_id=emb_id))
        return all_logits

    # ── Training ──

    def forward_loss(self, target_indices, vis, state, soft_targets=None,
                     n_outer=None, n_inner=None, h_max=None,
                     mask_ratio_max=1.0, emb_id=None, label_smoothing=0.0,
                     mask_sampler='linear'):
        """Random per-level mask, deep-supervision CE over the H cycles.
        Returns (loss, per_level_diag, all_cycle_logits).
        emb_id: per-emb head routing. label_smoothing: ε mass on uniform across non-target codes.
        mask_sampler: 'linear' = U(lo, mask_ratio_max) [legacy curriculum].
                      'cosine' = MaskGIT (Chang et al. 2022): r = cos(π/2·U(0,1)),
                                 clipped to [lo, mask_ratio_max]. Bias toward high mask;
                                 median ≈ 0.71, mean ≈ 0.64; sees hard cases from step 1."""
        B = vis.shape[0]
        dev = vis.device

        masks = []
        for l, T_l in enumerate(self.seq_lens):
            lo = 1.0 / T_l
            # MaskGIT cosine sampling (Chang+ 2022) — the ONLY sampler. r=cos(π/2·U(0,1)),
            # floored at 1/T (≥1 token masked). No curriculum cap, so training masks at the
            # SAME average ratio (~0.64) the val-probe uses → train/val are apples-to-apples.
            u = torch.rand(B, device=dev)
            r = torch.cos(math.pi * u / 2).clamp(min=lo)
            noise = torch.rand(B, T_l, device=dev)
            m = noise < r.unsqueeze(1)
            m[torch.arange(B, device=dev), noise.argmin(1)] = True
            masks.append(m)
        # Realized mask fraction this step — for honest logging. Masking is a FIXED MaskGIT
        # cosine distribution (mean ~0.637); there is NO curriculum. Exposed via _last_mask_frac.
        self._last_mask_frac = float(torch.stack([mm.float().mean() for mm in masks]).mean())

        n_outer_eff = (_random.randint(1, h_max) if h_max is not None else n_outer)

        all_logits = self.forward(target_indices, vis, state,
                                  mask_list=masks,
                                  n_outer=n_outer_eff, n_inner=n_inner, emb_id=emb_id)
        H = len(all_logits)

        total = 0.0
        per_level = [{'mask_correct': 0, 'mask_total': 0, 'loss': 0.0}
                     for _ in self.seq_lens]
        for h_idx, logits_list in enumerate(all_logits):
            cycle_loss = 0.0
            for l in range(self.n_levels):
                target = target_indices[l]
                m = masks[l]
                lp = F.log_softmax(logits_list[l], dim=-1)     # over K real codes
                if soft_targets is not None:
                    ce = -(soft_targets[l].to(lp.dtype) * lp).sum(-1)
                elif label_smoothing > 0.0:
                    target_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    uniform_lp = lp.mean(-1)
                    ce = -((1 - label_smoothing) * target_lp + label_smoothing * uniform_lp)
                else:
                    ce = -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                n_m = m.float().sum(1).clamp(min=1)
                loss_l = ((ce * m.float()).sum(1) / n_m).mean()
                cycle_loss = cycle_loss + loss_l
                if h_idx == H - 1:
                    with torch.no_grad():
                        preds = logits_list[l].argmax(-1)
                        correct = (preds == target) & m
                        per_level[l]['loss'] = loss_l.item()
                        per_level[l]['mask_correct'] = int(correct.float().sum().item())
                        per_level[l]['mask_total']   = int(m.float().sum().item())
            total = total + cycle_loss / self.n_levels
        loss = total / H
        return loss, per_level, all_logits


# ════════════════════════════════════════════════════════════
#  VAE-flavored variant — Gaussian belief latents (split state)
# ════════════════════════════════════════════════════════════

class STRMPolicyVAE(STRMPolicy):
    """Same deterministic additive-decay TRM recurrence, but the latent state
    is a **diagonal Gaussian belief**: the d-wide channel is split into
    `[μ | ρ]`, μ the mean and ρ the **log-precision** (`ρ = log 1/σ²`). The
    recurrence refines the full belief deterministically (μ and ρ accumulate
    via the same `+a_t·g` rule, so the convergence/decay guarantees are
    untouched). At **every supervision point** (each outer cycle) we draw a
    fresh reparameterized sample from the current belief and decode it —
    exactly the deterministic model's per-cycle readout, VAE-flavored.

    Log-precision is the natural additive quantity (precision = information,
    which accumulates as evidence arrives; variance shrinks, so it would be
    backwards to accumulate). It also keeps everything division-free:

        σ = exp(-ρ/2),   z̃ = μ + σ⊙ε
        KL(N(μ,σ²)‖N(0,1)) = ½(exp(-ρ) + μ² - 1 + ρ)        (per dim)

    The state width D is **split** (μ and ρ each D/2), so the sampled latent
    and the head are D/2 — same total parameter budget as the deterministic
    model (fair compute A/B). Init z=0 ⇒ μ=0, ρ=0 ⇒ belief N(0,1) = prior.

    Loss is a per-cycle ELBO averaged over H:
        (1/H) Σ_h [ masked_CE(head(z̃_h)) + β · KL(belief_h) ]
    with free-bits on the KL (per-dim floor `free_bits` nats) to prevent
    posterior collapse. At eval (`self.training=False`) we use the mean
    (ε=0, MAP) for a deterministic probe.
    """
    def __init__(self, *args, beta=1e-3, free_bits=0.1, n_embodiments=1, per_emb_head=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = self.dim // 2
        self.beta = float(beta)
        self.free_bits = float(free_bits)
        self.n_embodiments = int(n_embodiments)
        self.per_emb_head = bool(per_emb_head)
        # head reads the sampled latent (D/2), not the full belief state
        if self.per_emb_head:
            # one head set per embodiment — routed by emb_id at inference
            self.out_head = nn.ModuleList([
                nn.ModuleList([nn.Linear(self.latent_dim, self.k) for _ in range(self.n_levels)])
                for _ in range(self.n_embodiments)
            ])
        else:
            self.out_head = nn.ModuleList([
                nn.Linear(self.latent_dim, self.k) for _ in range(self.n_levels)
            ])

    def _sample_heads(self, z, emb_id=None):
        """Split z=[μ|ρ] on the channel dim, sample z̃=μ+exp(-ρ/2)⊙ε (mean at
        eval), return per-level logits + the belief (μ, ρ) for the KL.

        If per_emb_head is True, emb_id (B,) routes each batch sample through its
        embodiment's head set; otherwise the shared head is used.
        """
        mu      = z[..., :self.latent_dim]
        logprec = z[..., self.latent_dim:]
        std = torch.exp(-0.5 * logprec)
        eps = torch.randn_like(std) if self.training else torch.zeros_like(std)
        zt = mu + std * eps
        offset = 0; logits = []
        for l, T_l in enumerate(self.seq_lens):
            zt_slice = zt[:, offset:offset + T_l, :]                  # (B, T_l, latent_dim)
            if self.per_emb_head and emb_id is not None:
                B = zt_slice.shape[0]
                # Use the dtype of an actual head's output so it matches under autocast (bf16/fp16).
                # Build one sample first to discover the dtype, then allocate.
                first_eid = int(emb_id[0].item())
                if first_eid >= self.n_embodiments: first_eid = 0
                sample = self.out_head[first_eid][l](zt_slice[:1])
                out = torch.zeros(B, T_l, self.k, device=zt.device, dtype=sample.dtype)
                out[0:1] = sample
                # fill the rest, per embodiment
                for eid in emb_id.unique().tolist():
                    if eid >= self.n_embodiments: continue
                    mask = emb_id == eid
                    if mask[0]: mask[0] = False                   # already filled
                    if mask.any():
                        out[mask] = self.out_head[eid][l](zt_slice[mask])
                logits.append(out)
            else:
                head = self.out_head[l] if not self.per_emb_head else self.out_head[0][l]
                logits.append(head(zt_slice))
            offset += T_l
        return logits, mu, logprec

    def forward(self, indices_list, vis, state, mask_list=None,
                n_outer=None, n_inner=None, return_beliefs=False, emb_id=None):
        B = vis.shape[0]
        N = sum(self.seq_lens)
        H = n_outer if n_outer is not None else self.H_outer
        L = n_inner if n_inner is not None else self.L_inner
        dev = vis.device
        if mask_list is None:
            mask_list = [torch.ones(B, T_l, dtype=torch.bool, device=dev)
                         for T_l in self.seq_lens]
        kv = self._build_kv(vis, state)
        y = self._y_embed(B, dev, indices_list, mask_list)
        rL, rH = self._rhos()
        wL = self._weights(rL, L, dev, y.dtype)
        wH = self._weights(rH, H, dev, y.dtype)
        one_step = self.one_step_grad and self.training and self.update_mode == 'damped'
        z_H = torch.zeros(B, N, self.dim, device=dev, dtype=y.dtype)
        z_L_carry = None
        all_logits, all_mu, all_rho = [], [], []
        for h in range(H):
            if one_step:
                z_H, z_L_carry = self._outer_one_grad(z_H, y, kv, wL, wH, z_L_carry=z_L_carry, n_inner=L)
            elif self.grad_checkpoint and self.training:
                z_L = _ckpt(self._inner, z_H, y, kv, wL, use_reentrant=False)
                g_out_H = self._g_noisy(z_H, z_L + y, kv)
                z_H = (self._bayes_fuse(z_H, g_out_H, self.latent_dim) if self.update_mode == 'bayesian'
                       else (1 - wH) * z_H + wH * g_out_H if self.update_mode == 'damped'
                       else z_H + wH[h] * g_out_H)
            else:
                z_L = self._inner(z_H, y, kv, wL)
                g_out_H = self._g_noisy(z_H, z_L + y, kv)
                if self.update_mode == 'bayesian':
                    z_H = self._bayes_fuse(z_H, g_out_H, self.latent_dim)
                elif self.update_mode == 'damped':
                    alpha_H = wH
                    z_H = (1 - alpha_H) * z_H + alpha_H * g_out_H
                else:
                    z_H = z_H + wH[h] * g_out_H
            logits, mu, rho = self._sample_heads(z_H, emb_id=emb_id)  # sampled readout
            all_logits.append(logits)
            if return_beliefs:
                all_mu.append(mu); all_rho.append(rho)
        if return_beliefs:
            return all_logits, all_mu, all_rho
        return all_logits

    def forward_loss(self, target_indices, vis, state, soft_targets=None,
                     n_outer=None, n_inner=None, h_max=None,
                     mask_ratio_max=1.0, emb_id=None, label_smoothing=0.0,
                     mask_sampler='linear'):
        """Per-cycle ELBO: masked-CE on the sampled readout + β·KL(belief),
        averaged over H. Returns (loss, per_level_diag, all_cycle_logits).

        emb_id (B,): routes per-emb heads if per_emb_head is enabled.
        label_smoothing: 0.0 = standard CE; > 0 distributes ε mass uniformly across non-target codes.
        mask_sampler: 'linear' (legacy) or 'cosine' (MaskGIT, see STRMPolicy.forward_loss).
        """
        B = vis.shape[0]
        dev = vis.device
        masks = []
        for l, T_l in enumerate(self.seq_lens):
            lo = 1.0 / T_l
            # MaskGIT cosine sampling (Chang+ 2022) — the ONLY sampler. r=cos(π/2·U(0,1)),
            # floored at 1/T (≥1 token masked). No curriculum cap, so training masks at the
            # SAME average ratio (~0.64) the val-probe uses → train/val are apples-to-apples.
            u = torch.rand(B, device=dev)
            r = torch.cos(math.pi * u / 2).clamp(min=lo)
            noise = torch.rand(B, T_l, device=dev)
            m = noise < r.unsqueeze(1)
            m[torch.arange(B, device=dev), noise.argmin(1)] = True
            masks.append(m)
        # Realized mask fraction this step — for honest logging. Masking is a FIXED MaskGIT
        # cosine distribution (mean ~0.637); there is NO curriculum. Exposed via _last_mask_frac.
        self._last_mask_frac = float(torch.stack([mm.float().mean() for mm in masks]).mean())

        n_outer_eff = (_random.randint(1, h_max) if h_max is not None else n_outer)
        all_logits, all_mu, all_rho = self.forward(
            target_indices, vis, state, mask_list=masks,
            n_outer=n_outer_eff, n_inner=n_inner, return_beliefs=True, emb_id=emb_id)
        H = len(all_logits)

        total = 0.0
        kl_total = 0.0
        per_level = [{'mask_correct': 0, 'mask_total': 0, 'loss': 0.0}
                     for _ in self.seq_lens]
        for h_idx in range(H):
            logits_list = all_logits[h_idx]
            cycle_loss = 0.0
            for l in range(self.n_levels):
                target = target_indices[l]
                m = masks[l]
                lp = F.log_softmax(logits_list[l], dim=-1)
                if soft_targets is not None:
                    ce = -(soft_targets[l].to(lp.dtype) * lp).sum(-1)
                elif label_smoothing > 0.0:
                    # smoothed CE: (1-ε)·log p(target) + ε·mean(log p) — distributes ε mass uniformly
                    target_lp = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                    uniform_lp = lp.mean(-1)
                    ce = -((1 - label_smoothing) * target_lp + label_smoothing * uniform_lp)
                else:
                    ce = -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                n_m = m.float().sum(1).clamp(min=1)
                loss_l = ((ce * m.float()).sum(1) / n_m).mean()
                cycle_loss = cycle_loss + loss_l
                if h_idx == H - 1:
                    with torch.no_grad():
                        preds = logits_list[l].argmax(-1)
                        correct = (preds == target) & m
                        per_level[l]['loss'] = loss_l.item()
                        per_level[l]['mask_correct'] = int(correct.float().sum().item())
                        per_level[l]['mask_total']   = int(m.float().sum().item())
            total = total + cycle_loss / self.n_levels
            # KL of this cycle's belief vs N(0,1), free-bits floor per dim.
            mu, rho = all_mu[h_idx], all_rho[h_idx]
            kl = 0.5 * (torch.exp(-rho) + mu * mu - 1.0 + rho)      # (B,N,latent)
            kl = kl.clamp(min=self.free_bits).sum(-1).mean()        # per-token, free-bits
            kl_total = kl_total + kl

        loss = total / H + self.beta * (kl_total / H)
        return loss, per_level, all_logits


if __name__ == '__main__':
    import sys, time
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dim   = int(sys.argv[2]) if len(sys.argv) > 2 else 768
    L_in  = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    H_out = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = STRMPolicy(seq_lens=(4, 8, 16), k_codebook=128, dim=dim, heads=8,
                   depth=depth, L_inner=L_in, H_outer=H_out, state_dim=8).to(dev)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    rL, rH = m._rhos()
    print(f"depth={depth} dim={dim} L={L_in} H={H_out}  params={n:.2f}M")
    print(f"  ρ_L={rL.item():.3f}  ρ_H={rH.item():.3f}  (learnable; weights ρ^(t/(n-1)))")
    B = 2
    vis = torch.randn(B, 128, dim, device=dev)
    state = torch.randn(B, 8, device=dev)
    targets = [torch.randint(0, 128, (B, t), device=dev) for t in m.seq_lens]
    loss, _, all_logits = m.forward_loss(targets, vis, state)
    print(f"  cycles={len(all_logits)}  logits[0][0]={tuple(all_logits[0][0].shape)}")
    loss.backward()
    print(f"forward+backward OK  loss={loss.item():.3f}")
    # logit magnitude should now CHANGE with step count (un-normalized accum)
    m.eval()
    with torch.no_grad():
        for L in (2, 5, 20, 50):
            out = m(None, vis, state, mask_list=None, n_outer=4, n_inner=L)
            z_norm = out[-1][0].abs().mean().item()
            print(f"  L={L:>3}: |logits| mean={z_norm:.3f}  (grows with L; finite each)")
