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
pass-through) — the accumulation is the residual stream. The input to `g`
is the accumulated-so-far latent plus the usual TRM conditioning
(`z_other`, the mask/GT embedding `y`, and vision/state via cross-attention).

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

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.eps = eps

    def forward(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=self.eps) * self.g


class SelfAttention(nn.Module):
    """Vanilla MHSA with QK-norm (plain softmax, no sink)."""
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.drop_p = dropout
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        H, Hd = self.heads, self.head_dim
        q = self.wq(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.wk(x).view(B, T, H, Hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, Hd).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    """Vanilla MHCA with QK-norm (vision/state KV)."""
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.drop_p = dropout
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x, kv):
        B, T, D = x.shape
        _, M, _ = kv.shape
        H, Hd = self.heads, self.head_dim
        q = self.wq(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.wk(kv).view(B, M, H, Hd).transpose(1, 2)
        v = self.wv(kv).view(B, M, H, Hd).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop_p if self.training else 0.0)
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
    """The shared tiny net `g`. `depth` pre-norm sub-blocks (SelfAttn →
    CrossAttn → GeGLU). Returns the **sum of the sub-layer transforms** —
    a pure update direction, no identity pass-through (the accumulation in
    the policy is the residual stream).
    """
    def __init__(self, dim, heads, ff_hidden, depth=2, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'sa_norm': ScaleNorm(dim), 'sa': SelfAttention(dim, heads, dropout),
                'ca_norm': ScaleNorm(dim), 'ca': CrossAttention(dim, heads, dropout),
                'ff_norm': ScaleNorm(dim), 'ff': GeGLU(dim, ff_hidden, dropout),
            }) for _ in range(depth)
        ])

    def forward(self, u, kv):
        h = u
        out = torch.zeros_like(u)
        for blk in self.blocks:
            s = blk['sa'](blk['sa_norm'](h)); h = h + s; out = out + s
            c = blk['ca'](blk['ca_norm'](h), kv); h = h + c; out = out + c
            f = blk['ff'](blk['ff_norm'](h)); h = h + f; out = out + f
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
                 dim=768, heads=8, ff_hidden=None, depth=2,
                 L_inner=5, H_outer=4,
                 rho_L=0.1, rho_H=0.1,
                 dropout=0.0, max_prefix=160, state_dim=6,
                 grad_checkpoint=True):
        super().__init__()
        self.seq_lens = list(seq_lens)
        self.n_levels = len(self.seq_lens)
        self.dim = dim
        self.k = k_codebook
        self.L_inner = L_inner
        self.H_outer = H_outer
        self.depth = depth
        self.grad_checkpoint = grad_checkpoint
        self.mask_idx = k_codebook                       # MASK input marker
        # Decay rates ρ_L, ρ_H ∈ (0,1) — single learnable scalars (sigmoid).
        # Only the rate is learned; the closed-form shape a_t=ρ^(t/(n-1)) and
        # its decaying/bounded guarantee hold for every ρ ∈ (0,1).
        _logit = lambda p: math.log(p / (1 - p))
        self.rho_L_raw = nn.Parameter(torch.tensor(float(_logit(rho_L))))
        self.rho_H_raw = nn.Parameter(torch.tensor(float(_logit(rho_H))))

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
        self.g = TRMNet(dim, heads, ff_hidden, depth=depth, dropout=dropout)

        # Read-out head (plain Linear over the K real codes; deep supervision).
        # NO norm on the head — it reads the raw accumulated latent so logit
        # magnitude is free to grow as the latent sharpens. The decay lives
        # only in the z-update rule.
        self.out_head = nn.ModuleList([
            nn.Linear(dim, k_codebook) for _ in range(self.n_levels)
        ])

    # ── Helpers ──

    def _rhos(self):
        return torch.sigmoid(self.rho_L_raw), torch.sigmoid(self.rho_H_raw)

    @staticmethod
    def _weights(rho, n, device, dtype):
        """Closed-form refinement weights a_t = ρ^{t/(n-1)} = ρ^{linspace(0,1,n)}.
        Un-normalized; a_0 = 1, a_{n-1} = ρ exactly; the profile in loop-fraction
        t/(n-1) is identical for any n. n=1 → [1.0]. ρ may be a learnable tensor
        (gradient flows)."""
        expo = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        if not torch.is_tensor(rho):
            rho = torch.tensor(float(rho), device=device, dtype=dtype)
        return rho.to(device=device, dtype=dtype) ** expo

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

    def _heads(self, z):
        # raw latent → head (no norm): logit magnitude tracks the latent.
        offset = 0
        out = []
        for l, T_l in enumerate(self.seq_lens):
            out.append(self.out_head[l](z[:, offset:offset + T_l, :]))
            offset += T_l
        return out

    def _inner(self, z_H, y, kv, wL):
        """Inner loop: z_L accumulates the decayed transforms (fresh each
        outer cycle). wL = (L,) normalized geometric weights."""
        z_L = torch.zeros_like(y)
        for t in range(wL.shape[0]):
            z_L = z_L + wL[t] * self.g(z_L + z_H + y, kv)
        return z_L

    # ── Forward ──

    def forward(self, indices_list, vis, state, mask_list=None,
                n_outer=None, n_inner=None):
        """Run H outer cycles, each with L inner recursions. Returns a list of
        length H, each a list of per-level (B, T_l, K) logits."""
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

        z_H = torch.zeros(B, N, self.dim, device=dev, dtype=y.dtype)
        all_logits = []
        for h in range(H):
            if self.grad_checkpoint and self.training:
                z_L = _ckpt(self._inner, z_H, y, kv, wL, use_reentrant=False)
            else:
                z_L = self._inner(z_H, y, kv, wL)
            z_H = z_H + wH[h] * self.g(z_H + z_L + y, kv)
            all_logits.append(self._heads(z_H))
        return all_logits

    # ── Training ──

    def forward_loss(self, target_indices, vis, state, soft_targets=None,
                     n_outer=None, n_inner=None, h_max=None,
                     mask_ratio_max=1.0):
        """Random per-level mask, deep-supervision CE over the H cycles.
        Returns (loss, per_level_diag, all_cycle_logits)."""
        B = vis.shape[0]
        dev = vis.device

        masks = []
        for l, T_l in enumerate(self.seq_lens):
            lo = 1.0 / T_l
            hi = max(lo, mask_ratio_max)
            r = torch.rand(B, device=dev) * (hi - lo) + lo
            noise = torch.rand(B, T_l, device=dev)
            m = noise < r.unsqueeze(1)
            m[torch.arange(B, device=dev), noise.argmin(1)] = True
            masks.append(m)

        n_outer_eff = (_random.randint(1, h_max) if h_max is not None else n_outer)

        all_logits = self.forward(target_indices, vis, state,
                                  mask_list=masks,
                                  n_outer=n_outer_eff, n_inner=n_inner)
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
    def __init__(self, *args, beta=1e-3, free_bits=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_dim = self.dim // 2
        self.beta = float(beta)
        self.free_bits = float(free_bits)
        # head reads the sampled latent (D/2), not the full belief state
        self.out_head = nn.ModuleList([
            nn.Linear(self.latent_dim, self.k) for _ in range(self.n_levels)
        ])

    def _sample_heads(self, z):
        """Split z=[μ|ρ] on the channel dim, sample z̃=μ+exp(-ρ/2)⊙ε (mean at
        eval), return per-level logits + the belief (μ, ρ) for the KL."""
        mu      = z[..., :self.latent_dim]
        logprec = z[..., self.latent_dim:]
        std = torch.exp(-0.5 * logprec)
        eps = torch.randn_like(std) if self.training else torch.zeros_like(std)
        zt = mu + std * eps
        offset = 0; logits = []
        for l, T_l in enumerate(self.seq_lens):
            logits.append(self.out_head[l](zt[:, offset:offset + T_l, :]))
            offset += T_l
        return logits, mu, logprec

    def forward(self, indices_list, vis, state, mask_list=None,
                n_outer=None, n_inner=None, return_beliefs=False):
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
        z_H = torch.zeros(B, N, self.dim, device=dev, dtype=y.dtype)
        all_logits, all_mu, all_rho = [], [], []
        for h in range(H):
            if self.grad_checkpoint and self.training:
                z_L = _ckpt(self._inner, z_H, y, kv, wL, use_reentrant=False)
            else:
                z_L = self._inner(z_H, y, kv, wL)
            z_H = z_H + wH[h] * self.g(z_H + z_L + y, kv)   # deterministic belief
            logits, mu, rho = self._sample_heads(z_H)        # sampled readout
            all_logits.append(logits)
            if return_beliefs:
                all_mu.append(mu); all_rho.append(rho)
        if return_beliefs:
            return all_logits, all_mu, all_rho
        return all_logits

    def forward_loss(self, target_indices, vis, state, soft_targets=None,
                     n_outer=None, n_inner=None, h_max=None,
                     mask_ratio_max=1.0):
        """Per-cycle ELBO: masked-CE on the sampled readout + β·KL(belief),
        averaged over H. Returns (loss, per_level_diag, all_cycle_logits)."""
        B = vis.shape[0]
        dev = vis.device
        masks = []
        for l, T_l in enumerate(self.seq_lens):
            lo = 1.0 / T_l
            hi = max(lo, mask_ratio_max)
            r = torch.rand(B, device=dev) * (hi - lo) + lo
            noise = torch.rand(B, T_l, device=dev)
            m = noise < r.unsqueeze(1)
            m[torch.arange(B, device=dev), noise.argmin(1)] = True
            masks.append(m)

        n_outer_eff = (_random.randint(1, h_max) if h_max is not None else n_outer)
        all_logits, all_mu, all_rho = self.forward(
            target_indices, vis, state, mask_list=masks,
            n_outer=n_outer_eff, n_inner=n_inner, return_beliefs=True)
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
