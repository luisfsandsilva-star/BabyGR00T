"""S-TRM v3 policy over the 3-level CQ-VAE action codes.

Architecture (see docs/ARCHITECTURE.md for the full derivation):

  Two state streams x1 (slow), x2 (fast), shape (B, N, d), N = sum(seq_lens).

  Inner step (run L times):
      x1 = Ā₁ ⊙ x1 + (1/L) F(x2 + y_embed,  vis_state)
      x2 = Ā₂ ⊙ x2 + (1/L) G(x1 + y_embed)

  - Ā = exp(-|Δ| ⊙ exp(Ã))                (Parcae; Ā ∈ (0,1)^d per channel).
  - F: ScaleNorm → SelfAttn (QK-norm + softmax-1 sink) → conv-combo,
       then ScaleNorm → CrossAttn (QK-norm + softmax-1 sink) → conv-combo.
       Cross-attention KV: vision prefix (resampler latents) + state token.
  - G: ScaleNorm → GeGLU → conv-combo.
  - Convex-combo residual at every sublayer:  h' = (1-α) h + α f(h),
    α = sigmoid(α_logit), init at sigmoid ≈ 0.9.
  - Every Linear is wrapped in Miyato spectral normalization (‖W‖₂=1).

  Outer H cycles, each with its own deep-supervision loss.
  Between cycles, y_embed is updated with an outer-loop Parcae:
      y_embed ← Ā_H ⊙ y_embed + (1/H) candidate
    where the candidate is built from masked positions' soft predictions and
    unmasked positions' GT (teacher-forced training).

  No [MASK] token: masked positions start at a single learned e_init vector,
  differentiated by per-level + per-position embeddings.
"""
import math
import random as _random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as _sn
from torch.utils.checkpoint import checkpoint as _ckpt


def _snlin(in_dim, out_dim, bias=False):
    """Linear with Miyato spectral normalization (‖W‖₂ = 1, σ_max = 1)."""
    return _sn(nn.Linear(in_dim, out_dim, bias=bias))


# ── Building blocks ──

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.eps = eps

    def forward(self, x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=self.eps) * self.g


def _attn_with_sink(q, k, v, dropout_p=0.0):
    """Softmax-1 attention via a zero key+value sink token (Miller 2023).
    Lets a head abstain by dumping mass onto a no-op slot, instead of
    spreading attention across real keys.
    """
    B, H, _, Hd = k.shape
    zero = torch.zeros(B, H, 1, Hd, device=k.device, dtype=k.dtype)
    k = torch.cat([k, zero], dim=-2)
    v = torch.cat([v, zero], dim=-2)
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.drop_p = dropout
        self.wq = _snlin(dim, dim)
        self.wk = _snlin(dim, dim)
        self.wv = _snlin(dim, dim)
        self.wo = _snlin(dim, dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        H, Hd = self.heads, self.head_dim
        q = self.wq(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.wk(x).view(B, T, H, Hd).transpose(1, 2)
        v = self.wv(x).view(B, T, H, Hd).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k)
        o = _attn_with_sink(q, k, v,
                            dropout_p=self.drop_p if self.training else 0.0)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.drop_p = dropout
        self.wq = _snlin(dim, dim)
        self.wk = _snlin(dim, dim)
        self.wv = _snlin(dim, dim)
        self.wo = _snlin(dim, dim)
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
        o = _attn_with_sink(q, k, v,
                            dropout_p=self.drop_p if self.training else 0.0)
        return self.wo(o.transpose(1, 2).reshape(B, T, D))


class GeGLU(nn.Module):
    """GeGLU FFN (Shazeer 2020): GeLU(W1 x) ⊙ (W2 x) → W3."""
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.w1 = _snlin(dim, hidden)
        self.w2 = _snlin(dim, hidden)
        self.w3 = _snlin(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w3(F.gelu(self.w1(x)) * self.w2(x)))


def _alpha_logit(target):
    return math.log(target / (1 - target))


class FBlock(nn.Module):
    """F: SelfAttn (conv-combo) ∘ CrossAttn (conv-combo)."""
    def __init__(self, dim, heads, alpha_init=0.9, dropout=0.0):
        super().__init__()
        self.sa_norm = ScaleNorm(dim)
        self.sa = SelfAttention(dim, heads, dropout=dropout)
        self.alpha_sa = nn.Parameter(torch.tensor(_alpha_logit(alpha_init)))

        self.ca_norm = ScaleNorm(dim)
        self.ca = CrossAttention(dim, heads, dropout=dropout)
        self.alpha_ca = nn.Parameter(torch.tensor(_alpha_logit(alpha_init)))

    def forward(self, h, kv):
        h_sa = self.sa(self.sa_norm(h))
        a = torch.sigmoid(self.alpha_sa)
        h = (1 - a) * h + a * h_sa
        h_ca = self.ca(self.ca_norm(h), kv)
        a = torch.sigmoid(self.alpha_ca)
        h = (1 - a) * h + a * h_ca
        return h


class GBlock(nn.Module):
    """G: GeGLU FFN (conv-combo)."""
    def __init__(self, dim, hidden, alpha_init=0.9, dropout=0.0):
        super().__init__()
        self.norm = ScaleNorm(dim)
        self.ffn = GeGLU(dim, hidden, dropout=dropout)
        self.alpha = nn.Parameter(torch.tensor(_alpha_logit(alpha_init)))

    def forward(self, h):
        h_ffn = self.ffn(self.norm(h))
        a = torch.sigmoid(self.alpha)
        return (1 - a) * h + a * h_ffn


# ════════════════════════════════════════════════════════════
#  Policy
# ════════════════════════════════════════════════════════════

class STRMPolicy(nn.Module):
    """S-TRM over [vis | state | L0 | L1 | L2] — vis+state via cross-attn.

    seq_lens : per-level token counts (default (4, 8, 16) for the 3-level CQ-VAE).
    Each level has its own codebook embedding E_l (K, d) and output head W_l.

    Recipe knobs (v3 defaults):
      depth=2, dim=768, heads=8
      L_inner=5, H_outer=4 (training-time fixed; can sample h_max>=1 in forward_loss)
      ρ1=0.75, ρ2=0.65, ρ_H=0.85
      alpha_init=0.9 (convex residual gate)
      grad_checkpoint=True (per H cycle — keeps memory O(one cycle))
    """
    def __init__(self, seq_lens=(4, 8, 16), k_codebook=128,
                 dim=768, heads=8, ff_hidden=None, depth=2,
                 L_inner=5, H_outer=4,
                 rho1_target=0.75, rho2_target=0.65, rho_H_target=0.85,
                 alpha_init=0.9, dropout=0.0,
                 max_prefix=160, state_dim=6,
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

        if ff_hidden is None:
            ff_hidden = (int(dim * 8 / 3) + 63) // 64 * 64

        # Per-level codebook embeddings (no MASK row — soft embeddings instead)
        self.tok_emb = nn.ModuleList([
            nn.Embedding(k_codebook, dim) for _ in range(self.n_levels)
        ])
        self.level_emb = nn.Parameter(torch.randn(self.n_levels, dim) * 0.02)
        self.pos_emb = nn.ParameterList([
            nn.Parameter(torch.randn(t, dim) * 0.02) for t in self.seq_lens
        ])
        self.e_init = nn.Parameter(torch.randn(dim) * 0.02)

        # Vision prefix + state → cross-attn KV
        self.prefix_pos_emb = nn.Parameter(torch.randn(max_prefix, dim) * 0.02)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, dim), nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.state_pos_emb = nn.Parameter(torch.randn(1, dim) * 0.02)

        # Parcae per-channel parameters (Ā₁, Ā₂, Ā_H).
        # Init so Ā = exp(-|Δ|·exp(Ã)) hits the target ρ values channel-wise.
        delta1_init  = -math.log(rho1_target)
        delta2_init  = -math.log(rho2_target)
        delta_H_init = -math.log(rho_H_target)
        self.tilde_A1 = nn.Parameter(torch.zeros(dim))
        self.Delta1   = nn.Parameter(torch.full((dim,), float(delta1_init)))
        self.tilde_A2 = nn.Parameter(torch.zeros(dim))
        self.Delta2   = nn.Parameter(torch.full((dim,), float(delta2_init)))
        self.tilde_A_H = nn.Parameter(torch.zeros(dim))
        self.Delta_H   = nn.Parameter(torch.full((dim,), float(delta_H_init)))

        # depth × FBlock + depth × GBlock — applied in sequence each inner step
        self.f_blocks = nn.ModuleList([
            FBlock(dim, heads, alpha_init=alpha_init, dropout=dropout)
            for _ in range(depth)
        ])
        self.g_blocks = nn.ModuleList([
            GBlock(dim, ff_hidden, alpha_init=alpha_init, dropout=dropout)
            for _ in range(depth)
        ])

        # Output head (deep supervision over H, final head reads x1)
        self.out_norm = ScaleNorm(dim)
        self.out_head = nn.ModuleList([
            _snlin(dim, k_codebook) for _ in range(self.n_levels)
        ])

    # ── Helpers ──

    def _A_bar(self):
        A1 = torch.exp(-self.Delta1.abs() * torch.exp(self.tilde_A1))
        A2 = torch.exp(-self.Delta2.abs() * torch.exp(self.tilde_A2))
        return A1, A2

    def _A_H(self):
        return torch.exp(-self.Delta_H.abs() * torch.exp(self.tilde_A_H))

    def _build_kv(self, vis, state):
        B, P, _ = vis.shape
        vis_p = vis + self.prefix_pos_emb[:P].unsqueeze(0).to(vis.dtype)
        st = self.state_proj(state).unsqueeze(1) + self.state_pos_emb.unsqueeze(0)
        return torch.cat([vis_p, st], dim=1)

    def _y_embed(self, B, dev, indices_list, mask_list, logits_list=None):
        """Build (B, N, d) soft-embedding tensor.
        - logits_list None  → first cycle: e_init at masked, E[gt] at unmasked.
        - logits_list given → next cycle: Σ p·E at masked, E[gt] at unmasked.
        At inference (no GT), set indices_list=None — all positions go via the
        soft path.
        """
        outs = []
        for l, T_l in enumerate(self.seq_lens):
            if logits_list is None:
                soft_emb = self.e_init.view(1, 1, -1).expand(B, T_l, -1)
            else:
                p = F.softmax(logits_list[l], dim=-1)
                soft_emb = p @ self.tok_emb[l].weight

            if indices_list is not None and mask_list is not None:
                gt_emb = self.tok_emb[l](indices_list[l])
                m = mask_list[l].unsqueeze(-1)
                emb = torch.where(m, soft_emb, gt_emb)
            else:
                emb = soft_emb

            emb = (emb + self.level_emb[l].view(1, 1, -1)
                       + self.pos_emb[l].unsqueeze(0))
            outs.append(emb)
        return torch.cat(outs, dim=1)

    def _heads(self, x1):
        z = self.out_norm(x1)
        offset = 0
        out = []
        for l, T_l in enumerate(self.seq_lens):
            z_l = z[:, offset:offset + T_l, :]
            offset += T_l
            out.append(self.out_head[l](z_l))
        return out

    def _inner_loop(self, x1, x2, y_embed, kv, L):
        A1, A2 = self._A_bar()
        a1 = A1.view(1, 1, -1); a2 = A2.view(1, 1, -1)
        for _ in range(L):
            h = x2 + y_embed
            for fb in self.f_blocks:
                h = fb(h, kv)
            x1 = a1 * x1 + h / L
            h = x1 + y_embed
            for gb in self.g_blocks:
                h = gb(h)
            x2 = a2 * x2 + h / L
        return x1, x2

    # ── Forward ──

    def forward(self, indices_list, vis, state, mask_list=None,
                n_outer=None, n_inner=None):
        """Run H outer cycles, each with L inner steps. Returns a list of length
        H, each a list of per-level (B, T_l, K) logits.

        - `n_outer`/`n_inner` per-call overrides; do NOT mutate self.{H,L}_inner.
        - `mask_list=None` ⇒ all positions masked (cold eval / generation).
        """
        B = vis.shape[0]
        N = sum(self.seq_lens)
        H = n_outer if n_outer is not None else self.H_outer
        L = n_inner if n_inner is not None else self.L_inner
        dev = vis.device

        if mask_list is None:
            mask_list = [
                torch.ones(B, T_l, dtype=torch.bool, device=dev)
                for T_l in self.seq_lens
            ]

        kv = self._build_kv(vis, state)
        y_embed = self._y_embed(B, dev, indices_list, mask_list, logits_list=None)
        x1 = torch.zeros(B, N, self.dim, device=dev, dtype=y_embed.dtype)
        x2 = torch.zeros(B, N, self.dim, device=dev, dtype=y_embed.dtype)

        a_H = self._A_H().view(1, 1, -1)

        all_logits = []
        for h in range(H):
            if self.grad_checkpoint and self.training:
                x1, x2 = _ckpt(self._inner_loop, x1, x2, y_embed, kv, L,
                               use_reentrant=False)
            else:
                x1, x2 = self._inner_loop(x1, x2, y_embed, kv, L=L)
            logits_list = self._heads(x1)
            all_logits.append(logits_list)
            if h < H - 1:
                candidate = self._y_embed(B, dev, indices_list, mask_list,
                                          logits_list=logits_list)
                y_embed = a_H * y_embed + candidate / H
        return all_logits

    # ── Training (random mask per level + deep-supervision loss over H) ──

    def forward_loss(self, target_indices, vis, state, soft_targets=None,
                     n_outer=None, n_inner=None, h_max=None,
                     mask_ratio_max=1.0):
        """- `h_max`: if given, sample H ~ Uniform{1..h_max} (stochastic depth).
                     v3 trains with h_max=12 and beats fixed-H at deployment time.
        - `mask_ratio_max`: upper bound for the per-batch mask-ratio sample.
                            v3 uses a curriculum 0.3 → 1.0 over the first 50%.

        Returns (loss, per_level_diag, all_cycle_logits) where
        `all_cycle_logits` is a list of length H_used; each entry is the per-
        level [(B,T_l,K)] logits at that cycle. The trainer can pick the final
        cycle (`all_cycle_logits[-1]`) for a final-only auxiliary loss, or
        iterate the full list for an all-cycles auxiliary loss.
        """
        B = vis.shape[0]
        dev = vis.device

        # Random mask ratio per level, ≥1 masked. Uniform [1/T_l, mask_ratio_max].
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
                                  n_outer=n_outer_eff,
                                  n_inner=n_inner)
        H = len(all_logits)

        total = 0.0
        per_level = [{'mask_correct': 0, 'mask_total': 0, 'loss': 0.0}
                     for _ in self.seq_lens]
        for h_idx, logits_list in enumerate(all_logits):
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
        loss = total / H
        return loss, per_level, all_logits


if __name__ == '__main__':
    import sys, time
    depth = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    dim   = int(sys.argv[2]) if len(sys.argv) > 2 else 768
    L_in  = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    H_out = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = STRMPolicy(seq_lens=(4, 8, 16), k_codebook=128, dim=dim,
                   heads=8, depth=depth,
                   L_inner=L_in, H_outer=H_out).to(dev)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"depth={depth}  dim={dim}  L={L_in}  H={H_out}")
    print(f"params: {n:.2f}M")
    A1, A2 = m._A_bar()
    A_H = m._A_H()
    print(f"  ρ(Ā1) mean={A1.mean().item():.3f}  ρ(Ā2) mean={A2.mean().item():.3f}  "
          f"ρ(Ā_H) mean={A_H.mean().item():.3f}")
    B = 2
    vis = torch.randn(B, 128, dim, device=dev)
    state = torch.randn(B, 6, device=dev)
    targets = [torch.randint(0, 128, (B, t), device=dev) for t in m.seq_lens]
    if dev == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss, _, all_logits = m.forward_loss(targets, vis, state)
    print(f"  cycles returned: {len(all_logits)}  shapes: {[lg[0].shape for lg in all_logits[:1]]}...")
    loss.backward()
    if dev == 'cuda':
        torch.cuda.synchronize()
    print(f"forward+backward OK in {(time.perf_counter()-t0)*1000:.0f} ms  "
          f"loss={loss.item():.3f}")
