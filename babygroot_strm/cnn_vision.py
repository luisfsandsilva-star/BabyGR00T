"""Efficient-but-capable CNN vision encoder.

Replaces the InternVL3-cache → LayerAggregator → PerceiverResampler pathway,
which collapsed to an input-independent constant (see scripts/diag_vision.py:
shuffled-vision Δlogit ≈ 0%, resampler output std/|mean| ≈ 0). This encoder is
trained end-to-end with the policy so the vision branch actually carries
image-dependent signal.

Design (all the requested ingredients):
  • patchify stem (4×4 stride-4 conv)
  • inverted-residual blocks: DEPTHWISE-SEPARABLE spatial mixing (depthwise k×k
    + pointwise 1×1) with a CONV-GLU (GeGLU) gate on the expanded channels
  • RESIDUAL connections, pre-norm with SCALENORM (channel-wise L2 × learned g)
  • strided-conv downsampling between 4 stages
  • outputs (B, n_tokens, out_dim) spatial tokens for the policy cross-attention
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixStyle(nn.Module):
    """MixStyle (Zhou et al., ICLR 2021) — parameter-free texture/style randomizer.

    Style lives in per-instance channel statistics. This normalizes each sample's
    channel mean/std, then re-applies a Beta-interpolated blend with a permuted
    sample's stats → randomizes style while preserving spatial content. Inserted in
    the EARLY CNN stages (where style dominates). Training-only; identity in eval().
    Complements APR (which randomizes the FFT amplitude at the input)."""
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):                        # (B, C, H, W)
        if not self.training or self.p <= 0.0 or x.size(0) < 2 or random.random() > self.p:
            return x
        B = x.size(0)
        mu  = x.mean(dim=(2, 3), keepdim=True)
        sig = (x.var(dim=(2, 3), keepdim=True) + self.eps).sqrt()
        x_n = (x - mu) / sig
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device, x.dtype)
        perm = torch.randperm(B, device=x.device)
        mu_mix  = lam * mu  + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_n * sig_mix + mu_mix


class ScaleNorm2d(nn.Module):
    """ScaleNorm for NCHW: per-location L2-normalize over channels, × learned g."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1) * (dim ** 0.5))
        self.eps = eps

    def forward(self, x):                       # (B, C, H, W)
        return x / x.norm(dim=1, keepdim=True).clamp(min=self.eps) * self.g


class LayerNorm2d(nn.Module):
    """Channels-last LayerNorm for NCHW (centers + per-channel affine; ConvNeXt-style)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):                       # (B, C, H, W)
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def _make_norm(dim, kind):
    return LayerNorm2d(dim) if kind == 'layernorm' else ScaleNorm2d(dim)


class ConvGLUBlock(nn.Module):
    """Inverted-residual block, pre-norm:
        x → ScaleNorm → depthwise(k×k) → pointwise(C→2H) → GeGLU(a·gelu(b))
          → pointwise(H→C) → + residual
    Depthwise-separable (cheap spatial mixing) + ConvGLU gate (expressive)."""
    def __init__(self, dim, expand=3, k=7, dropout=0.0, norm='scalenorm'):
        super().__init__()
        h = int(dim * expand)
        self.norm   = _make_norm(dim, norm)
        self.dw     = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim)   # depthwise
        self.pw_in  = nn.Conv2d(dim, 2 * h, 1)                             # expand (×2 for GLU)
        self.pw_out = nn.Conv2d(h, dim, 1)                                 # project back
        self.drop   = nn.Dropout2d(dropout)

    def forward(self, x):
        r = x
        x = self.norm(x)
        x = self.dw(x)
        a, b = self.pw_in(x).chunk(2, dim=1)
        x = a * F.gelu(b)                        # ConvGLU / GeGLU
        x = self.pw_out(x)
        return r + self.drop(x)


class Downsample(nn.Module):
    """ScaleNorm → strided 2×2 conv (halves H,W, changes channels)."""
    def __init__(self, cin, cout, norm='scalenorm'):
        super().__init__()
        self.norm = _make_norm(cin, norm)
        self.conv = nn.Conv2d(cin, cout, 2, stride=2)

    def forward(self, x):
        return self.conv(self.norm(x))


class EfficientCNN(nn.Module):
    """Patchify stem → 4 stages of ConvGLU blocks (with downsampling) → tokens.

    For img_size=224: 224 →stem/4→ 56 → 28 → 14 → 7, i.e. 7×7 = 49 tokens.
    Returns (B, n_tokens, out_dim) plus the token grid (h, w).

    Optional embodiment-conditioned FiLM (`n_embodiments > 0`): each emb has its own
    (γ, β) per stage, applied right after the stage's ConvGLU blocks. Identity-
    initialized (γ=0, β=0 → effective `out = stage·1 + 0`) so adding FiLM at init
    leaves the network unchanged. Same pattern as RT-1's FiLM-on-EfficientNet, but
    conditioned on robot identity (embodiment) instead of language.
    """
    def __init__(self, in_ch=3, dims=(64, 128, 256, 384), depths=(2, 2, 4, 2),
                 expand=3, kernel=7, out_dim=384, dropout=0.0, norm='scalenorm',
                 pos_emb=False, img_size=224, n_embodiments=0, mixstyle_p=0.0,
                 mixstyle_stages=2,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, dims[0], 4, stride=4)
        # MixStyle in the early stages (style lives in early-layer channel stats).
        # Parameter-free → no state_dict change; identity in eval() → inference unaffected.
        self.mixstyle = MixStyle(mixstyle_p) if mixstyle_p > 0 else None
        self.mixstyle_stages = mixstyle_stages
        self.stages, self.downs = nn.ModuleList(), nn.ModuleList()
        for i, (d, n) in enumerate(zip(dims, depths)):
            self.stages.append(nn.Sequential(
                *[ConvGLUBlock(d, expand, kernel, dropout, norm) for _ in range(n)]))
            if i < len(dims) - 1:
                self.downs.append(Downsample(d, dims[i + 1], norm))
        self.out_norm = _make_norm(dims[-1], norm)
        self.proj = (nn.Identity() if dims[-1] == out_dim
                     else nn.Conv2d(dims[-1], out_dim, 1))
        self.out_dim = out_dim
        # learned 2D positional embedding over the token grid (stem/4 then 3× /2 = /32)
        g = img_size // 32
        self.pos_emb = nn.Parameter(torch.zeros(1, g * g, out_dim)) if pos_emb else None
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))

        # Optional embodiment-FiLM: per-emb (γ, β) per stage. Identity-init.
        self.dims = list(dims)
        self.n_embodiments = int(n_embodiments)
        if self.n_embodiments > 0:
            total = sum(self.dims) * 2                                # γ + β per stage
            self.film = nn.Embedding(self.n_embodiments, total)
            nn.init.zeros_(self.film.weight)                          # identity init
        else:
            self.film = None

    def _apply_film(self, x, stage_idx, emb_id):
        """x: (B, C, H, W); emb_id: (B,) long. Modulate channels per-sample."""
        if self.film is None or emb_id is None: return x
        gb = self.film(emb_id)                                        # (B, 2·sum(dims))
        offset = 2 * sum(self.dims[:stage_idx])
        d = self.dims[stage_idx]
        g = gb[:, offset      : offset + d].view(-1, d, 1, 1)
        b = gb[:, offset + d  : offset + 2*d].view(-1, d, 1, 1)
        return x * (1.0 + g) + b

    def forward(self, x, normalize=True, emb_id=None):
        """x: (B, 3, H, W) in [0,1]. Returns (B, h*w, out_dim), (h, w).
        emb_id: optional (B,) long tensor — if FiLM is enabled (n_embodiments > 0),
        modulates each stage's channels by per-embodiment (γ, β)."""
        if normalize:
            x = (x - self.mean) / self.std
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            x = self._apply_film(x, i, emb_id)                        # ← FiLM here
            if self.mixstyle is not None and i < self.mixstyle_stages:
                x = self.mixstyle(x)                                  # ← style randomization
            if i < len(self.downs):
                x = self.downs[i](x)
        x = self.proj(self.out_norm(x))         # (B, out_dim, h, w)
        B, C, h, w = x.shape
        tok = x.flatten(2).transpose(1, 2)       # (B, h*w, out_dim)
        if self.pos_emb is not None and self.pos_emb.shape[1] == tok.shape[1]:
            tok = tok + self.pos_emb
        return tok, (h, w)
