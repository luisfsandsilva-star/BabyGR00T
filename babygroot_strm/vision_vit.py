"""Transformer-first 2-frame vision encoder (replaces the tiny rank-collapsing CNN).

Design (per spec):
  • shallow CONVOLUTIONAL PATCHIFIER (SmallStem) processes each of TWO time steps
    (shared weights) → patch tokens per frame.
  • the two frames' tokens are concatenated so the downstream ViT SEES BOTH frames,
    with correct PE: 2D spatial PE (shared across frames) + temporal PE (per frame).
  • shallow ViT built from the POLICY's own efficient blocks: GQA SelfAttention +
    QK-norm + ScaleNorm (pre-norm) + GeGLU (same SwiGLU-style FFN as the policy).
  • output tokens projected to the policy dim.

Rationale: the old 0.52M EfficientCNN collapsed to ~6 effective dims and memorized.
This is transformer-first (Octo-style: shallow conv stem, capacity in the ViT) with
temporal context (2 frames → motion) so vision carries a richer, generalizable signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .policy import ScaleNorm, SelfAttention, GeGLU
from .cnn_vision import MixStyle


class SmallStem(nn.Module):
    """Shallow conv patchifier: 224 →(/2 ×3)→ 28 →(patch 4)→ 7×7 tokens. Shared per frame.
    Optional MixStyle (channel-stat randomization) on the early conv feature maps."""
    def __init__(self, dim, in_ch=3, img_size=224, mixstyle_p=0.0):
        super().__init__()
        chs = [32, 64, 128]; self.blocks = nn.ModuleList(); c = in_ch
        for o in chs:
            self.blocks.append(nn.Sequential(nn.Conv2d(c, o, 3, stride=2, padding=1),
                                             nn.GroupNorm(8, o), nn.GELU()))
            c = o
        self.patch = nn.Conv2d(c, dim, 4, stride=4)           # 28 → 7 patch-embed
        self.mixstyle = MixStyle(mixstyle_p) if mixstyle_p > 0 else None
        self.grid = img_size // 32                            # 224→7
    def forward(self, x):                                     # (B,3,H,W) → (B, grid², dim)
        for i, b in enumerate(self.blocks):
            x = b(x)
            if self.mixstyle is not None and i < 2: x = self.mixstyle(x)   # early-stage style randomization
        return self.patch(x).flatten(2).transpose(1, 2)


class ViTBlock(nn.Module):
    """Pre-norm ScaleNorm → GQA SelfAttention(QK-norm) → ScaleNorm → GeGLU. Same blocks as policy."""
    def __init__(self, dim, heads, kv_heads, ff_hidden, dropout=0.0):
        super().__init__()
        self.n1 = ScaleNorm(dim); self.attn = SelfAttention(dim, heads, dropout, kv_heads)
        self.n2 = ScaleNorm(dim); self.ff = GeGLU(dim, ff_hidden, dropout)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x


class TwoFrameViT(nn.Module):
    """2 frames → SmallStem (shared) → tokens + spatial&temporal PE → shallow GQA-ViT → (B, 2·grid², out_dim).

    forward(frames): frames (B, 2, 3, H, W) in [0,1]. Returns (tokens, (n_frames, h, w)).
    """
    def __init__(self, dim=384, depth=6, heads=6, kv_heads=2, ff_hidden=1536, out_dim=512,
                 img_size=224, n_frames=2, dropout=0.0, mixstyle_p=0.0,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.n_frames = n_frames
        self.stem = SmallStem(dim, 3, img_size, mixstyle_p=mixstyle_p)
        g = self.stem.grid; self.n_patch = g * g; self.grid = g
        self.spatial_pe  = nn.Parameter(torch.zeros(1, 1, self.n_patch, dim))
        self.temporal_pe = nn.Parameter(torch.zeros(1, n_frames, 1, dim))
        nn.init.trunc_normal_(self.spatial_pe, std=0.02)
        nn.init.trunc_normal_(self.temporal_pe, std=0.02)
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, kv_heads, ff_hidden, dropout) for _ in range(depth)])
        self.out_norm = ScaleNorm(dim)
        self.proj = nn.Identity() if dim == out_dim else nn.Linear(dim, out_dim)
        self.out_dim = out_dim
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, frames, normalize=True):
        if frames.dim() == 4:                                # (B,3,H,W) → single frame, tile to n_frames
            frames = frames.unsqueeze(1).expand(-1, self.n_frames, -1, -1, -1)
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, H, W)
        if normalize:
            x = (x - self.mean) / self.std
        x = self.stem(x).view(B, T, self.n_patch, -1)        # (B, T, n_patch, dim)
        x = x + self.spatial_pe + self.temporal_pe[:, :T]     # spatial (shared) + temporal PE
        x = x.flatten(1, 2)                                   # (B, T·n_patch, dim)
        for blk in self.blocks:
            x = blk(x)
        tok = self.proj(self.out_norm(x))                    # (B, T·n_patch, out_dim)
        return tok, (T, self.grid, self.grid)
