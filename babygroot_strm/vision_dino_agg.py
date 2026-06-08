"""Frozen DINOv2 + LayerAggregator (per-layer) + MLP → policy vision tokens.

Mirrors the InternVL aggregator pattern: take the per-transformer-layer hidden
states from a FROZEN DINOv2 backbone, run them through the same `LayerAggregator`
we used for InternVL/T5, pool the 16×16 patch grid to 7×7=49 tokens, then a small
trainable MLP projects to the policy dim. Single-frame input (DINOv2 is heavy;
2-frame variant can be added later if useful).

For the post-overfit retrain: rich pretrained vision (~22M frozen) instead of
the rank-6-collapsing 0.5M CNN, with a generalizable feature manifold the policy
can actually learn from.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from . import LayerAggregator


class DinoLayerAggMLP(nn.Module):
    def __init__(self, out_dim=512, hidden=1024, dropout=0.1,
                 dino_name='facebook/dinov2-small', n_tok_side=7,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.dino = AutoModel.from_pretrained(dino_name)
        for p in self.dino.parameters(): p.requires_grad_(False)
        self.dino.eval()
        d_in = self.dino.config.hidden_size              # 384 for small / 768 for base
        n_layers = self.dino.config.num_hidden_layers    # 12 for small
        self.agg = LayerAggregator(hidden_dim=d_in, n_layers=n_layers)
        self.norm = nn.LayerNorm(d_in)
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.Dropout(dropout),
        )
        self.n_tok_side = n_tok_side                     # 7 → 49 vision tokens
        self.out_dim = out_dim
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, frames, normalize=True, **_):
        # Accept (B,3,H,W) or (B,T,3,H,W) — for the 2-frame loader, use current frame only.
        if frames.dim() == 5:
            frames = frames[:, -1]
        x = (frames - self.mean) / self.std if normalize else frames
        with torch.no_grad():                            # backbone is frozen
            out = self.dino(pixel_values=x, output_hidden_states=True)
        hs = list(out.hidden_states[1:])                 # 12 × (B, 1+P, d_in); skip embed
        agg = self.agg(hs)                               # (B, 1+P, d_in)
        patches = agg[:, 1:, :]                          # drop CLS → (B, P, d_in)
        B, P, D = patches.shape; s = int(P ** 0.5)
        grid = patches.transpose(1, 2).reshape(B, D, s, s)
        pooled = F.adaptive_avg_pool2d(grid, self.n_tok_side).flatten(2).transpose(1, 2)
        tok = self.mlp(self.norm(pooled))                # (B, n_tok_side², out_dim)
        return tok, (self.n_tok_side, self.n_tok_side)
