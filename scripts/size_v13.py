#!/usr/bin/env python3
"""v13 sizing: shared backbone + per-emb scales (full 11-embodiment OXE)."""
import os, sys
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import torch, torch.nn as nn
from babygroot_strm import STRMPolicy, LayerAggregator, ScaleNorm
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import EMBODIMENTS

# Full OXE coverage — 11 distinct embodiments + unknown (matches the EMBODIMENTS list)
# (in v12 we had only 5 present because we'd only downloaded 10 of 36 datasets)
all_emb = list(EMBODIMENTS)
n_emb_total = len(EMBODIMENTS) + 1
n_emb_prefix = 16

print(f"Sizing for full OXE: {len(all_emb)} embodiments present\n")

def build(dim, depth, heads, kv_heads, ff_hidden, present_emb_count):
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=2, out_dim=192,
                       norm='scalenorm', pos_emb=True, img_size=224, dropout=0.1)
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
    cnn_proj = nn.Linear(192, dim)
    text_proj = nn.Linear(512, dim)
    kv_norm = ScaleNorm(dim)
    emb_present = all_emb[:present_emb_count]
    state_encoders = nn.ModuleDict({
        e: nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
        for e in emb_present})
    emb_id_emb = nn.Embedding(n_emb_total, dim * n_emb_prefix)
    n_vis = (224 // 32) ** 2
    max_prefix = n_vis + 24 + 16 + n_emb_prefix
    policy = STRMPolicy(seq_lens=(4,), k_codebook=512, dim=dim, heads=heads,
                        kv_heads=kv_heads, ff_hidden=ff_hidden, depth=depth,
                        L_inner=5, H_outer=2, state_dim=dim, max_prefix=max_prefix,
                        update_mode='damped', alpha_parametrization='sigmoid',
                        alpha_per_dim=True, n_embodiments=n_emb_total,
                        per_emb_head=True)
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    # decompose params
    shared_other = sum(p.numel() for p in cnn.parameters()) \
                 + sum(p.numel() for p in text_agg.parameters()) \
                 + sum(p.numel() for p in cnn_proj.parameters()) \
                 + sum(p.numel() for p in text_proj.parameters()) \
                 + sum(p.numel() for p in kv_norm.parameters())
    perEmb_state = sum(p.numel() for p in state_encoders.parameters())
    perEmb_id = sum(p.numel() for p in emb_id_emb.parameters())
    g_backbone = sum(p.numel() for p in policy.g.parameters())
    perEmb_head = sum(p.numel() for p in policy.out_head.parameters())
    policy_other = sum(p.numel() for p in policy.parameters()) - g_backbone - perEmb_head
    total = sum(p.numel() for m in mods for p in m.parameters())
    return dict(total=total,
                shared_backbone=g_backbone,
                shared_other=shared_other + policy_other,
                perEmb_state=perEmb_state,
                perEmb_head=perEmb_head,
                perEmb_id=perEmb_id)

configs = [
    ('v12 (current, 5 emb present)',    1024, 3, 16, 4, 8192,  5),
    ('v12 with full 11 emb',            1024, 3, 16, 4, 8192,  len(all_emb)),
    ('v13a: ff=10240',                  1024, 3, 16, 4, 10240, len(all_emb)),
    ('v13b: ff=12288',                  1024, 3, 16, 4, 12288, len(all_emb)),
    ('v13c: ff=14336 (Mistral-ish)',    1024, 3, 16, 4, 14336, len(all_emb)),
    ('v13d: depth=4 + ff=8192',         1024, 4, 16, 4, 8192,  len(all_emb)),
    ('v13e: dim=1152 + ff=8192',        1152, 3, 18, 6, 8192,  len(all_emb)),
]

print(f"{'config':<38s} {'total':>8s}  {'shared_bb':>10s} {'shared_oth':>11s} "
      f"{'pe_state':>9s} {'pe_head':>8s} {'pe_id':>7s}")
print('='*98)
for label, dim, depth, heads, kvh, ff, npe in configs:
    c = build(dim, depth, heads, kvh, ff, npe)
    print(f"{label:<38s} {c['total']/1e6:>6.1f}M  {c['shared_backbone']/1e6:>8.1f}M "
          f"{c['shared_other']/1e6:>9.1f}M  {c['perEmb_state']/1e6:>7.1f}M "
          f"{c['perEmb_head']/1e6:>6.1f}M  {c['perEmb_id']/1e6:>5.1f}M")
