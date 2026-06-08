#!/usr/bin/env python3
"""Estimate param count for several v12 candidate configs to hit 100-150M target.
Builds the full model bag (CNN + state encoders + policy + projections) like
scripts/train_oxe.py does, and reports trainable params per config.
"""
import os, sys, glob
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import torch, torch.nn as nn
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import EMBODIMENTS

OXE_ROOT = 'data/oxe'

# fetch real embodiment count (5 present in our 10-dataset OXE subset)
present_emb = ['franka', 'google_robot', 'jaco_2', 'ur5', 'widowx']
n_emb_total = len(EMBODIMENTS) + 1                                # 11 + 1 unknown = 12
n_emb_prefix = 16


def build(dim, depth, heads, kv_heads, ff_hidden, per_emb_head=True, seq_lens=(4,), K=512):
    # CNN (fixed config from v11)
    cnn = EfficientCNN(dims=[24, 48, 96, 192], depths=[1,1,1,1], expand=2,
                       out_dim=192, norm='scalenorm', pos_emb=True,
                       img_size=224, dropout=0.1)
    # T5 dim from cache (flan-t5-small=512)
    t5_dim = 512; t5_layers = 9
    text_agg = LayerAggregator(hidden_dim=t5_dim, n_layers=t5_layers)
    cnn_proj = nn.Linear(192, dim)
    text_proj = nn.Linear(t5_dim, dim)
    kv_norm = ScaleNorm(dim)
    n_vis = (224 // 32) ** 2
    state_encoders = nn.ModuleDict({
        emb: nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
        for emb in present_emb})
    emb_id_emb = nn.Embedding(n_emb_total, dim * n_emb_prefix)
    max_prefix = n_vis + 24 + 16 + n_emb_prefix
    policy = STRMPolicy(seq_lens=seq_lens, k_codebook=K, dim=dim, heads=heads,
                        kv_heads=kv_heads, ff_hidden=ff_hidden, depth=depth,
                        L_inner=5, H_outer=2, state_dim=dim, max_prefix=max_prefix,
                        weighting='geometric', rho_L=0.0, rho_H=0.0,
                        update_mode='damped', alpha_parametrization='sigmoid',
                        alpha_per_dim=True, n_embodiments=n_emb_total,
                        per_emb_head=per_emb_head)
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    return mods, policy


def count(mods, policy):
    total = sum(p.numel() for m in mods for p in m.parameters())
    # break down by category
    cnn_p = sum(p.numel() for p in mods[0].parameters())
    txt_p = sum(p.numel() for p in mods[1].parameters()) + sum(p.numel() for p in mods[3].parameters())
    proj_p = sum(p.numel() for p in mods[2].parameters()) + sum(p.numel() for p in mods[4].parameters())
    se_p = sum(p.numel() for p in mods[5].parameters())
    emb_p = sum(p.numel() for p in mods[6].parameters())
    pol_p = sum(p.numel() for p in policy.parameters())
    # within policy: backbone (g) vs heads vs embeddings
    g_p = sum(p.numel() for p in policy.g.parameters())
    head_p = sum(p.numel() for p in policy.out_head.parameters())
    tok_p = sum(p.numel() for p in policy.tok_emb.parameters())
    misc_pol = pol_p - g_p - head_p - tok_p
    return dict(total=total, cnn=cnn_p, txt=txt_p, proj=proj_p, se=se_p,
                emb_id=emb_p, policy=pol_p, g_backbone=g_p, heads=head_p,
                tok_emb=tok_p, misc_pol=misc_pol)


def fmt(c):
    return (f"total {c['total']/1e6:>6.1f}M = "
            f"g {c['g_backbone']/1e6:>5.1f}M + heads {c['heads']/1e6:>5.1f}M + "
            f"emb_id {c['emb_id']/1e6:>4.1f}M + state_enc {c['se']/1e6:>4.1f}M + "
            f"cnn {c['cnn']/1e6:>4.1f}M + other {(c['txt']+c['proj']+c['tok_emb']+c['misc_pol'])/1e6:>4.1f}M")


configs = [
    # (label, dim, depth, heads, kv_heads, ff_hidden)
    ('v11 baseline (dim=768, h=8, ff=2048)',         768,  3,  8, None, 2048),
    ('v12a: dim=1024, h=16, kv=4, ff=2752 (ratio 8/3)', 1024, 3, 16, 4, 2752),
    ('v12b: dim=1024, h=16, kv=4, ff=4096 (ratio 4 GPT-2)', 1024, 3, 16, 4, 4096),
    ('v12c: dim=1024, h=16, kv=4, ff=5120 (ratio 5)', 1024, 3, 16, 4, 5120),
    ('v12d: dim=1024, h=16, kv=4, ff=6144 (ratio 6)', 1024, 3, 16, 4, 6144),
    ('v12e: dim=1024, h=16, kv=4, ff=8192 (ratio 8)', 1024, 3, 16, 4, 8192),
    ('v12f: dim=1024, h=16, kv=8 (looser GQA), ff=4096', 1024, 3, 16, 8, 4096),
    ('v12g: dim=1152, h=18, kv=6, ff=4096', 1152, 3, 18, 6, 4096),
]

print(f"{'config':<55s} {'params':<70s}")
print('=' * 130)
for label, dim, depth, heads, kv_heads, ff_hidden in configs:
    try:
        mods, policy = build(dim, depth, heads, kv_heads, ff_hidden)
        c = count(mods, policy)
        print(f"{label:<55s} {fmt(c)}")
    except Exception as e:
        print(f"{label:<55s} ERROR: {e}")
