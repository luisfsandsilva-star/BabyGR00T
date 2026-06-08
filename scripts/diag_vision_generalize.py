#!/usr/bin/env python3
"""Phase-1 battery: is the OVERFIT in the vision features, and does generalizable
visual signal even exist? Compares the trained CNN vs frozen DINOv2-small on:

  splits (all real bridge):
    FIT     = trained episodes (len 40-60), used to fit the linear probe / kNN ref
    INDIST  = DISJOINT trained episodes (len 40-60) — in-distribution control
    HELDOUT = length==32 episodes — never trained (disjoint by construction)

  metrics:
    A. feature-OOD kNN ratio: kNN(split→FIT)/kNN(FIT→FIT). INDIST should be ~1
       (in-dist), HELDOUT reveals OOD. Overfit encoder → HELDOUT >> 1.
    B. linear (ridge) probe features→action[0] (7-dim), fit on FIT, R² on INDIST & HELDOUT.
       Overfit encoder: INDIST R² ok but HELDOUT R² ~0. Generalizing encoder: both ok.

If CNN HELDOUT collapses while DINOv2 HELDOUT holds → vision features are the overfit
locus AND generalizable visual signal exists (→ encoder is the fix).

Run: .venv/bin/python scripts/diag_vision_generalize.py [--n-fit 256 --n-eval 192]
"""
import os, sys, json, glob, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset
from babygroot_strm.perimg_norm import normalize_image
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt-path', default='data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt')
ap.add_argument('--n-fit', type=int, default=256)
ap.add_argument('--n-eval', type=int, default=192)
ap.add_argument('--ridge', type=float, default=10.0)
args = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

BRIDGE = next(d for d in sorted(glob.glob('data/oxe/*'))
              if 'bridge_orig' in d and os.path.isfile(os.path.join(d, 'meta', 'info.json')))


def ep_ids(lo, hi):
    out = []
    with open(os.path.join(BRIDGE, 'meta', 'episodes.jsonl')) as f:
        for line in f:
            r = json.loads(line); L = r.get('length') or r.get('num_frames') or 0
            if lo <= L <= hi: out.append(r['episode_index'])
    return out


def load_split(ep_list, n, seed):
    """Return list of (PIL frame @start=16, action[0] (7,))."""
    sp = load_dataset_spec(BRIDGE, chunk_len=16, lookback=16, chunk_stride=16)
    sp.chunk_index = [(ep, 16) for ep in ep_list]
    ds = MultiOXEDataset([sp], chunk_len=16, lookback=16)
    rng = random.Random(seed); pool = list(range(len(ds))); rng.shuffle(pool)
    out = []
    for idx in pool:
        if len(out) >= n: break
        try:
            fr, st, ac, pv, tk, eid, di = ds[idx]
            # target = visible proprioceptive STATE (arm pose) — deterministic from the frame.
            out.append((fr.convert('RGB'), np.asarray(st, dtype=np.float32)))
        except Exception: pass
    return out


# ── splits (disjoint episode sets) ──
trained = ep_ids(40, 60); random.Random(0).shuffle(trained)
fit_eps = trained[:4000]; indist_eps = trained[4000:8000]      # disjoint trained pools
held_eps = ep_ids(32, 32)
print(f"pools: trained={len(trained)}  heldout(len32)={len(held_eps)}", flush=True)
FIT = load_split(fit_eps, args.n_fit, 1)
IND = load_split(indist_eps, args.n_eval, 2)
HLD = load_split(held_eps, args.n_eval, 3)
print(f"loaded FIT={len(FIT)} INDIST={len(IND)} HELDOUT={len(HLD)}", flush=True)
Y = {k: np.stack([a for _, a in v]) for k, v in [('FIT', FIT), ('INDIST', IND), ('HELDOUT', HLD)]}

# ── encoder 1: trained CNN (EMA) ──
ck = torch.load(args.ckpt_path, map_location='cpu', weights_only=False); a = ck['args']
cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=a['cnn_expand'], out_dim=a['cnn_out_dim'],
                   norm=a['cnn_norm'], pos_emb=a['cnn_pe'], img_size=a['img_size'], dropout=0.0, n_embodiments=0)
cnn_proj = nn.Linear(a['cnn_out_dim'], a['dim'])
cnn.load_state_dict(ck['cnn']); cnn_proj.load_state_dict(ck['cnn_proj'])
for nm, m in [('cnn', cnn), ('cnn_proj', cnn_proj)]:
    sd = m.state_dict()
    for k in list(sd):
        if f'{nm}.{k}' in ck.get('ema_params', {}): sd[k] = ck['ema_params'][f'{nm}.{k}'].to(sd[k].dtype)
    m.load_state_dict(sd)
cnn = cnn.to(dev).eval(); cnn_proj = cnn_proj.to(dev).eval()
imgvar = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)['var_global'].to(dev)
IMG = a['img_size']

@torch.no_grad()
def cnn_feats(pils):
    out = []
    for i in range(0, len(pils), 16):
        x = torch.stack([torch.from_numpy(np.asarray(p.resize((IMG, IMG))).copy()).permute(2,0,1).float()/255.
                         for p in pils[i:i+16]]).to(dev)
        x = normalize_image(x, imgvar)
        v, _ = cnn(x); v = cnn_proj(v)
        out.append(v.mean(1).float().cpu().numpy())
    return np.concatenate(out)

# ── encoder 2: frozen DINOv2-small ──
from transformers import AutoModel, AutoImageProcessor
proc = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
dino = AutoModel.from_pretrained('facebook/dinov2-small').to(dev).eval()

@torch.no_grad()
def dino_feats(pils):
    out = []
    for i in range(0, len(pils), 16):
        inp = proc(images=pils[i:i+16], return_tensors='pt').to(dev)
        h = dino(**inp).last_hidden_state          # (B, 1+P, 384)
        out.append(h[:, 1:, :].mean(1).float().cpu().numpy())
    return np.concatenate(out)


def knn(A, B, k):
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2); return np.sort(d, axis=1)[:, :k].mean()

def ridge_r2(Xf, Yf, Xe, Ye, lam):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    Xf = (Xf - mu) / sd; Xe = (Xe - mu) / sd
    ym, ys = Yf.mean(0), Yf.std(0) + 1e-6
    Yf2 = (Yf - ym) / ys
    d = Xf.shape[1]
    W = np.linalg.solve(Xf.T @ Xf + lam * np.eye(d), Xf.T @ Yf2)
    pred = (Xe @ W) * ys + ym
    ss_res = ((Ye - pred) ** 2).sum(); ss_tot = ((Ye - Ye.mean(0)) ** 2).sum()
    return 1 - ss_res / ss_tot

for name, feat in [('trained-CNN', cnn_feats), ('DINOv2-small', dino_feats)]:
    print(f"\n=== {name} ===", flush=True)
    Ff, Fi, Fh = feat([p for p, _ in FIT]), feat([p for p, _ in IND]), feat([p for p, _ in HLD])
    rr = knn(Ff[:min(50, len(Ff))], Ff, 6)
    print(f"  feature-OOD kNN ratio (want INDIST~1):  INDIST={knn(Fi,Ff,5)/rr:.2f}   HELDOUT={knn(Fh,Ff,5)/rr:.2f}")
    r2_i = ridge_r2(Ff, Y['FIT'], Fi, Y['INDIST'], args.ridge)
    r2_h = ridge_r2(Ff, Y['FIT'], Fh, Y['HELDOUT'], args.ridge)
    print(f"  probe feats→STATE/pose R² :             INDIST={r2_i:.3f}    HELDOUT={r2_h:.3f}")
print("\n(Overfit encoder: HELDOUT kNN>>1 and HELDOUT R²~0 while INDIST is fine.)")
