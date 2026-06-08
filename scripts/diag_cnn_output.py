#!/usr/bin/env python3
"""Characterize WHAT the trained CNN actually outputs — is it conditional/structured,
or collapsed/constant — across TRAIN(real) / HELDOUT(len==32 real) / SIM.

Metrics (pooled = mean over 49 tokens, dim=512):
  1. inter-frame cosine sim   : mean cos(f_i, f_j) over DIFFERENT frames.
       ~1 ⇒ CNN outputs ~the same vector regardless of input (COLLAPSE / constant).
  2. effective rank (part.ratio): (Σλ)²/Σλ² of the feature covariance.
       small ⇒ features on a thin manifold (degenerate); large ⇒ rich/structured.
  3. rel. between-frame spread : mean ||f_i - mean|| / ||mean||.  ~0 ⇒ constant.
  4. excess kurtosis (per-dim) : ~0 ⇒ Gaussian-ish; >>0 ⇒ sparse/heavy-tailed structure.
  5. spatial token diversity   : within-image std(49 tokens) / between-image std.
       ~0 ⇒ all spatial tokens identical (no spatial structure).
  cross-domain: mean-shift ||μ_x-μ_train||/||μ_train|| and variance ratio (collapse if <1).
  aug stability (train): cos(feat(x), feat(aug(x))) under photometric jitter and APR.

Run: .venv/bin/python scripts/diag_cnn_output.py [--n 256]
"""
import os, sys, json, glob, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm import augment
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt-path', default='data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt')
ap.add_argument('--n', type=int, default=256)
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


def load_frames(ep_list, n, seed):
    sp = load_dataset_spec(BRIDGE, chunk_len=16, lookback=16, chunk_stride=16)
    sp.chunk_index = [(ep, 16) for ep in ep_list]
    ds = MultiOXEDataset([sp], chunk_len=16, lookback=16)
    rng = random.Random(seed); pool = list(range(len(ds))); rng.shuffle(pool)
    out = []
    for idx in pool:
        if len(out) >= n: break
        try: out.append(ds[idx][0].convert('RGB'))
        except Exception: pass
    return out


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


def pil_to_t(pils):
    return torch.stack([torch.from_numpy(np.asarray(p.resize((IMG, IMG))).copy()).permute(2,0,1).float()/255.
                        for p in pils])

@torch.no_grad()
def feats(x01):                       # x01: (N,3,H,W) in [0,1] → pooled (N,512), tokens (N,49,512)
    pooled, toks = [], []
    for i in range(0, len(x01), 16):
        x = normalize_image(x01[i:i+16].to(dev), imgvar)
        v, _ = cnn(x); v = cnn_proj(v)
        pooled.append(v.mean(1).float().cpu().numpy()); toks.append(v.float().cpu().numpy())
    return np.concatenate(pooled), np.concatenate(toks)


def eff_rank(X):
    Xc = X - X.mean(0); C = (Xc.T @ Xc) / len(X)
    ev = np.linalg.eigvalsh(C); ev = np.clip(ev, 0, None)
    return (ev.sum() ** 2) / (np.square(ev).sum() + 1e-12)

def inter_cos(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = Xn @ Xn.T; iu = np.triu_indices(len(X), 1)
    return S[iu].mean()

def rel_spread(X):
    mu = X.mean(0); return (np.linalg.norm(X - mu, axis=1).mean()) / (np.linalg.norm(mu) + 1e-8)

def tok_diversity(T):                 # T: (N,49,512)
    within = T.std(axis=1).mean()                       # avg std across the 49 tokens, per image
    between = T.mean(axis=1).std(axis=0).mean()         # std of per-image pooled, across images
    return within / (between + 1e-8)

# ── load domains ──
tr = ep_ids(40, 60); random.Random(0).shuffle(tr)
TRAIN = pil_to_t(load_frames(tr[:8000], args.n, 1))
HELD  = pil_to_t(load_frames(ep_ids(32, 32), args.n, 3))
sim_np = np.load('/tmp/sim_frames.npy')
SIM = torch.from_numpy(sim_np[np.random.RandomState(0).choice(len(sim_np), args.n, replace=False)]
                       .astype(np.float32) / 255.).permute(0, 3, 1, 2)
print(f"loaded TRAIN={len(TRAIN)} HELDOUT={len(HELD)} SIM={len(SIM)}\n", flush=True)

P = {}
for name, x in [('TRAIN', TRAIN), ('HELDOUT', HELD), ('SIM', SIM)]:
    pf, tf = feats(x); P[name] = pf
    print(f"== {name} ==")
    print(f"   inter-frame cos sim  = {inter_cos(pf):.3f}   (1=constant/collapsed)")
    print(f"   effective rank /512  = {eff_rank(pf):6.1f}")
    print(f"   rel between-frame spread = {rel_spread(pf):.3f}   (0=constant)")
    print(f"   mean |excess kurtosis|   = {np.abs(((pf-pf.mean(0))/(pf.std(0)+1e-8))**4).mean()-3:.2f}")
    print(f"   spatial token diversity  = {tok_diversity(tf):.3f}   (0=tokens identical)")
    print()

mu_tr = P['TRAIN'].mean(0); v_tr = P['TRAIN'].var(0).mean()
for name in ['HELDOUT', 'SIM']:
    shift = np.linalg.norm(P[name].mean(0) - mu_tr) / (np.linalg.norm(mu_tr) + 1e-8)
    vr = P[name].var(0).mean() / (v_tr + 1e-8)
    print(f"{name:8s} vs TRAIN:  mean-shift={shift:.3f}   variance-ratio={vr:.3f}  (<1 ⇒ collapse on {name})")

# ── augmentation stability (train) ──
print("\n-- aug stability (train frames; cos between feat(x) and feat(aug x)) --")
base = TRAIN[:64]
pf0, _ = feats(base)
# photometric jitter (training aug)
import random as _r
ph = pil_to_t([augment._apply_visual_params(
        Image.fromarray((base[i].permute(1,2,0).numpy()*255).astype(np.uint8)),
        augment._sample_visual_params(_r.Random(i), brightness=(0.6,1.4), contrast=(0.6,1.4),
                                      saturation=(0.6,1.4), blur_sigma=(0.0,1.5), crop_keep=(0.7,1.0)))
      for i in range(len(base))])
pfp, _ = feats(ph)
apr = augment.apr_augment(base.clone(), p=1.0, eta_max=1.0)
pfa, _ = feats(apr)
def cos_rows(A, B):
    A = A/(np.linalg.norm(A,axis=1,keepdims=True)+1e-8); B = B/(np.linalg.norm(B,axis=1,keepdims=True)+1e-8)
    return (A*B).sum(1).mean()
print(f"   photometric jitter : cos={cos_rows(pf0,pfp):.3f}")
print(f"   APR amplitude swap : cos={cos_rows(pf0,pfa):.3f}")
