#!/usr/bin/env python3
"""Is the length==32 held-out a FAIR generalization probe, or are its TARGETS OOD?

The held-out eval window is always the TERMINAL chunk of a short episode (frames
16-31 = episode end: gripper release / retreat). If those produce action codes
rare in training, ~6% held-out reflects TARGET NOVELTY, not a vision/generalization
failure — and no encoder could fix it.

Compares, train(len40-60) vs heldout(len32):
  - per-position action-code histograms → total-variation distance + KL.
  - marginal-baseline acc: predict the train-argmax code per position; measure on
    train vs heldout. Big drop ⇒ targets shifted.
  - raw action[0] stats (mean/std per dim) → are the motions themselves different.

Run: .venv/bin/python scripts/diag_code_dist.py [--n 600]
"""
import os, sys, json, glob, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch
from babygroot_strm import ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset

ap = argparse.ArgumentParser(); ap.add_argument('--robot', default='widowx'); ap.add_argument('--n', type=int, default=600)
args = ap.parse_args()
BRIDGE = next(d for d in sorted(glob.glob('data/oxe/*'))
              if 'bridge_orig' in d and os.path.isfile(os.path.join(d, 'meta', 'info.json')))
c = torch.load(f'data/ckpts/oxe_vqvae_{args.robot}.pt', map_location='cpu', weights_only=False)
vae = ActionVQVAE1d(action_dim=c['action_dim'], vq_cls=VQ1d_EMA, k=c.get('k', 128)).eval()
vae.load_state_dict(c['vae'])
VG = c['action_var_global'].view(1, 1, -1); K = c.get('k', 256)

def ep_ids(lo, hi):
    out = []
    with open(os.path.join(BRIDGE, 'meta', 'episodes.jsonl')) as f:
        for line in f:
            r = json.loads(line); L = r.get('length') or r.get('num_frames') or 0
            if lo <= L <= hi: out.append(r['episode_index'])
    return out

def collect(ep_list, n, seed):
    s = load_dataset_spec(BRIDGE, chunk_len=16, lookback=16, chunk_stride=16)
    s.chunk_index = [(ep, 16) for ep in ep_list]
    d = MultiOXEDataset([s], chunk_len=16, lookback=16)
    rng = random.Random(seed); pool = list(range(len(d))); rng.shuffle(pool)
    acs, pvs, a0 = [], [], []
    for i in pool:
        if len(acs) >= n: break
        try:
            fr, st, ac, pv, tk, eid, di = d[i]
            if ac.shape[-1] != c['action_dim']: continue
            acs.append(ac); pvs.append(pv); a0.append(np.asarray(ac[0], np.float32))
        except Exception: pass
    ac = torch.stack(acs); pv = torch.stack(pvs)
    nT = ac.shape[1]; m = pv.mean(1, keepdim=True); S = ((pv - m) ** 2).sum(1, keepdim=True)
    lam = nT / (S + nT * VG); xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
    with torch.no_grad(): gt, _ = vae.encode_with_soft(xn, tau=0.1)
    return gt[0].numpy(), np.stack(a0)            # codes (N,T_l), action0 (N,7)

tr_codes, tr_a0 = collect(ep_ids(40, 60), args.n, 1)
hd_codes, hd_a0 = collect(ep_ids(32, 32), args.n, 3)
Tl = tr_codes.shape[1]
print(f"codes shape: train {tr_codes.shape}  heldout {hd_codes.shape}  (K={K})\n")

# per-position histograms → TV distance + KL + marginal baseline
print("position |  TV(train,held) |  KL(held||train) | marginal-acc train/held")
for p in range(Tl):
    ht = np.bincount(tr_codes[:, p], minlength=K) + 1e-6; ht /= ht.sum()
    hh = np.bincount(hd_codes[:, p], minlength=K) + 1e-6; hh /= hh.sum()
    tv = 0.5 * np.abs(ht - hh).sum()
    kl = (hh * np.log(hh / ht)).sum()
    am = ht.argmax()
    acc_tr = (tr_codes[:, p] == am).mean(); acc_hd = (hd_codes[:, p] == am).mean()
    print(f"   {p}     |     {tv:.3f}      |     {kl:6.2f}      |   {acc_tr*100:4.1f}% / {acc_hd*100:4.1f}%")

# fraction of held-out codes that are rare in train
rare = 0; tot = 0
for p in range(Tl):
    ht = np.bincount(tr_codes[:, p], minlength=K) / len(tr_codes)
    rare += (ht[hd_codes[:, p]] < 0.005).sum(); tot += len(hd_codes)
print(f"\nheld-out codes that are RARE in train (<0.5% freq): {100*rare/tot:.1f}%")

# raw action[0] stats
print("\naction[0] mean±std per dim (train vs held):")
for j, nm in enumerate(['dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw', 'grip']):
    print(f"  {nm:6s} train {tr_a0[:,j].mean():+.3f}±{tr_a0[:,j].std():.3f}   held {hd_a0[:,j].mean():+.3f}±{hd_a0[:,j].std():.3f}")
