#!/usr/bin/env python3
"""MEMORIZATION vs ILL-POSED TARGETS: how brittle are the VQ action codes to a
small window shift? For the SAME episodes, encode the action chunk at consecutive
starts (16..24) and measure how much changes per 1-frame shift.

  - raw action smoothness : ||a_chunk(s+1) - a_chunk(s)|| / std(a)  — should be small.
  - code change rate      : mean fraction of the 4 code positions that DIFFER between
                            consecutive starts, and for the 16→18 (off-stride) shift.
  - "in-between" coverage : for start=18, fraction of code positions equal to the
                            start-16 OR start-20 (trained) code at that position.

Reading:
  codes STABLE (low change, 18 mostly ∈ {16,20}) ⇒ off-stride is predictable ⇒ 13% = MEMORIZATION.
  codes FLIP every frame (high change, 18 ∉ {16,20}) ⇒ targets are brittle/high-freq ⇒
    discrete-VQ + lookback-relative precision-norm is ILL-POSED under shift (no model could).

Run: .venv/bin/python scripts/diag_code_stability.py [--n 300]
"""
import os, sys, glob, json, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch
from babygroot_strm import ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.multi_oxe import load_dataset_spec, _load_episode_parquet

ap = argparse.ArgumentParser(); ap.add_argument('--robot', default='widowx'); ap.add_argument('--n', type=int, default=300)
args = ap.parse_args()
BRIDGE = next(d for d in sorted(glob.glob('data/oxe/*'))
              if 'bridge_orig' in d and os.path.isfile(os.path.join(d, 'meta', 'info.json')))
c = torch.load(f'data/ckpts/oxe_vqvae_{args.robot}.pt', map_location='cpu', weights_only=False)
vae = ActionVQVAE1d(action_dim=c['action_dim'], vq_cls=VQ1d_EMA, k=c.get('k', 128)).eval(); vae.load_state_dict(c['vae'])
VG = c['action_var_global'].view(1, 1, -1)
sp = load_dataset_spec(BRIDGE, chunk_len=16, lookback=16, chunk_stride=16)

# len 45-60 episodes
eps = []
with open(os.path.join(BRIDGE, 'meta', 'episodes.jsonl')) as f:
    for line in f:
        r = json.loads(line); L = r.get('length') or 0
        if 45 <= L <= 60: eps.append(r['episode_index'])
eps = eps[:args.n]
STARTS = list(range(16, 25))

def codes_for(actions, start):                       # actions: (L,7) tensor
    ac = actions[start:start + 16][None]; pv = actions[start - 16:start][None]
    m = pv.mean(1, keepdim=True); S = ((pv - m) ** 2).sum(1, keepdim=True)
    lam = 16 / (S + 16 * VG); xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
    with torch.no_grad(): gt, _ = vae.encode_with_soft(xn, tau=0.1)
    return gt[0][0].numpy()                          # (T_l,)

per_ep_codes, raw_dadt, raw_std = {}, [], []
for ep in eps:
    try:
        actions, states, task_ix = _load_episode_parquet(sp, ep)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    except Exception:
        continue
    if actions.shape[0] < 25 + 16: continue
    per_ep_codes[ep] = {s: codes_for(actions, s) for s in STARTS}
    a = actions.numpy()
    raw_dadt.append(np.abs(a[17:25] - a[16:24]).mean(0))   # per-dim |Δ| frame-to-frame
    raw_std.append(a.std(0))
print(f"episodes used: {len(per_ep_codes)}\n")

# raw action smoothness
dadt = np.stack(raw_dadt).mean(0); astd = np.stack(raw_std).mean(0)
print("raw action |Δ per frame| / std  (small = smooth motion):")
print("  " + "  ".join(f"{nm}={dadt[j]/(astd[j]+1e-6):.2f}" for j, nm in
                        enumerate(['dx','dy','dz','rr','rp','ry','grip'])))

# code change rate per shift
Tl = len(next(iter(per_ep_codes.values()))[16])
def change(sa, sb):
    diffs = [(per_ep_codes[e][sa] != per_ep_codes[e][sb]).mean() for e in per_ep_codes]
    return np.mean(diffs)
print(f"\ncode-change fraction (of {Tl} positions):")
for s in STARTS[:-1]:
    print(f"  start {s}→{s+1} (1-frame): {change(s, s+1):.2f}")
print(f"  start 16→18 (2-frame, off-stride): {change(16, 18):.2f}")
print(f"  start 16→20 (trained→trained):     {change(16, 20):.2f}")

# is start=18 'in between' the trained 16 & 20?
inb = []
for e in per_ep_codes:
    c18, c16, c20 = per_ep_codes[e][18], per_ep_codes[e][16], per_ep_codes[e][20]
    inb.append(((c18 == c16) | (c18 == c20)).mean())
print(f"\nstart-18 code positions equal to start-16 OR start-20 (trained): {100*np.mean(inb):.0f}%")
print("(high ⇒ off-stride target is bracketed by trained codes ⇒ predictable ⇒ 13% = memorization;")
print(" low  ⇒ off-stride codes are novel/brittle ⇒ ill-posed target under shift)")
