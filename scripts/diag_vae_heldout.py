#!/usr/bin/env python3
"""Is the action VQ-VAE tokenizer overfit? — split-aware reconstruction check.

Now that we have a proper episode-level split (data/splits/{train,test}_eps.json),
measure the tokenizer's reconstruction on WidowX/bridge chunks drawn from TRAIN
episodes vs from held-out TEST episodes. If the tokenizer overfit, held-out recon
MSE >> train recon MSE, and the discrete code targets the policy learns are
ill-defined on held-out data (capping held-out accuracy regardless of the policy).

Metrics per split: recon MSE (action units + normalized space), per-dim MSE,
codebook usage / entropy. Reports the train→test gap.

CAVEAT (printed in the report): the shared VAE was trained with a *chunk-level*
10% val split (train_oxe_vaes.py), NOT an episode split — so test-split episodes
were very likely seen by the VAE during its training. A near-zero train→test gap
here is therefore necessary-but-not-sufficient for "generalizes"; the definitive
test is retraining the VAE with test_eps excluded. A *non-zero* gap, by contrast,
is a strong positive signal of overfitting even under this leakage.
"""
import os, sys, json, random
os.environ['CUDA_VISIBLE_DEVICES'] = ''
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch
from collections import Counter
from babygroot_strm.cond_vae import CondActionVQVAE1d
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                      _episode_paths, EMBODIMENTS, EMBODIMENT_ID)

VAE_CKPT = 'data/ckpts/oxe_shared_vae.pt'
BRIDGE   = 'data/oxe/bridge_orig_lerobot'
N_PER_SPLIT = 512
torch.set_num_threads(4)
random.seed(42); torch.manual_seed(42); np.random.seed(42)


@torch.no_grad()
def collect(vae, sc, ds, eid_fixed, K, A, n_target):
    """Sample up to n_target chunks from ds, return recon stats."""
    order = list(range(len(ds))); random.shuffle(order)
    ac_list, pv_list = [], []
    for i in order:
        if len(ac_list) >= n_target: break
        try:
            _, _, ac, pv, _, _, _ = ds[i]
        except Exception:
            continue
        if ac.shape[-1] != A or pv.shape[-1] != A: continue
        ac_list.append(ac); pv_list.append(pv)
    if not ac_list:
        return None
    ac = torch.stack(ac_list); pv = torch.stack(pv_list)          # (N,16,A)
    N = ac.shape[0]
    vg = sc['action_var_globals'][eid_fixed].view(1, 1, -1)
    m = pv.mean(dim=1, keepdim=True)
    S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
    lam = 16 / (S + 16 * vg)
    xn = ((ac - m) * lam.sqrt()).transpose(1, 2)                  # (N,A,16)
    eid_t = torch.full((N,), eid_fixed, dtype=torch.long)
    codes, _ = vae.encode_with_soft(xn, eid_t, tau=0.1)
    recon = vae.decode_from_indices(codes, eid_t)                 # (N,A,16)
    recon_t = recon.transpose(1, 2)
    inv_sqrt = 1.0 / lam.sqrt()
    ac_recon = recon_t * inv_sqrt + m
    mse_action = ((ac_recon - ac) ** 2).mean().item()
    mse_per_dim = ((ac_recon - ac) ** 2).mean(dim=(0, 1)).tolist()
    mse_norm = ((recon - xn) ** 2).mean().item()
    # data scale for context
    act_var = ac.var(dim=(0, 1)).mean().item()
    codes_flat = codes[0].flatten().tolist()
    usage = Counter(codes_flat); total = sum(usage.values())
    probs = np.array([c / total for c in usage.values()])
    H = float(-(probs * np.log(probs + 1e-12)).sum())
    return dict(n=N, mse_action=mse_action, mse_norm=mse_norm, mse_per_dim=mse_per_dim,
                nrmse=(mse_action / (act_var + 1e-12)) ** 0.5,
                n_unique=len(usage), entropy_norm=H / np.log(K),
                codes_used=set(usage.keys()))


@torch.no_grad()
def main():
    print(f"loading shared cond-VAE: {VAE_CKPT}")
    sc = torch.load(VAE_CKPT, map_location='cpu', weights_only=False)
    vae = CondActionVQVAE1d(action_dim=sc['action_dim'], n_embodiments=sc['n_embodiments'], k=sc['k'])
    vae.load_state_dict(sc['vae']); vae.eval()
    K, A = sc['k'], sc['action_dim']
    eid = EMBODIMENT_ID['widowx']
    print(f"  K={K} action_dim={A} widowx eid={eid}")

    sp = load_dataset_spec(BRIDGE, chunk_len=16, lookback=16)
    print(f"  bridge spec: robot={sp.robot} total_chunks={len(sp.chunk_index)}")

    tr = set(json.load(open('data/splits/train_eps.json')))
    te = set(json.load(open('data/splits/test_eps.json')))
    # keep only episodes whose video is present (so __getitem__ works)
    def has_vid(ep): return os.path.isfile(_episode_paths(sp, ep)[1])

    import copy
    out = {}
    for name, epset in (('TRAIN', tr), ('TEST', te)):
        spc = copy.copy(sp)
        ci = [(ep, s) for (ep, s) in sp.chunk_index if ep in epset and has_vid(ep)]
        spc.chunk_index = ci
        n_eps = len({ep for ep, _ in ci})
        print(f"  [{name}] {len(ci)} chunks from {n_eps} episodes (video present)")
        ds = MultiOXEDataset([spc], chunk_len=16, lookback=16)
        out[name] = collect(vae, sc, ds, eid, K, A, N_PER_SPLIT)

    print(f"\n{'='*70}\nWIDOWX TOKENIZER RECON — TRAIN vs HELD-OUT TEST\n{'='*70}")
    print(f"{'split':<8s} {'N':>5s} {'MSE_action':>12s} {'NRMSE':>8s} {'MSE_norm':>10s} "
          f"{'uniqCodes':>10s} {'entropy/Hmax':>12s}")
    for name in ('TRAIN', 'TEST'):
        s = out[name]
        if s is None: print(f"  {name}: no samples"); continue
        print(f"  {name:<6s} {s['n']:>5d} {s['mse_action']:>12.6f} {s['nrmse']:>8.4f} "
              f"{s['mse_norm']:>10.5f} {s['n_unique']:>6d}/{K:<3d} {s['entropy_norm']:>11.3f}")
    if out['TRAIN'] and out['TEST']:
        ga = out['TEST']['mse_action'] / max(out['TRAIN']['mse_action'], 1e-12)
        gn = out['TEST']['mse_norm'] / max(out['TRAIN']['mse_norm'], 1e-12)
        print(f"\n  GAP (test/train):  MSE_action ×{ga:.3f}   MSE_norm ×{gn:.3f}")
        print(f"  → {'OVERFIT SIGNAL' if ga > 1.15 else 'no meaningful gap'} "
              f"(>1.15× action-MSE would flag overfit)")
        jac = len(out['TRAIN']['codes_used'] & out['TEST']['codes_used']) / \
              max(len(out['TRAIN']['codes_used'] | out['TEST']['codes_used']), 1)
        print(f"  code-usage Jaccard(train,test) = {jac:.3f} (1.0 = identical code distribution)")
        print(f"\n  per-dim MSE_action (train | test):")
        for d in range(A):
            print(f"    d{d}: {out['TRAIN']['mse_per_dim'][d]:.6f} | {out['TEST']['mse_per_dim'][d]:.6f}")
    print(f"\n  CAVEAT: shared VAE used a chunk-level (not episode) val split → test")
    print(f"  episodes likely seen in VAE training. Zero gap is necessary-not-sufficient;")
    print(f"  a non-zero gap is strong evidence of overfit. Definitive test = retrain VAE w/o test_eps.")


if __name__ == '__main__':
    main()
