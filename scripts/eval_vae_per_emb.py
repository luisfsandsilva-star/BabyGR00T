#!/usr/bin/env python3
"""Detailed shared-VAE diagnostic per embodiment (CPU only).

Measures:
  - Reconstruction MSE per emb (encode → decode → compare GT actions)
  - Per-action-dim MSE breakdown (which dims are well/poorly modeled)
  - Codebook usage per emb (which of the K=512 codes does each emb actually use?)
  - Code entropy per emb (concentrated vs uniform usage)
  - Code overlap between embs (Jaccard) — do robots share codes or specialize?

Uses local OXE only, large N samples per emb for stable stats.
"""
import os, sys, glob, json, random
os.environ['CUDA_VISIBLE_DEVICES'] = ''
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn.functional as F
from collections import defaultdict, Counter
from babygroot_strm.cond_vae import CondActionVQVAE1d
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)

VAE_CKPT = 'data/ckpts/oxe_shared_vae.pt'
N_PER_EMB = 256
torch.set_num_threads(2)
random.seed(42); torch.manual_seed(42); np.random.seed(42)


@torch.no_grad()
def main():
    print(f"loading shared VAE ...")
    sc = torch.load(VAE_CKPT, map_location='cpu', weights_only=False)
    vae = CondActionVQVAE1d(action_dim=sc['action_dim'], n_embodiments=sc['n_embodiments'], k=sc['k'])
    vae.load_state_dict(sc['vae']); vae.eval()
    K = sc['k']; A = sc['action_dim']
    print(f"  K={K}, action_dim={A}, n_embodiments={sc['n_embodiments']}")
    print(f"  embodiments in var_globals: {sorted(sc['action_var_globals'].keys())}")
    var_globals = {EMBODIMENTS[i] if i < len(EMBODIMENTS) else f'eid{i}':
                   sc['action_var_globals'][i].view(1, 1, -1) for i in sc['action_var_globals']}

    print(f"\nloading dataset specs (local only)...")
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16)
            if sp.chunk_index: specs.append(sp)
        except Exception: pass
    print(f"  {len(specs)} datasets")
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)

    # collect balanced samples by emb (require matching action_dim)
    by_emb_idx = defaultdict(list)
    pool = list(range(len(ds))); random.shuffle(pool)
    for idx in pool:
        di, _ = ds.flat[idx]
        emb = specs[di].robot
        if emb not in var_globals: continue
        if len(by_emb_idx[emb]) >= N_PER_EMB: continue
        by_emb_idx[emb].append(idx)

    # Run encode → decode for each
    per_emb_stats = {}
    for emb, idxs in by_emb_idx.items():
        eid = EMBODIMENT_ID.get(emb, -1)
        if eid not in sc['action_var_globals']: continue
        ac_list, pv_list, codes_list = [], [], []
        for i in idxs:
            try:
                _, _, ac, pv, _, _, _ = ds[i]
            except Exception: continue
            if ac.shape[-1] != A or pv.shape[-1] != A: continue
            ac_list.append(ac); pv_list.append(pv)
        if not ac_list: continue
        ac = torch.stack(ac_list)          # (N, 16, A)
        pv = torch.stack(pv_list)          # (N, 16, A)
        N = ac.shape[0]
        vg = sc['action_var_globals'][eid].view(1, 1, -1)
        m = pv.mean(dim=1, keepdim=True)
        S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
        lam = 16 / (S + 16 * vg)
        xn = ((ac - m) * lam.sqrt()).transpose(1, 2)    # (N, A, 16)
        eid_t = torch.full((N,), eid, dtype=torch.long)
        # encode → codes
        codes, _ = vae.encode_with_soft(xn, eid_t, tau=0.1)
        idx_set = codes[0]                  # (N, T_bottleneck)
        # decode
        recon = vae.decode_from_indices(codes, eid_t)    # (N, A, 16)
        # de-normalize back to action units
        inv = (1.0 / lam.sqrt()).transpose(1, 2)         # (N, 1, A)... wait shape
        # lam shape (N, 1, A); inv shape (N, 1, A)
        inv_sqrt = (1.0 / lam.sqrt())                    # (N, 1, A)
        # recon is (N, A, 16) → transpose to (N, 16, A) for comparison
        recon_t = recon.transpose(1, 2)                  # (N, 16, A)
        # un-normalize: ac_recon = recon_normalized * (1/sqrt(lam)) + m
        ac_recon = recon_t * inv_sqrt + m                # (N, 16, A)

        mse_total = ((ac_recon - ac) ** 2).mean().item()
        mse_per_dim = ((ac_recon - ac) ** 2).mean(dim=(0, 1)).tolist()    # (A,)
        mse_normalized = ((recon - xn) ** 2).mean().item()                 # in normalized space

        # codebook usage
        codes_flat = idx_set.flatten().tolist()
        usage = Counter(codes_flat)
        n_unique = len(usage)
        # entropy
        total = sum(usage.values())
        probs = np.array([c / total for c in usage.values()])
        H = -(probs * np.log(probs + 1e-12)).sum()
        H_max = np.log(K)

        per_emb_stats[emb] = dict(
            n=N, mse_total=mse_total, mse_per_dim=mse_per_dim, mse_normalized=mse_normalized,
            n_unique_codes=n_unique, code_entropy=H, code_entropy_norm=H/H_max,
            codes_used=set(usage.keys()),
        )

    # ============ REPORT ============
    print(f"\n{'='*78}\nPER-EMBODIMENT VAE RECONSTRUCTION (N={N_PER_EMB} held-out chunks each)\n{'='*78}")
    print(f"\n{'emb':<14s} {'N':>4s} {'MSE_action':>11s} {'MSE_norm':>10s} {'unique_codes':>13s} {'entropy/Hmax':>13s}")
    for emb, st in sorted(per_emb_stats.items(), key=lambda x: x[1]['mse_total']):
        print(f"  {emb:<12s} {st['n']:>4d}  {st['mse_total']:>10.5f}  {st['mse_normalized']:>9.4f}  "
              f"{st['n_unique_codes']:>5d}/{K:<4d}     {st['code_entropy_norm']:>10.3f}")

    print(f"\n{'='*78}\nMSE PER ACTION DIM (raw units; usually pos+rot+gripper)\n{'='*78}")
    print(f"{'emb':<14s} " + ' '.join(f"d{i:<7d}" for i in range(A)))
    for emb, st in per_emb_stats.items():
        dims = ' '.join(f"{d:.5f}" for d in st['mse_per_dim'])
        print(f"  {emb:<12s} {dims}")

    # Code overlap (Jaccard)
    print(f"\n{'='*78}\nCODE-OVERLAP MATRIX (Jaccard: shared codes / union; 1.0 = identical usage)\n{'='*78}")
    embs = list(per_emb_stats.keys())
    header = ' '*14 + ' '.join(f"{e[:6]:>7s}" for e in embs)
    print(header)
    for e1 in embs:
        c1 = per_emb_stats[e1]['codes_used']
        row = f"  {e1:<12s}  "
        for e2 in embs:
            c2 = per_emb_stats[e2]['codes_used']
            j = len(c1 & c2) / max(len(c1 | c2), 1)
            row += f"{j:>7.3f}"
        print(row)


if __name__ == '__main__':
    main()
