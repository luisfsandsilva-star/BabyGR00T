#!/usr/bin/env python3
"""Full per-emb VAE diagnostic: per-dim MSE, var_globals, action scale, code distribution."""
import os, glob, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, torch
from collections import Counter
from babygroot_strm import ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset

N_PER_EMB = 256


@torch.no_grad()
def main():
    # load all per-emb VAEs + metadata
    vaes, meta = {}, {}
    for p in sorted(glob.glob('data/ckpts/oxe_vqvae_*.pt')):
        if 'backup' in p: continue
        c = torch.load(p, map_location='cpu', weights_only=False)
        emb = c.get('embodiment', os.path.basename(p).replace('oxe_vqvae_', '').replace('.pt', ''))
        vae = ActionVQVAE1d(action_dim=c['action_dim'], vq_cls=VQ1d_EMA, k=c['k'])
        vae.load_state_dict(c['vae']); vae.eval()
        vaes[emb] = vae
        meta[emb] = c

    # gather samples
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16)
            if sp.chunk_index: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    import random
    rng = random.Random(42)
    by_emb_idx = {e: [] for e in vaes}
    pool = list(range(len(ds))); rng.shuffle(pool)
    for idx in pool:
        di, _ = ds.flat[idx]
        emb = specs[di].robot
        if emb not in by_emb_idx: continue
        if len(by_emb_idx[emb]) >= N_PER_EMB: continue
        by_emb_idx[emb].append(idx)
        if all(len(v) >= N_PER_EMB for v in by_emb_idx.values()): break

    print(f"\n{'='*120}")
    print(f"VAE TRAINING / DATA META")
    print(f"{'='*120}")
    print(f"{'emb':<14s} {'K':>4s} {'n_train':>8s} {'epochs6k':>9s} {'final_recon_train':>17s} "
          f"{'var_global (per-dim sqrt)':<60s}")
    for emb, c in sorted(meta.items()):
        ntr = c.get('n_train_pairs', 0)
        epochs = (6000 * 64) / max(ntr, 1)
        vg_sqrt = c['action_var_global'].sqrt().tolist()
        vg_str = ' '.join(f"{v:.2f}" for v in vg_sqrt)
        print(f"  {emb:<12s} {c['k']:>4d} {ntr:>8d}  {epochs:>7.1f}  {'-':>17s}  {vg_str}")

    print(f"\n{'='*120}")
    print(f"PER-EMB EVAL (N={N_PER_EMB} held-out chunks)")
    print(f"{'='*120}")

    per_dim_results = {}
    for emb, idxs in sorted(by_emb_idx.items()):
        ac_list, pv_list = [], []
        for i in idxs:
            try: _, _, ac, pv, _, _, _ = ds[i]
            except: continue
            if ac.shape[-1] != vaes[emb].action_dim: continue
            ac_list.append(ac); pv_list.append(pv)
        if not ac_list:
            print(f"  {emb}: SKIP (no samples matched action_dim)"); continue
        ac = torch.stack(ac_list); pv = torch.stack(pv_list)
        N = ac.shape[0]
        vg = meta[emb]['action_var_global'].view(1, 1, -1)
        m = pv.mean(dim=1, keepdim=True)
        S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
        lam = 16 / (S + 16 * vg)
        xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
        embs, _, codes = vaes[emb].encode(xn)
        recon_xn = vaes[emb].decode(embs)
        recon_t = recon_xn.transpose(1, 2)
        ac_recon = recon_t * (1.0 / lam.sqrt()) + m
        # per-dim
        per_dim_mse = ((ac_recon - ac) ** 2).mean(dim=(0, 1))    # (A,)
        per_dim_var = ac.var(dim=(0, 1))                          # ground-truth per-dim variance
        per_dim_results[emb] = (per_dim_mse.tolist(), per_dim_var.tolist())

        # codebook stats
        codes_flat = codes[0].flatten().tolist()
        uc = Counter(codes_flat)
        n_unique = len(uc)
        probs = np.array([c/sum(uc.values()) for c in uc.values()])
        H = -(probs * np.log(probs + 1e-12)).sum()
        K = meta[emb]['k']
        H_max = np.log(K)
        # what fraction of codes carries 90% of mass?
        sorted_p = sorted(probs, reverse=True)
        cum = np.cumsum(sorted_p); n90 = int(np.argmax(cum >= 0.9)) + 1

        print(f"\n  [{emb}] (N={N}, K={K})")
        print(f"    total MSE:    {((ac_recon - ac)**2).mean().item():.5f}   (predict-mean = {((m.expand_as(ac)-ac)**2).mean().item():.5f})")
        print(f"    norm-space MSE: {((recon_xn - xn)**2).mean().item():.4f}")
        print(f"    code usage:   {n_unique}/{K} unique, top-90%-mass = {n90} codes, entropy/Hmax = {H/H_max:.2f}")

    print(f"\n{'='*120}")
    print(f"PER-DIM MSE / GROUND-TRUTH VARIANCE  (ratio: 1 = predict-mean; <1 = VAE adds value; 0 = perfect)")
    print(f"{'='*120}")
    print(f"{'emb':<14s} " + ' '.join(f"d{i}_mse/var" for i in range(7)))
    for emb, (mses, vars_) in per_dim_results.items():
        row = '  ' + f"{emb:<12s}  "
        for mse, vr in zip(mses, vars_):
            ratio = mse / max(vr, 1e-9)
            row += f"{ratio:>10.3f} "
        print(row)


if __name__ == '__main__':
    main()
