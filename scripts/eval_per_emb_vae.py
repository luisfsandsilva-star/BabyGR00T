#!/usr/bin/env python3
"""Eval each per-emb VAE's reconstruction quality on actual held-out chunks (CPU)."""
import os, glob, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, torch
from babygroot_strm import ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)

N_PER_EMB = 256


@torch.no_grad()
def main():
    # load all per-emb VAEs
    vaes, var_globals, k_list = {}, {}, {}
    for p in sorted(glob.glob('data/ckpts/oxe_vqvae_*.pt')):
        if 'backup' in p: continue
        c = torch.load(p, map_location='cpu', weights_only=False)
        emb = c.get('embodiment', os.path.basename(p).replace('oxe_vqvae_', '').replace('.pt', ''))
        vae = ActionVQVAE1d(action_dim=c['action_dim'], vq_cls=VQ1d_EMA, k=c['k'])
        vae.load_state_dict(c['vae']); vae.eval()
        vaes[emb] = vae
        var_globals[emb] = c['action_var_global'].view(1, 1, -1)
        k_list[emb] = c['k']

    print(f"loaded {len(vaes)} per-emb VAEs: {sorted(vaes.keys())}\n")

    # build dataset across all local OXE
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16)
            if sp.chunk_index: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    print(f"  {len(specs)} datasets, {len(ds)} chunks total\n")

    # collect balanced samples
    import random
    rng = random.Random(42)
    by_emb_idx = {}
    pool = list(range(len(ds))); rng.shuffle(pool)
    for idx in pool:
        di, _ = ds.flat[idx]
        emb = specs[di].robot
        if emb not in vaes: continue
        if len(by_emb_idx.get(emb, [])) >= N_PER_EMB: continue
        by_emb_idx.setdefault(emb, []).append(idx)
        if all(len(by_emb_idx.get(e, [])) >= N_PER_EMB for e in vaes): break

    # per-emb VAE recon eval
    print(f"{'='*100}")
    print(f"PER-EMB VAE RECONSTRUCTION (N={N_PER_EMB}, action units)")
    print(f"{'='*100}")
    print(f"{'emb':<14s} {'n':>4s} {'K':>4s} {'recon_MSE':>11s} {'norm_MSE':>10s} "
          f"{'predict-mean':>13s} {'random-code':>12s} {'unique':>8s}")
    print('-' * 100)

    results = []
    for emb, idxs in sorted(by_emb_idx.items()):
        if len(idxs) == 0: continue
        ac_list, pv_list = [], []
        for i in idxs:
            try:
                _, _, ac, pv, _, _, _ = ds[i]
            except Exception: continue
            adim_vae = vaes[emb].action_dim
            if ac.shape[-1] != adim_vae:
                continue                    # skip dim-mismatched datasets (libero axis-angle etc handled by VAE)
            ac_list.append(ac); pv_list.append(pv)
        if not ac_list: continue
        ac = torch.stack(ac_list)           # (N, 16, A)
        pv = torch.stack(pv_list)           # (N, 16, A)
        N = ac.shape[0]
        vg = var_globals[emb]
        m = pv.mean(dim=1, keepdim=True)
        S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
        lam = 16 / (S + 16 * vg)
        xn = ((ac - m) * lam.sqrt()).transpose(1, 2)   # (N, A, 16)

        # encode
        embs, _, codes = vaes[emb].encode(xn)
        # decode
        recon_xn = vaes[emb].decode(embs)              # (N, A, 16)
        # de-normalize
        inv_sqrt_lam = 1.0 / lam.sqrt()                # (N, 1, A)
        recon_t = recon_xn.transpose(1, 2)             # (N, 16, A)
        ac_recon = recon_t * inv_sqrt_lam + m          # (N, 16, A)

        recon_mse = ((ac_recon - ac) ** 2).mean().item()
        norm_mse = ((recon_xn - xn) ** 2).mean().item()
        # baselines
        mean_baseline = ((m.expand_as(ac) - ac) ** 2).mean().item()
        # random codes
        K = k_list[emb]
        rand_idx = torch.randint(0, K, codes[0].shape)
        rand_emb = vaes[emb].vq.emb(rand_idx).view(N, codes[0].shape[1], -1).permute(0, 2, 1)
        rand_xn = vaes[emb].decode([rand_emb])
        rand_t = rand_xn.transpose(1, 2)
        ac_rand = rand_t * inv_sqrt_lam + m
        rand_mse = ((ac_rand - ac) ** 2).mean().item()

        n_unique = len(torch.unique(codes[0]).tolist())
        print(f"{emb:<14s} {N:>4d} {K:>4d}  {recon_mse:>10.5f} {norm_mse:>10.4f}  "
              f"{mean_baseline:>11.5f}  {rand_mse:>11.5f}    {n_unique:>3d}/{K}")
        results.append((emb, recon_mse, mean_baseline, rand_mse, n_unique, K))

    # summary
    print(f"\n{'='*100}\nSUMMARY: recon_MSE / predict-mean baseline ratio (lower = VAE adds value)\n{'='*100}")
    for emb, mse, baseline, rmse, _, _ in results:
        useful_factor = baseline / mse if mse > 0 else float('inf')
        print(f"  {emb:<14s}  recon={mse:.5f}   predict-mean={baseline:.5f}   "
              f"VAE-vs-mean: {useful_factor:.1f}x better")


if __name__ == '__main__':
    main()
