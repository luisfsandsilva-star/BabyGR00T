#!/usr/bin/env python3
"""v14 gradient diagnostic: per-layer ||grad||, special focus on ρ_L/ρ_H gradients
to understand why the gates are stuck near 0.5. CPU only."""
import os, glob, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, torch, torch.nn as nn
from babygroot_strm import STRMPolicy, LayerAggregator, ScaleNorm, ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image
import random

CKPT = 'data/ckpts/oxe_policy_v14_peremb.pt'
B = 16  # small batch for CPU
torch.set_num_threads(2); random.seed(0); torch.manual_seed(0); np.random.seed(0)


def main():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    args = ck['args']

    # load per-emb VAEs
    vaes, var_globals, per_emb_ad = {}, {}, {}
    for emb in EMBODIMENTS:
        ck_path = f'data/ckpts/oxe_vqvae_{emb}.pt'
        if not os.path.isfile(ck_path): continue
        c = torch.load(ck_path, map_location='cpu', weights_only=False)
        vae = ActionVQVAE1d(action_dim=c['action_dim'], vq_cls=VQ1d_EMA, k=c.get('k', 128))
        vae.load_state_dict(c['vae']); vae.eval()
        vaes[emb] = vae
        var_globals[emb] = c['action_var_global'].view(1, 1, -1)
        per_emb_ad[emb] = c['action_dim']

    # rebuild model
    n_emb_total = len(EMBODIMENTS) + 1
    dim = args['dim']
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=args['cnn_expand'],
                       out_dim=args['cnn_out_dim'], norm=args['cnn_norm'], pos_emb=args['cnn_pe'],
                       img_size=args['img_size'], dropout=0.0,
                       n_embodiments=n_emb_total if args.get('cnn_film_by_emb') else 0)
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
    cnn_proj = nn.Linear(args['cnn_out_dim'], dim)
    text_proj = nn.Linear(512, dim)
    kv_norm = ScaleNorm(dim)
    present_emb = sorted(vaes.keys())
    state_encoders = nn.ModuleDict({e: nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
                                     for e in present_emb})
    emb_id_emb = nn.Embedding(n_emb_total, dim * args['n_emb_prefix'])
    n_vis = (args['img_size'] // 32) ** 2
    max_prefix = n_vis + args['max_text'] + 16 + args['n_emb_prefix']
    policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=dim, heads=args['heads'],
                        kv_heads=args.get('kv_heads'), ff_hidden=args.get('ff_hidden'),
                        depth=args['depth'], L_inner=args['L_inner'], H_outer=args['H_outer'],
                        state_dim=dim, max_prefix=max_prefix,
                        weighting=args['weighting'],
                        update_mode=args['update_mode'],
                        alpha_parametrization=args['alpha_parametrization'],
                        alpha_per_dim=args['alpha_per_dim'],
                        n_embodiments=n_emb_total, per_emb_head=args['per_emb_head'])
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']
    for k, m in zip(keys, mods): m.load_state_dict(ck[k])
    for m in mods: m.train()           # enable grad

    # get one batch via the dataset
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=16)
            if sp.chunk_index: specs.append(sp)
        except: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    # pick B samples, balanced across embodiments
    rng = random.Random(7)
    by_emb_idx = {e: [] for e in present_emb}
    pool = list(range(len(ds))); rng.shuffle(pool)
    for idx in pool:
        di, _ = ds.flat[idx]
        emb = specs[di].robot
        if emb not in by_emb_idx or len(by_emb_idx[emb]) >= 2: continue
        by_emb_idx[emb].append(idx)
        if sum(len(v) for v in by_emb_idx.values()) >= B: break
    chosen = []
    for v in by_emb_idx.values(): chosen.extend(v)

    frames, states, actions, prevs, tasks, embs = [], [], [], [], [], []
    for idx in chosen:
        try:
            fr, st, ac, pv, tk, eid, di = ds[idx]
        except: continue
        emb = specs[di].robot
        # filter for VAE-compatible action_dim per emb
        if emb not in per_emb_ad: continue
        if ac.shape[-1] != per_emb_ad[emb] or st.shape[-1] != 8: continue
        from PIL import Image
        pil = fr.convert('RGB').resize((args['img_size'], args['img_size']))
        frames.append(torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.)
        states.append(st); actions.append(ac); prevs.append(pv); tasks.append(tk); embs.append(emb)
    Bf = len(frames)
    print(f"\nbuilt grad-diag batch: B={Bf} (embs: {dict((e, embs.count(e)) for e in set(embs))})")

    frames = torch.stack(frames); states = torch.stack(states); actions = torch.stack(actions); prevs = torch.stack(prevs)

    # forward to compute gradients
    eid_t = torch.tensor([EMBODIMENT_ID.get(r, len(EMBODIMENTS)) for r in embs], dtype=torch.long)

    # vision via FiLM
    img = normalize_image(frames, torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)['var_global'])
    vtok, _ = cnn(img, emb_id=eid_t if args.get('cnn_film_by_emb') else None)
    vtok = cnn_proj(vtok)
    # text
    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    t5s = torch.zeros(9, Bf, args['max_text'], 512)
    for b, tk in enumerate(tasks):
        e = t5['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float()
        t = min(h.shape[1], args['max_text'])
        L = min(h.shape[0], 9)
        t5s[:L, b, :t, :] = h[:L, :t, :]
    tagg = text_agg([t5s[l] for l in range(9)])
    ttok = text_proj(tagg)
    etok = emb_id_emb(eid_t).view(Bf, args['n_emb_prefix'], dim)
    vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
    # state
    s_enc = torch.zeros(Bf, dim)
    for emb in set(embs):
        mask = torch.tensor([r == emb for r in embs])
        if mask.sum() == 0: continue
        s_enc[mask] = state_encoders[emb](states[mask])
    # targets
    gt_codes = [torch.zeros(Bf, 4, dtype=torch.long)]
    for emb in set(embs):
        mask = torch.tensor([r == emb for r in embs])
        if mask.sum() == 0: continue
        ac = actions[mask]; pv = prevs[mask]
        vg = var_globals[emb]
        nT = ac.shape[1]
        m = pv.mean(dim=1, keepdim=True)
        S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
        lam = nT / (S + nT * vg)
        xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
        with torch.no_grad():
            cd, _ = vaes[emb].encode_with_soft(xn, tau=0.1)
        gt_codes[0][mask] = cd[0]

    # forward_loss
    loss, per, _ = policy.forward_loss(gt_codes, vis, s_enc, n_inner=args['L_inner'], h_max=None,
                                        n_outer=args['H_outer'],
                                        mask_ratio_max=0.6, emb_id=eid_t,
                                        label_smoothing=args['label_smoothing'],
                                        mask_sampler=args['mask_sampler'])
    print(f"\nloss = {loss.item():.4f}")
    loss.backward()

    # Collect per-param gradient stats
    print(f"\n{'='*100}")
    print(f"GRADIENT STATS BY PARAMETER (sorted by ||grad||)")
    print(f"{'='*100}")
    rows = []
    for nm, m in zip(keys, mods):
        for pname, p in m.named_parameters():
            if p.grad is None: continue
            full = f"{nm}.{pname}"
            rows.append((full, p.numel(), p.norm().item(), p.grad.norm().item(),
                        p.grad.abs().max().item()))
    rows.sort(key=lambda r: -r[3])
    print(f"\n  TOP 15 by ||grad||:")
    print(f"  {'name':<55s} {'numel':>10s} {'||W||':>10s} {'||g||':>10s} {'max|g|':>10s}")
    for nm, n, wn, gn, gx in rows[:15]:
        print(f"  {nm:<55s} {n:>10d} {wn:>10.3f} {gn:>10.3e} {gx:>10.3e}")

    # ρ gradients specifically
    print(f"\n  ρ-GATE GRADIENTS (the ones we care about — why are gates stuck?):")
    rho_rows = [r for r in rows if 'rho_' in r[0]]
    for nm, n, wn, gn, gx in rho_rows:
        print(f"  {nm:<55s} numel={n} ||grad||={gn:.3e}  max|g|={gx:.3e}")

    # FiLM gradients
    print(f"\n  FiLM gradients (vision conditioning):")
    film_rows = [r for r in rows if 'film' in r[0].lower()]
    for nm, n, wn, gn, gx in film_rows:
        print(f"  {nm:<55s} numel={n} ||grad||={gn:.3e}")

    # per-emb head gradients
    print(f"\n  per-emb head gradients:")
    head_rows = [r for r in rows if 'out_head' in r[0]][:12]
    for nm, n, wn, gn, gx in head_rows:
        print(f"  {nm:<55s} numel={n} ||W||={wn:.2f} ||grad||={gn:.3e}")


if __name__ == '__main__':
    main()
