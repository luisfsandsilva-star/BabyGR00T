#!/usr/bin/env python3
"""End-to-end eval of v14: action MSE per emb + gradient + α stats. CPU."""
import os, glob, json, random
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
import numpy as np, torch, torch.nn as nn
from collections import defaultdict
from babygroot_strm import STRMPolicy, LayerAggregator, ScaleNorm, ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image

CKPT = 'data/ckpts/oxe_policy_v14_peremb.pt'
N_PER_EMB = 32
torch.set_num_threads(2)
random.seed(42); torch.manual_seed(42); np.random.seed(42)


def build():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    args = ck['args']
    print(f"v14 step: {ck['step']}, dim={args['dim']}, H={args['H_outer']}, K=256 per-emb")

    # per-emb VAEs
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
    print(f"  {len(vaes)} per-emb VAEs loaded")

    # modules
    n_emb_total = len(EMBODIMENTS) + 1
    n_emb_prefix = args['n_emb_prefix']
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
    emb_id_emb = nn.Embedding(n_emb_total, dim * n_emb_prefix)
    n_vis = (args['img_size'] // 32) ** 2
    max_prefix = n_vis + args['max_text'] + 16 + n_emb_prefix
    policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=dim, heads=args['heads'],
                        kv_heads=args.get('kv_heads'), ff_hidden=args.get('ff_hidden'),
                        depth=args['depth'], L_inner=args['L_inner'], H_outer=args['H_outer'],
                        state_dim=dim, max_prefix=max_prefix,
                        weighting=args['weighting'],
                        update_mode=args['update_mode'],
                        alpha_parametrization=args['alpha_parametrization'],
                        alpha_per_dim=args['alpha_per_dim'],
                        n_embodiments=n_emb_total,
                        per_emb_head=args['per_emb_head'], dropout=0.0)
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']
    for k, m in zip(keys, mods): m.load_state_dict(ck[k])
    for m in mods: m.eval()

    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)

    return dict(cnn=cnn, text_agg=text_agg, cnn_proj=cnn_proj, text_proj=text_proj,
                kv_norm=kv_norm, state_encoders=state_encoders, emb_id_emb=emb_id_emb,
                policy=policy, vaes=vaes, var_globals=var_globals, per_emb_ad=per_emb_ad,
                t5_emb=t5['embeddings'], var_global_img=img_var['var_global'],
                args=args, step=ck['step'], present_emb=present_emb)


@torch.no_grad()
def predict_codes(env, frame_t, state, task, emb):
    a = env['args']; dim = a['dim']
    img = normalize_image(frame_t.unsqueeze(0), env['var_global_img'])
    eid = torch.tensor([EMBODIMENT_ID.get(emb, len(EMBODIMENTS))], dtype=torch.long)
    # FiLM-conditioned vision
    vtok, _ = env['cnn'](img, emb_id=eid if a.get('cnn_film_by_emb') else None)
    vtok = env['cnn_proj'](vtok)
    e = env['t5_emb'].get(task)
    T_text = a['max_text']
    t5s = torch.zeros(9, 1, T_text, 512)
    if e is not None:
        h = e['hidden'].float()
        t = min(h.shape[1], T_text)
        L = min(h.shape[0], 9)
        t5s[:L, 0, :t, :] = h[:L, :t, :]
    tagg = env['text_agg']([t5s[l] for l in range(9)])
    ttok = env['text_proj'](tagg)
    etok = env['emb_id_emb'](eid).view(1, a['n_emb_prefix'], dim)
    vis = env['kv_norm'](torch.cat([etok, vtok, ttok], dim=1))
    s_enc = env['state_encoders'][emb](state.unsqueeze(0))
    K = 256
    indices = [torch.full((1, 4), K, dtype=torch.long)]
    mask_list = [torch.ones(1, 4, dtype=torch.bool)]
    all_logits = env['policy'](indices, vis, s_enc, mask_list=mask_list, emb_id=eid)
    pred_idx = all_logits[-1][0].argmax(dim=-1)
    return pred_idx, eid


@torch.no_grad()
def main():
    env = build()
    print(f"\nbuilding eval dataset...")
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=16)
            if sp.chunk_index: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    rng = random.Random(42)
    pool = list(range(len(ds))); rng.shuffle(pool)
    by_emb_idx = {e: [] for e in env['present_emb']}
    for idx in pool:
        di, _ = ds.flat[idx]
        emb = specs[di].robot
        if emb not in by_emb_idx: continue
        adim_vae = env['per_emb_ad'][emb]
        if len(by_emb_idx[emb]) >= N_PER_EMB: continue
        by_emb_idx[emb].append(idx)

    print(f"\neval samples per emb: { {e: len(v) for e, v in by_emb_idx.items()} }")
    print(f"\nrunning eval (v14 step={env['step']})...\n")

    results = []
    rng_codes = torch.Generator().manual_seed(42)
    for emb in sorted(by_emb_idx):
        idxs = by_emb_idx[emb]
        if not idxs: continue
        vae = env['vaes'][emb]; vg = env['var_globals'][emb]
        adim = env['per_emb_ad'][emb]
        for i in idxs:
            try:
                frame, st, ac_gt, pv, task, emb_id_int, di = ds[i]
            except Exception: continue
            if ac_gt.shape[-1] != adim or st.shape[-1] != 8: continue
            from PIL import Image as _PI
            pil = frame.convert('RGB').resize((env['args']['img_size'], env['args']['img_size']))
            ft = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.

            pred_idx, eid_t = predict_codes(env, ft, st, task, emb)
            m_pv = pv.mean(dim=0, keepdim=True)
            S = ((pv - m_pv) ** 2).sum(dim=0, keepdim=True)
            lam = 16 / (S + 16 * vg.squeeze(0))
            xn = ((ac_gt - m_pv) * lam.sqrt()).transpose(0, 1).unsqueeze(0)
            gt_codes, _ = vae.encode_with_soft(xn, tau=0.1)
            rand_idx = torch.randint(0, vae.vq.K, (1, 4), generator=rng_codes)

            xn_pred = vae.decode_from_indices([pred_idx])
            xn_gt_floor = vae.decode_from_indices([gt_codes[0]])
            xn_rand = vae.decode_from_indices([rand_idx])

            inv = (1.0 / lam.sqrt()).reshape(1, -1, 1)
            mean_shift = m_pv.transpose(0, 1).unsqueeze(0)
            ac_pred = (xn_pred * inv + mean_shift).squeeze(0).transpose(0, 1)
            ac_gt_floor = (xn_gt_floor * inv + mean_shift).squeeze(0).transpose(0, 1)
            ac_rand = (xn_rand * inv + mean_shift).squeeze(0).transpose(0, 1)

            mse_pred = ((ac_pred - ac_gt) ** 2).mean().item()
            mse_floor = ((ac_gt_floor - ac_gt) ** 2).mean().item()
            mse_rand = ((ac_rand - ac_gt) ** 2).mean().item()
            ac_mean_baseline = m_pv.expand_as(ac_gt)
            mse_mean = ((ac_mean_baseline - ac_gt) ** 2).mean().item()
            tok_acc = (pred_idx == gt_codes[0]).float().mean().item()
            results.append(dict(emb=emb, mse_pred=mse_pred, mse_floor=mse_floor,
                                mse_rand=mse_rand, mse_mean=mse_mean, tok_acc=tok_acc))

    if not results:
        print("no eval samples got through — check dataset compat"); return

    import statistics
    def agg(rs, key):
        v = [r[key] for r in rs]
        return statistics.mean(v), statistics.median(v)

    print(f"\n{'='*72}\nEVAL ON {len(results)} HELD-OUT SAMPLES (v14 step {env['step']})\n{'='*72}")
    print(f"\n{'metric':<40s} {'mean':>10s} {'median':>10s}")
    for label, key in [('predicted action MSE', 'mse_pred'),
                        ('discretization floor MSE (VAE recon GT)', 'mse_floor'),
                        ('random-codes baseline MSE', 'mse_rand'),
                        ('predict-mean baseline MSE', 'mse_mean'),
                        ('per-token code accuracy', 'tok_acc')]:
        m, md = agg(results, key)
        print(f"  {label:<38s} {m:>10.5f} {md:>10.5f}")

    print(f"\nper-emb predicted MSE:")
    for emb in sorted(set(r['emb'] for r in results)):
        sub = [r for r in results if r['emb'] == emb]
        m, _ = agg(sub, 'mse_pred')
        mf, _ = agg(sub, 'mse_floor')
        mm, _ = agg(sub, 'mse_mean')
        mt, _ = agg(sub, 'tok_acc')
        ratio_to_floor = m / max(mf, 1e-9)
        ratio_to_mean = m / max(mm, 1e-9)
        print(f"  {emb:<14s} n={len(sub):>3}: MSE={m:.5f} (floor={mf:.5f}, "
              f"{ratio_to_floor:.1f}× floor)  tok_acc={mt:.1%}  vs predict-mean: {ratio_to_mean:.2f}×")


if __name__ == '__main__':
    main()
