#!/usr/bin/env python3
"""Evaluate v13's actual action prediction quality (not just CE accuracy).

Pipeline:
  1. Load latest v13 ckpt + shared VAE (CPU-only, no GPU contention)
  2. Sample N held-out chunks per embodiment
  3. For each chunk:
     a. encode_modalities (vision + text + state + emb_id) — same as training
     b. run policy.forward with FULL MASK (predict all 4 action codes from scratch)
     c. argmax over each token's logits → predicted code indices
     d. decode_from_indices through shared VAE → predicted normalized action sequence
     e. de-normalize (undo precision norm) → predicted actions
     f. compare to GT actions: per-dim MSE, per-emb MSE
  4. Report baselines:
     - Random codes (samples uniformly from K=512): MSE upper bound
     - GT codes (the discretization floor): MSE lower bound — best the model could do
     - Predict-mean (predict mean action across chunk): trivial baseline
"""
import os, sys, json, glob, random
os.environ['CUDA_VISIBLE_DEVICES'] = ''
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.cond_vae import CondActionVQVAE1d
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image

CKPT = 'data/ckpts/oxe_policy_v13_scaled.pt'
VAE_CKPT = 'data/ckpts/oxe_shared_vae.pt'
T5_CACHE = 'data/cache/t5_text_cache_paraphrased.pt'
IMG_VAR  = 'data/cache/image_var_global.pt'
N_PER_EMB = 32                              # how many chunks to eval per embodiment
DEV = 'cpu'

torch.set_num_threads(2)                    # don't peg the box
random.seed(42); torch.manual_seed(42); np.random.seed(42)


def build():
    print(f"loading v13 ckpt + VAE...")
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    sc = torch.load(VAE_CKPT, map_location=DEV, weights_only=False)
    print(f"  v13 step: {ck['step']}")
    print(f"  shared VAE: K={sc['k']}, n_embodiments={sc['n_embodiments']}")

    args = ck['args']
    # build same modules as training
    n_emb_total = len(EMBODIMENTS) + 1
    n_emb_prefix = args['n_emb_prefix']
    dim = args['dim']
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=args['cnn_expand'],
                       out_dim=args['cnn_out_dim'], norm=args['cnn_norm'], pos_emb=args['cnn_pe'],
                       img_size=args['img_size'], dropout=0.0)              # no dropout for eval
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
    cnn_proj = nn.Linear(args['cnn_out_dim'], dim)
    text_proj = nn.Linear(512, dim)
    kv_norm = ScaleNorm(dim)
    present_emb = ['widowx', 'google_robot', 'franka', 'ur5', 'jaco_2']
    state_encoders = nn.ModuleDict({e: nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
                                     for e in present_emb})
    emb_id_emb = nn.Embedding(n_emb_total, dim * n_emb_prefix)
    n_vis = (args['img_size'] // 32) ** 2
    max_prefix = n_vis + args['max_text'] + 16 + n_emb_prefix
    policy = STRMPolicy(seq_lens=(4,), k_codebook=sc['k'], dim=dim, heads=args['heads'],
                        kv_heads=args.get('kv_heads'), ff_hidden=args.get('ff_hidden'),
                        depth=args['depth'], L_inner=args['L_inner'], H_outer=args['H_outer'],
                        state_dim=dim, max_prefix=max_prefix,
                        weighting=args['weighting'],
                        update_mode=args['update_mode'],
                        alpha_parametrization=args['alpha_parametrization'],
                        alpha_per_dim=args['alpha_per_dim'],
                        n_embodiments=n_emb_total,
                        per_emb_head=args['per_emb_head'],
                        dropout=0.0)
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']
    for k, m in zip(keys, mods): m.load_state_dict(ck[k])
    for m in mods: m.eval()

    vae = CondActionVQVAE1d(action_dim=sc['action_dim'], n_embodiments=sc['n_embodiments'], k=sc['k'])
    vae.load_state_dict(sc['vae']); vae.eval()

    # T5 cache for tasks
    t5 = torch.load(T5_CACHE, map_location='cpu', weights_only=False)
    img_var = torch.load(IMG_VAR, map_location='cpu', weights_only=False)
    var_global_img = img_var['var_global']

    # var_globals from VAE (same dict construction as train_oxe.py)
    var_globals = {emb: sc['action_var_globals'][EMBODIMENT_ID[emb]].view(1, 1, -1)
                   for emb in present_emb}

    return dict(cnn=cnn, text_agg=text_agg, cnn_proj=cnn_proj, text_proj=text_proj,
                kv_norm=kv_norm, state_encoders=state_encoders, emb_id_emb=emb_id_emb,
                policy=policy, vae=vae, t5_emb=t5['embeddings'], var_global_img=var_global_img,
                var_globals=var_globals, args=args, step=ck['step'])


@torch.no_grad()
def predict_codes(env, frame_t, state, task, emb_robot):
    """Run the full v13 forward pass on one chunk → predicted code indices (shape (1,4))."""
    a = env['args']
    # vision
    img = normalize_image(frame_t.unsqueeze(0), env['var_global_img'])
    vtok, _ = env['cnn'](img); vtok = env['cnn_proj'](vtok)
    # text
    e = env['t5_emb'].get(task)
    B, T_text = 1, a['max_text']
    t5s = torch.zeros(9, B, T_text, 512)
    if e is not None:
        h = e['hidden'].float()
        t = min(h.shape[1], T_text)
        t5s[:, 0, :t, :] = h[:, :t, :]
    tagg = env['text_agg']([t5s[l] for l in range(9)])
    ttok = env['text_proj'](tagg)
    # emb id prefix
    eid = torch.tensor([EMBODIMENT_ID.get(emb_robot, len(EMBODIMENTS))], dtype=torch.long)
    etok = env['emb_id_emb'](eid).view(1, a['n_emb_prefix'], a['dim'])
    vis = env['kv_norm'](torch.cat([etok, vtok, ttok], dim=1))
    # state
    s_enc = env['state_encoders'][emb_robot](state.unsqueeze(0))
    # full-mask: indices all = mask_idx (= K), so policy predicts from scratch
    K = env['vae'].vqs[0].K
    indices = [torch.full((1, 4), K, dtype=torch.long)]    # K = MASK marker
    mask_list = [torch.ones(1, 4, dtype=torch.bool)]
    # forward
    eid_t = torch.tensor([EMBODIMENT_ID.get(emb_robot, len(EMBODIMENTS))], dtype=torch.long)
    all_logits = env['policy'](indices, vis, s_enc, mask_list=mask_list, emb_id=eid_t)
    # take last cycle's logits, argmax
    logits = all_logits[-1][0]                              # (1, 4, K)
    pred_idx = logits.argmax(dim=-1)                        # (1, 4)
    return pred_idx, eid_t


@torch.no_grad()
def main():
    env = build()
    sc = torch.load(VAE_CKPT, map_location=DEV, weights_only=False)
    vae = env['vae']
    print(f"\nbuilding dataset specs (local only)...")
    specs = []
    for ds_dir in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(ds_dir, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(ds_dir, chunk_len=16, lookback=16, chunk_stride=16)   # no overlap for eval
            if sp.chunk_index: specs.append(sp)
        except Exception as e:
            pass
    print(f"  {len(specs)} datasets loaded for eval")
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)

    # collect samples balanced by emb
    by_emb = {}
    rng = random.Random(42)
    indices_pool = list(range(len(ds)))
    rng.shuffle(indices_pool)
    for idx in indices_pool:
        di, _ = ds.flat[idx]
        emb = specs[di].robot
        if emb not in ['widowx', 'google_robot', 'franka', 'ur5', 'jaco_2']: continue
        if len(by_emb.get(emb, [])) >= N_PER_EMB: continue
        by_emb.setdefault(emb, []).append(idx)
        if all(len(by_emb.get(e, [])) >= N_PER_EMB for e in ['widowx','google_robot','franka','ur5','jaco_2']):
            break

    print(f"\nsamples per emb: { {e: len(v) for e, v in by_emb.items()} }")
    print(f"\nrunning eval (v13 step={env['step']})...\n")

    results = []
    rng_codes = torch.Generator().manual_seed(42)
    for emb, idxs in by_emb.items():
        for i in idxs:
            try:
                frame, st, ac_gt, pv, task, emb_id_int, di = ds[i]
            except Exception:
                continue
            # filter: action_dim must match the VAE (7)
            if ac_gt.shape[-1] != sc['action_dim']:
                continue
            # state_dim must be 8 (state encoder expects this)
            if st.shape[-1] != 8:
                continue
            # Match training pipeline: resize to 224x224, convert to (3, H, W) float in [0,1]
            from PIL import Image as _PI
            if hasattr(frame, 'mode'):
                pil = frame.convert('RGB').resize((env['args']['img_size'], env['args']['img_size']))
                ft = torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.
            else:
                ft = frame.float()
                if ft.shape[0] != 3:
                    ft = ft.permute(2, 0, 1) if ft.shape[-1] == 3 else ft

            # === predict codes ===
            pred_idx, eid_t = predict_codes(env, ft, st, task, emb)
            # === GT codes (the discretization floor) ===
            vg = env['var_globals'][emb]
            m_pv = pv.mean(dim=0, keepdim=True)
            S = ((pv - m_pv) ** 2).sum(dim=0, keepdim=True)
            lam = 16 / (S + 16 * vg.squeeze(0))
            xn = ((ac_gt - m_pv) * lam.sqrt()).transpose(0, 1).unsqueeze(0)  # (1, A, 16)
            gt_codes, _ = vae.encode_with_soft(xn, eid_t, tau=0.1)
            # === random codes ===
            rand_idx = torch.randint(0, vae.vqs[0].K, (1, 4), generator=rng_codes)

            # === decode all three back to normalized action sequences ===
            xn_pred = vae.decode_from_indices([pred_idx], eid_t)            # (1, A, 16)
            xn_gt_codes = vae.decode_from_indices([gt_codes[0]], eid_t)     # discretization floor
            xn_rand = vae.decode_from_indices([rand_idx], eid_t)

            # === undo precision norm to get actions ===
            inv_lam = (1.0 / lam.sqrt()).transpose(0, 1).unsqueeze(0)        # (1, 16... wait shape)
            # xn shape: (1, A, 16). We need to multiply by 1/sqrt(lam) and add back mean.
            # lam shape (1, A); inv_sqrt_lam shape (1, A, 1)
            inv = (1.0 / lam.sqrt()).reshape(1, -1, 1)                       # (1, A, 1)
            mean_shift = m_pv.transpose(0, 1).unsqueeze(0)                   # (1, A, 1)
            ac_pred = (xn_pred * inv + mean_shift).squeeze(0).transpose(0, 1)        # (16, A)
            ac_gt_floor = (xn_gt_codes * inv + mean_shift).squeeze(0).transpose(0, 1)
            ac_rand = (xn_rand * inv + mean_shift).squeeze(0).transpose(0, 1)

            # === MSE ===
            mse_pred = ((ac_pred - ac_gt) ** 2).mean().item()
            mse_floor = ((ac_gt_floor - ac_gt) ** 2).mean().item()
            mse_rand = ((ac_rand - ac_gt) ** 2).mean().item()
            # baseline: predict the mean of the lookback as the next 16 frames
            ac_mean_baseline = m_pv.expand_as(ac_gt)
            mse_mean = ((ac_mean_baseline - ac_gt) ** 2).mean().item()
            # token acc
            tok_acc = (pred_idx == gt_codes[0]).float().mean().item()

            results.append(dict(emb=emb, mse_pred=mse_pred, mse_floor=mse_floor,
                                mse_rand=mse_rand, mse_mean=mse_mean, tok_acc=tok_acc))

    # aggregate
    import statistics
    def agg(rs, key):
        vals = [r[key] for r in rs]
        return statistics.mean(vals), statistics.median(vals)

    print(f"\n{'='*70}\nEVAL ON {len(results)} HELD-OUT SAMPLES (v13 step {env['step']})\n{'='*70}")
    print(f"\n{'metric':<35s} {'mean':>10s} {'median':>10s}")
    for label, key in [('predicted action MSE',          'mse_pred'),
                        ('discretization floor MSE (VAE re-encode)', 'mse_floor'),
                        ('random-codes baseline MSE',     'mse_rand'),
                        ('predict-mean baseline MSE',     'mse_mean'),
                        ('per-token code accuracy',       'tok_acc')]:
        m, md = agg(results, key)
        print(f"  {label:<33s} {m:>10.5f} {md:>10.5f}")

    print(f"\nper-embodiment predicted MSE:")
    for emb in sorted(set(r['emb'] for r in results)):
        sub = [r for r in results if r['emb'] == emb]
        m, md = agg(sub, 'mse_pred')
        mf, _ = agg(sub, 'mse_floor')
        mt, _ = agg(sub, 'tok_acc')
        print(f"  {emb:<14s} n={len(sub):>3}: MSE={m:.5f} (floor={mf:.5f}, gap={m-mf:.5f})  tok_acc={mt:.1%}")


if __name__ == '__main__':
    main()
