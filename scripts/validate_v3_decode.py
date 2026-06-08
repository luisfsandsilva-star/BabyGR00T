#!/usr/bin/env python3
"""Offline validation of the v3 inference action-decode+denorm path BEFORE sim.

Catches the class of bug documented in memory (wrong denorm → actions 5-13× too
small). Takes held-out widowx samples, runs the FULL inference path exactly as
run_simpler_bench_v3.py does (per-image norm → cnn → policy full-mask → decode →
precision denorm using the sample's OWN lookback as prev_chunk), and compares
predicted actions vs ground-truth:
  - per-dim magnitude ratio  std(pred)/std(GT)   (should be ~0.6-1.0)
  - per-dim correlation      corr(pred, GT)        (should be >0.5 for translation)
  - overall |pred| vs |GT|

This uses the TRAINING t5 cache (real task strings) — text path is incidental
to the decode-magnitude check.
"""
import os, sys, glob, random
os.environ['CUDA_VISIBLE_DEVICES'] = ''
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm, ActionVQVAE1d, VQ1d_EMA)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset, EMBODIMENTS, EMBODIMENT_ID)

torch.set_num_threads(4); random.seed(0); torch.manual_seed(0)
CKPT = 'data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt'
ROBOT = 'widowx'; N = 32


def main():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False); a = ck['args']
    print(f"ckpt step {ck.get('step')}, dim={a['dim']}")
    DIM = a['dim']; PFX = a['n_emb_prefix']; IMG = a['img_size']; MAX_TEXT = a['max_text']
    H, L = a['H_outer'], a['L_inner']
    N_EMB = len(EMBODIMENTS) + 1
    eid = torch.tensor([EMBODIMENT_ID[ROBOT]], dtype=torch.long)

    vck = torch.load(f'data/ckpts/oxe_vqvae_{ROBOT}.pt', map_location='cpu', weights_only=False)
    adim = vck['action_dim']; K = vck.get('k', 256)
    vae = ActionVQVAE1d(action_dim=adim, vq_cls=VQ1d_EMA, k=K); vae.load_state_dict(vck['vae']); vae.eval()
    var_global = vck['action_var_global'].view(1, 1, -1)
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)['var_global']
    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)

    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=a['cnn_expand'],
                       out_dim=a['cnn_out_dim'], norm=a['cnn_norm'], pos_emb=a['cnn_pe'],
                       img_size=IMG, dropout=0.0, n_embodiments=0)
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
    cnn_proj = nn.Linear(a['cnn_out_dim'], DIM); text_proj = nn.Linear(512, DIM)
    kv_norm = ScaleNorm(DIM)
    se_keys = sorted({k.split('.')[0] for k in ck['state_encoders'].keys()})
    state_encoders = nn.ModuleDict({e: nn.Sequential(nn.Linear(8, DIM), nn.GELU(), nn.Linear(DIM, DIM)) for e in se_keys})
    emb_id_emb = nn.Embedding(N_EMB, DIM * PFX)
    n_vis = (IMG // 32) ** 2
    policy = STRMPolicy(seq_lens=(4,), k_codebook=K, dim=DIM, heads=a['heads'], kv_heads=a.get('kv_heads'),
                        ff_hidden=a.get('ff_hidden'), depth=a['depth'], L_inner=L, H_outer=H, state_dim=DIM,
                        max_prefix=n_vis+MAX_TEXT+16+PFX, weighting=a['weighting'], update_mode=a['update_mode'],
                        alpha_parametrization=a['alpha_parametrization'], alpha_per_dim=a['alpha_per_dim'],
                        n_embodiments=N_EMB, per_emb_head=a['per_emb_head'], dropout=0.0, g_input_noise=0.0)
    mods = [('cnn',cnn),('text_agg',text_agg),('cnn_proj',cnn_proj),('text_proj',text_proj),
            ('kv_norm',kv_norm),('state_encoders',state_encoders),('emb_id_emb',emb_id_emb),('policy',policy)]
    for nm, m in mods: m.load_state_dict(ck[nm])
    # EMA swap
    if 'ema_params' in ck:
        for nm, m in mods:
            sd = m.state_dict()
            for k in list(sd.keys()):
                full = f"{nm}.{k}"
                if full in ck['ema_params']: sd[k] = ck['ema_params'][full].to(sd[k].dtype)
            m.load_state_dict(sd)
        print("[ema] swapped EMA params")
    for nm, m in mods: m.eval()

    # held-out samples
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=16)
            if sp.chunk_index and sp.robot == ROBOT: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    rng = random.Random(98765); pool = list(range(len(ds))); rng.shuffle(pool)

    preds, gts = [], []
    n = 0
    for idx in pool:
        if n >= N: break
        try:
            fr, st, ac, pv, tk, _, _ = ds[idx]
            if ac.shape[-1] != adim or st.shape[-1] != 8: continue
        except Exception: continue
        from PIL import Image
        pil = fr.convert('RGB').resize((IMG, IMG))
        x = torch.from_numpy(np.asarray(pil).copy()).permute(2,0,1).float()[None] / 255.
        x = normalize_image(x, img_var)
        with torch.no_grad():
            vtok, _ = cnn(x); vtok = cnn_proj(vtok)
            # text
            t5s = torch.zeros(9, 1, MAX_TEXT, 512)
            e = t5['embeddings'].get(tk)
            if e is not None:
                h = e['hidden'].float(); tt = min(h.shape[1], MAX_TEXT); LL = min(h.shape[0], 9)
                t5s[:LL, 0, :tt, :] = h[:LL, :tt, :]
            ttok = text_proj(text_agg([t5s[l] for l in range(9)]))
            etok = emb_id_emb(eid).view(1, PFX, DIM)
            vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
            s_enc = state_encoders[ROBOT](st[None])
            all_logits = policy(None, vis, s_enc, mask_list=None, n_outer=H, n_inner=L, emb_id=eid)
            code = all_logits[-1][0][..., :K].argmax(-1)
            xn = vae.decode_from_indices([code]).transpose(1, 2)   # (1,16,7)
            # denorm using the sample's OWN lookback (pv) as prev_chunk
            pc = pv[None]; nT = pc.shape[1]; m = pc.mean(dim=1, keepdim=True)
            S = ((pc - m) ** 2).sum(dim=1, keepdim=True)
            lam = nT / (S + nT * var_global)
            ac_pred = (xn / lam.sqrt() + m)[0]    # (16,7)
        preds.append(ac_pred.numpy()); gts.append(ac.numpy()); n += 1

    preds = np.stack(preds); gts = np.stack(gts)   # (N,16,7)
    print(f"\nvalidated on {n} held-out widowx samples")
    print(f"\n{'dim':>4s} {'std(pred)':>10s} {'std(GT)':>10s} {'ratio':>7s} {'corr':>7s}  {'meaning'}")
    names = ['x','y','z','roll','pitch','yaw','grip']
    for d in range(adim):
        p = preds[:, :, d].ravel(); g = gts[:, :, d].ravel()
        sp, sg = p.std(), g.std()
        corr = np.corrcoef(p, g)[0, 1] if sp > 1e-8 and sg > 1e-8 else 0.0
        print(f"{names[d]:>4s} {sp:>10.4f} {sg:>10.4f} {sp/max(sg,1e-8):>7.2f} {corr:>7.3f}")
    overall_p = np.abs(preds).mean(); overall_g = np.abs(gts).mean()
    print(f"\noverall |pred|={overall_p:.4f}  |GT|={overall_g:.4f}  ratio={overall_p/overall_g:.2f}")
    print(f"action MSE = {((preds-gts)**2).mean():.5f}")
    print(f"\n(healthy: per-dim ratio ~0.6-1.0, translation corr >0.5, overall ratio ~0.7-1.2;")
    print(f" the memory's bug had ratio ~0.1 = actions 5-13× too small)")


if __name__ == '__main__':
    main()
