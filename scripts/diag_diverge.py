#!/usr/bin/env python3
"""Compare BEFORE (best @ step 7000) vs AFTER (broken @ step 9000) ckpts to
locate the source of the catastrophic gradient explosion at step 8800-9000.

Investigates:
  1. Per-param weight-norm DELTA across the 2000 steps between ckpts.
  2. Forward-pass activation magnitudes (z_L per inner iter, z_H per outer)
     to find where activations explode.
  3. Empirical Lipschitz of g(x, kv) estimated by finite differences:
     ||g(x + δ) - g(x)|| / ||δ||  for small random δ. If > 1 (with ρ_L=0.5 and
     damped iter z_L = 0.5 z_L + 0.5 g(...)), the inner loop diverges
     (Banach contraction requires Lipschitz < (1/α - 1) = 1).
"""
import os, sys, glob, random
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm,
                             ActionVQVAE1d, VQ1d_EMA)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image

CKPT_BEFORE = 'data/ckpts/oxe_policy_v14_widowx_v2_resumed_best.pt'   # step 7000, val 19.33%
CKPT_AFTER  = 'data/ckpts/oxe_policy_v14_widowx_v2_resumed.pt'         # step 9000, diverged
ROBOT = 'widowx'
BS = 32
torch.set_num_threads(4); random.seed(7); torch.manual_seed(7); np.random.seed(7)


def build_modules(ck):
    args = ck['args']
    DIM = args['dim']; HEADS = args['heads']; KV = args.get('kv_heads', HEADS)
    FF = args.get('ff_hidden'); DEPTH = args['depth']
    L_I = args['L_inner']; H_O = args['H_outer']
    N_EMB = len(EMBODIMENTS) + 1
    IMG = args['img_size']; MAX_TEXT = args['max_text']; PFX = args['n_emb_prefix']
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=args['cnn_expand'],
                       out_dim=args['cnn_out_dim'], norm=args['cnn_norm'], pos_emb=args['cnn_pe'],
                       img_size=IMG, dropout=0.0,
                       n_embodiments=N_EMB if args.get('cnn_film_by_emb') else 0)
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
    cnn_proj = nn.Linear(args['cnn_out_dim'], DIM)
    text_proj = nn.Linear(512, DIM)
    kv_norm = ScaleNorm(DIM)
    se_keys = sorted({k.split('.')[0] for k in ck['state_encoders'].keys()})
    state_encoders = nn.ModuleDict({e: nn.Sequential(nn.Linear(8, DIM), nn.GELU(), nn.Linear(DIM, DIM))
                                     for e in se_keys})
    emb_id_emb = nn.Embedding(N_EMB, DIM * PFX)
    n_vis = (IMG // 32) ** 2
    max_prefix = n_vis + MAX_TEXT + 16 + PFX
    policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=DIM, heads=HEADS,
                        kv_heads=KV, ff_hidden=FF, depth=DEPTH, L_inner=L_I, H_outer=H_O,
                        state_dim=DIM, max_prefix=max_prefix,
                        weighting=args['weighting'], update_mode=args['update_mode'],
                        alpha_parametrization=args['alpha_parametrization'],
                        alpha_per_dim=args['alpha_per_dim'],
                        n_embodiments=N_EMB, per_emb_head=args['per_emb_head'], dropout=0.0,
                        g_input_noise=0.0)   # disable noise for clean diag
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']
    for k, m in zip(keys, mods):
        res = m.load_state_dict(ck[k], strict=False)
        if res.missing_keys: print(f"  [load {k}] missing: {res.missing_keys[:2]}")
        if res.unexpected_keys: print(f"  [load {k}] unexpected: {res.unexpected_keys[:2]}")
    for m in mods: m.cuda().eval()
    return mods, keys, args


def build_batch(args, ckpt_args, vae_data):
    print(f"[diag] loading data...")
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=4)
            if sp.chunk_index and sp.robot == ROBOT: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    rng = random.Random(123); pool = list(range(len(ds))); rng.shuffle(pool)
    samples = []
    vae_c, vae, var_global, adim = vae_data
    for idx in pool:
        if len(samples) >= BS: break
        try:
            fr, st, ac, pv, tk, eid_, di = ds[idx]
            if ac.shape[-1] != adim or st.shape[-1] != 8: continue
            from PIL import Image
            pil = fr.convert('RGB').resize((args['img_size'], args['img_size']))
            ft = torch.from_numpy(np.asarray(pil).copy()).permute(2,0,1).float() / 255.
            samples.append((ft, st, ac, pv, tk))
        except Exception: pass
    Bf = len(samples)
    frames = torch.stack([s[0] for s in samples]).cuda()
    states = torch.stack([s[1] for s in samples]).cuda()
    actions = torch.stack([s[2] for s in samples]).cuda()
    prevs = torch.stack([s[3] for s in samples]).cuda()
    tasks = [s[4] for s in samples]
    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    t5s = torch.zeros(9, Bf, args['max_text'], 512, device='cuda')
    for b, tk in enumerate(tasks):
        e = t5['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float().cuda()
        t = min(h.shape[1], args['max_text']); L = min(h.shape[0], 9)
        t5s[:L, b, :t, :] = h[:L, :t, :]
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)
    img_n = normalize_image(frames, img_var['var_global'].cuda())
    return img_n, t5s, states, actions, prevs, tasks


@torch.no_grad()
def forward_diag(mods, args, img_n, t5s, states, actions, prevs):
    cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy = mods
    DIM = args['dim']; PFX = args['n_emb_prefix']
    Bf = img_n.shape[0]
    eid_t = torch.tensor([EMBODIMENT_ID.get(ROBOT, len(EMBODIMENTS))] * Bf, dtype=torch.long).cuda()

    vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
    tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
    etok = emb_id_emb(eid_t).view(Bf, PFX, DIM)
    vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
    s_enc = state_encoders[ROBOT](states)

    # Manually run the policy forward to capture intermediate magnitudes
    seq_lens = policy.seq_lens
    N = sum(seq_lens)
    H = policy.H_outer; L = policy.L_inner
    rL, rH = policy._rhos()                     # current ρ_L, ρ_H values
    wL = policy._weights(rL, L, vis.device, vis.dtype)   # (L,) closed form
    wH = policy._weights(rH, H, vis.device, vis.dtype)
    kv = policy._build_kv(vis, s_enc)
    # use a fake target = MASK marker for all positions to maximize masking pressure
    target_indices = [torch.full((Bf, T_l), policy.k, dtype=torch.long, device=vis.device)
                      for T_l in seq_lens]
    mask_list = [torch.ones(Bf, T_l, dtype=torch.bool, device=vis.device) for T_l in seq_lens]
    y = policy._y_embed(Bf, vis.device, target_indices, mask_list)

    z_H = torch.zeros(Bf, N, DIM, device=vis.device, dtype=y.dtype)
    print(f"\n  ρ_L scalar mean: {rL.mean().item():.4f}  ρ_H scalar mean: {rH.mean().item():.4f}")
    print(f"  ρ_L per-dim range: [{rL.min().item():.4f}, {rL.max().item():.4f}]")
    print(f"  ρ_H per-dim range: [{rH.min().item():.4f}, {rH.max().item():.4f}]")
    print(f"  initial |y|: {y.abs().mean().item():.4f}")
    print(f"  initial |vis|: {vis.abs().mean().item():.4f}  |s_enc|: {s_enc.abs().mean().item():.4f}")

    for h_idx in range(H):
        # inner loop
        z_L = torch.zeros_like(y)
        for t_inner in range(L):
            g_in = z_L + z_H + y
            g_out = policy.g(g_in, kv)
            if policy.update_mode == 'damped':
                alpha_L = wL                                # (D,) per-dim
                z_L = (1 - alpha_L) * z_L + alpha_L * g_out
            else:
                z_L = z_L + wL[t_inner] * g_out
            print(f"  outer={h_idx} inner={t_inner}: |g_in|={g_in.abs().mean():.3e}  |g_out|={g_out.abs().mean():.3e}  |z_L|={z_L.abs().mean():.3e}")
        # outer
        g_in_H = z_H + z_L + y
        g_out_H = policy.g(g_in_H, kv)
        if policy.update_mode == 'damped':
            alpha_H = wH
            z_H = (1 - alpha_H) * z_H + alpha_H * g_out_H
        else:
            z_H = z_H + wH[h_idx] * g_out_H
        print(f"  outer={h_idx} (post): |g_out_H|={g_out_H.abs().mean():.3e}  |z_H|={z_H.abs().mean():.3e}")
    return vis, s_enc, kv, y, z_L


@torch.no_grad()
def lipschitz_g(policy, y_sample, kv, n_probes=8, delta_scale=0.01):
    """Empirically estimate g's Lipschitz constant via random small perturbations."""
    ratios = []
    g0 = policy.g(y_sample, kv)
    for _ in range(n_probes):
        delta = torch.randn_like(y_sample) * delta_scale
        g1 = policy.g(y_sample + delta, kv)
        r = (g1 - g0).norm().item() / delta.norm().item()
        ratios.append(r)
    return min(ratios), sum(ratios)/len(ratios), max(ratios)


def main():
    print(f"="*100)
    print(f"DIAGNOSTIC: BEFORE (step 7000, val 19.33%) vs AFTER (step 9000, diverged)")
    print(f"="*100)

    # Load VAE once
    vae_c = torch.load(f'data/ckpts/oxe_vqvae_{ROBOT}.pt', map_location='cpu', weights_only=False)
    adim = vae_c['action_dim']
    vae = ActionVQVAE1d(action_dim=adim, vq_cls=VQ1d_EMA, k=vae_c.get('k', 128)).cuda().eval()
    vae.load_state_dict(vae_c['vae'])
    var_global = vae_c['action_var_global'].view(1, 1, -1).cuda()
    vae_data = (vae_c, vae, var_global, adim)

    # build batch (use args from the first ckpt — both have same args)
    ck_before = torch.load(CKPT_BEFORE, map_location='cpu', weights_only=False)
    img_n, t5s, states, actions, prevs, tasks = build_batch(ck_before['args'], None, vae_data)

    # 1. PARAM NORMS COMPARISON
    print(f"\n{'─'*100}\nPARAM NORM ΔRATIO BEFORE vs AFTER (top 15 by ratio):")
    ck_after = torch.load(CKPT_AFTER, map_location='cpu', weights_only=False)
    rows = []
    for sec in ['cnn', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']:
        for name, t_after in ck_after[sec].items():
            t_before = ck_before[sec].get(name)
            if t_before is None: continue
            n_before = t_before.float().norm().item()
            n_after = t_after.float().norm().item()
            ratio = n_after / max(n_before, 1e-12)
            rows.append((f"{sec}.{name}", n_before, n_after, ratio))
    rows.sort(key=lambda r: -abs(r[3] - 1.0))
    print(f"  {'name':<55s} {'||W||_before':>12s} {'||W||_after':>12s} {'ratio':>8s}")
    for nm, b, a, r in rows[:20]:
        flag = ' ⚠ HUGE' if (r > 5 or r < 0.2) else ''
        print(f"  {nm:<55s} {b:>12.3e} {a:>12.3e} {r:>8.2f}{flag}")

    # 2. FORWARD PASS DIAGNOSTIC for BEFORE model
    print(f"\n{'─'*100}\nFORWARD PASS — BEFORE (step 7000, val 19.33%):")
    mods_b, keys, args_b = build_modules(ck_before)
    vis_b, s_enc_b, kv_b, y_b, z_L_b = forward_diag(mods_b, args_b, img_n, t5s, states, actions, prevs)

    # Lipschitz estimate
    print(f"\n  ── empirical Lipschitz of g (8 probes, δ=0.01) ──")
    lo, mn, hi = lipschitz_g(mods_b[-1], y_b, kv_b)
    print(f"  Lipschitz min={lo:.3f}  mean={mn:.3f}  max={hi:.3f}   (need <1 for inner damped to contract @ α=0.5)")

    # cleanup
    del mods_b; torch.cuda.empty_cache()

    # 3. FORWARD PASS DIAGNOSTIC for AFTER model
    print(f"\n{'─'*100}\nFORWARD PASS — AFTER (step 9000, post-divergence):")
    mods_a, keys, args_a = build_modules(ck_after)
    vis_a, s_enc_a, kv_a, y_a, z_L_a = forward_diag(mods_a, args_a, img_n, t5s, states, actions, prevs)
    print(f"\n  ── empirical Lipschitz of g (8 probes, δ=0.01) ──")
    lo, mn, hi = lipschitz_g(mods_a[-1], y_a, kv_a)
    print(f"  Lipschitz min={lo:.3f}  mean={mn:.3f}  max={hi:.3f}")

    print(f"\n{'='*100}\nINTERPRETATION:")
    print(f"  - If a param's norm grew >5× → it drifted to extreme values.")
    print(f"  - If forward activations explode (|z_L| or |g_out| growing across inner iters):")
    print(f"    Banach fixed-point iteration is NOT contracting — Lipschitz of g > 1 - 1/α = 1 (for α=0.5).")
    print(f"  - Lipschitz constant > 1 means g is expansive → inner loop amplifies any perturbation.")
    print(f"    Tikhonov input noise (g_input_noise) IS supposed to bound this. If Lipschitz drifted up,")
    print(f"    Tikhonov pressure was insufficient.")


if __name__ == '__main__':
    main()
