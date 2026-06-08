#!/usr/bin/env python3
"""Three deeper probes into the magnitude decay observed in g(t):

  (a) Per-dimension decay: is the decay uniform across all channels of g,
      or concentrated in some dims (suggesting channel-specific saturation)?
  (b) Raw ||g(t)|| vs weighted ||wL[t]*g(t)||: does the apparent decay
      vanish if we remove the geometric weighting? (i.e. is the model
      *compensating* for the weight schedule?)
  (c) Alternative weighting schemes that still give a convex combination
      (Cauchy-convergent): uniform, harmonic, exp-decay, geometric (current).
      Same model state, different w_t — does the ||g(t)|| pattern change?
"""
import os, sys, glob, math, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from babygroot_strm import (ActionVQVAE1d, VQ1d_EMA, STRMPolicyVAE,
                            LayerAggregator, ScaleNorm)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset, EMBODIMENTS, EMBODIMENT_ID
from babygroot_strm.cond_vae import CondActionVQVAE1d


def alt_weights(scheme: str, n: int, rho: float, dev, dtype):
    """Convex-normalized weight vector w_t for inner loop. All schemes sum to 1.

    For 'linear': `rho` is interpreted as the slope of a linear w[t] = 1 + slope*(t - (n-1)/2),
    centered around the middle to keep average ≈1, clamped to a small floor, then normalized.
    slope > 0 → increasing weights (later t weighted more).
    slope = 0 → uniform.
    slope < 0 → decreasing (earlier t weighted more, like geometric ρ < 1).
    """
    t = torch.arange(n, device=dev, dtype=dtype)
    if scheme == 'geometric':
        expo = t / max(n - 1, 1)
        w = torch.tensor(rho, device=dev, dtype=dtype) ** expo
    elif scheme == 'uniform':
        w = torch.ones(n, device=dev, dtype=dtype)
    elif scheme == 'harmonic':
        w = 1.0 / (t + 1)
    elif scheme == 'exp_decay':
        if rho >= 1.0: rho = 0.999
        if rho <= 0.0: rho = 1e-3
        tau = -1.0 / math.log(rho)
        w = torch.exp(-t / tau)
    elif scheme == 'cosine':
        w = 0.5 * (1 - torch.cos(2 * math.pi * (t + 1) / (n + 1)))
    elif scheme == 'linear':
        slope = rho                                          # use rho param as the slope (any real)
        w = 1.0 + slope * (t - (n - 1) / 2)
        w = w.clamp(min=1e-3)                                # keep strictly positive after clamp
    else:
        raise ValueError(scheme)
    return w / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='data/ckpts/oxe_policy_v5_clamp.pt')
    ap.add_argument('--oxe-root', default='data/oxe')
    ap.add_argument('--t5-cache', default='data/cache/t5_text_cache.pt')
    ap.add_argument('--image-var', default='data/cache/image_var_global.pt')
    ap.add_argument('--vae-dir', default='data/ckpts')
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0); random.seed(0)

    # ── boilerplate load (same as diag_bimodality.py) ──
    c = torch.load(args.ckpt, map_location=dev, weights_only=False); a = c['args']
    print(f"loaded {args.ckpt} step={c['step']}")
    sc = torch.load(os.path.join(args.vae_dir, 'oxe_shared_vae.pt'), map_location=dev, weights_only=False)
    shared_vae = CondActionVQVAE1d(action_dim=sc['action_dim'], n_embodiments=sc['n_embodiments'], k=sc['k']).to(dev)
    shared_vae.load_state_dict(sc['vae']); shared_vae.eval()
    var_globals = {emb: sc['action_var_globals'][EMBODIMENT_ID.get(emb, -1)].to(dev).view(1, 1, -1)
                   for emb in EMBODIMENTS if EMBODIMENT_ID.get(emb, -1) in sc['action_var_globals']}
    K = shared_vae.vqs[0].K; seq_lens = tuple(shared_vae.seq_lens)
    t5 = torch.load(args.t5_cache, map_location='cpu', weights_only=False)
    t5_emb, t5_dim, t5_layers = t5['embeddings'], t5['dim'], t5['n_layers']
    img_var = torch.load(args.image_var, map_location='cpu', weights_only=False)
    var_global_img = img_var['var_global'].to(dev)
    cnn = EfficientCNN(dims=tuple(a['cnn_dims']), depths=tuple(a['cnn_depths']),
                       expand=a['cnn_expand'], out_dim=a['cnn_out_dim'],
                       norm=a['cnn_norm'], pos_emb=a['cnn_pe'], img_size=a['img_size']).to(dev)
    text_agg = LayerAggregator(hidden_dim=t5_dim, n_layers=t5_layers).to(dev)
    cnn_proj = nn.Linear(a['cnn_out_dim'], a['dim']).to(dev)
    text_proj = nn.Linear(t5_dim, a['dim']).to(dev)
    kv_norm = ScaleNorm(a['dim']).to(dev)
    n_vis = (a['img_size'] // 32) ** 2
    state_encoders = nn.ModuleDict({
        emb: nn.Sequential(nn.Linear(8, a['dim']), nn.GELU(), nn.Linear(a['dim'], a['dim']))
        for emb in sorted({k.split('.')[0] for k in c['state_encoders'].keys()})}).to(dev)
    emb_id_emb = nn.Embedding(len(EMBODIMENTS) + 1, a['dim']).to(dev)
    policy = STRMPolicyVAE(seq_lens=seq_lens, k_codebook=K, dim=a['dim'], heads=8,
                           depth=a['depth'], L_inner=a['L_inner'], H_outer=a['H_outer'],
                           state_dim=a['dim'], max_prefix=n_vis + a['max_text'] + 16 + 1,
                           beta=a['beta'], free_bits=a['free_bits']).to(dev)
    for nm, m in [('cnn', cnn), ('text_agg', text_agg), ('cnn_proj', cnn_proj),
                  ('text_proj', text_proj), ('kv_norm', kv_norm),
                  ('state_encoders', state_encoders), ('emb_id_emb', emb_id_emb),
                  ('policy', policy)]:
        m.load_state_dict(c[nm]); m.eval()

    # one batch
    specs = []
    for ds_dir in sorted(glob.glob(os.path.join(args.oxe_root, '*'))):
        if not os.path.isfile(os.path.join(ds_dir, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(ds_dir, chunk_len=16, lookback=16)
            if sp.chunk_index: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    idxs = [random.randint(0, len(ds) - 1) for _ in range(args.batch_size)]
    batch = [ds[i] for i in idxs]
    frames = torch.stack([
        (torch.from_numpy(np.asarray(b[0].convert('RGB').resize((a['img_size'], a['img_size']))))
         .permute(2, 0, 1).float() / 255.) for b in batch]).to(dev)
    states  = torch.stack([b[1] for b in batch]).to(dev)
    actions = torch.stack([b[2] for b in batch]).to(dev)
    prevs   = torch.stack([b[3] for b in batch]).to(dev)
    tasks   = [b[4] for b in batch]
    emb_robots = [EMBODIMENTS[b[5]] if b[5] < len(EMBODIMENTS) else 'unknown' for b in batch]

    with torch.no_grad():
        all_codes = [torch.zeros(actions.shape[0], T_l, dtype=torch.long, device=dev) for T_l in seq_lens]
        for emb in set(emb_robots):
            mask = torch.tensor([r == emb for r in emb_robots], device=dev)
            if mask.sum() == 0: continue
            ac = actions[mask]; pv = prevs[mask]
            vg = var_globals[emb]; nT = ac.shape[1]
            m = pv.mean(dim=1, keepdim=True); S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
            lam = nT / (S + nT * vg)
            xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
            eid = torch.full((ac.shape[0],), EMBODIMENT_ID.get(emb, len(EMBODIMENTS)), dtype=torch.long, device=dev)
            gt_c, _ = shared_vae.encode_with_soft(xn, eid, tau=0.1)
            for l in range(len(seq_lens)): all_codes[l][mask] = gt_c[l]
        x = normalize_image(frames, var_global_img)
        vtok, _ = cnn(x); vtok = cnn_proj(vtok)
        B, T = vtok.shape[0], a['max_text']
        out = torch.zeros(t5_layers, B, T, t5_dim)
        for b, tk in enumerate(tasks):
            e = t5_emb.get(tk)
            if e is None: continue
            h = e['hidden'].float(); t = min(h.shape[1], T)
            out[:, b, :t, :] = h[:, :t, :]
        t5s = out.to(dev)
        tagg = text_agg([t5s[l] for l in range(t5s.shape[0])])
        ttok = text_proj(tagg)
        idx = torch.tensor([EMBODIMENT_ID.get(r, len(EMBODIMENTS)) for r in emb_robots], device=dev)
        etok = emb_id_emb(idx).unsqueeze(1)
        vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
        s_enc = torch.zeros(states.shape[0], a['dim'], device=dev)
        for emb in set(emb_robots):
            mask = torch.tensor([r == emb for r in emb_robots], device=dev)
            if mask.sum() == 0: continue
            s_enc[mask] = state_encoders[emb](states[mask])

    gt = all_codes
    B = vis.shape[0]
    masks_full = [torch.ones(B, T_l, dtype=torch.bool, device=dev) for T_l in seq_lens]

    # ── (a) Per-dimension decay at ρ_L=0.5 ──
    print("\n=== (a) PER-DIMENSION ||g(t)|| at ρ_L=0.5 (geometric weighting) ===")
    print("  showing distribution: min / p25 / median / p75 / max of per-channel norm,")
    print("  and how the TOP-10 highest-magnitude channels at t=0 evolve.")
    with torch.no_grad():
        kv_full = policy._build_kv(vis, s_enc)
        y_full = policy._y_embed(B, dev, gt, masks_full)
        wL = alt_weights('geometric', a['L_inner'], 0.5, dev, y_full.dtype)
        z_L = torch.zeros_like(y_full); z_H = torch.zeros_like(y_full)
        per_dim_norms = []          # list of (D,) per-channel ||g(t)|| per step
        for t in range(a['L_inner']):
            g_t = policy.g(z_L + z_H + y_full, kv_full)
            # ||·|| per channel, averaged across batch + position
            ch_norm = g_t.pow(2).mean(dim=(0, 1)).sqrt()        # (D,)
            per_dim_norms.append(ch_norm)
            z_L = z_L + wL[t] * g_t
        per_dim_norms = torch.stack(per_dim_norms)             # (L, D)
        print(f"  {'t':>3s} | {'min':>7s} {'p25':>7s} {'med':>7s} {'p75':>7s} {'max':>7s} | top-channel mean (top 10 by t=0)")
        top10 = per_dim_norms[0].topk(10).indices
        for t in range(a['L_inner']):
            v = per_dim_norms[t]
            q = torch.quantile(v, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=dev))
            top10_mean = v[top10].mean().item()
            print(f"  {t:>3d} | {q[0]:>7.3f} {q[1]:>7.3f} {q[2]:>7.3f} {q[3]:>7.3f} {q[4]:>7.3f} | top10 ⟨||·||⟩={top10_mean:>7.3f}")
        # Per-channel decay ratio between t=0 and t=4
        ratio = per_dim_norms[-1] / per_dim_norms[0].clamp(min=1e-6)
        print(f"  per-channel decay (||g(t=4)||/||g(t=0)||): min={ratio.min().item():.3f}  "
              f"max={ratio.max().item():.3f}  median={ratio.median().item():.3f}")
        # Number of channels that GREW vs SHRANK
        n_grew = (ratio > 1.05).sum().item()
        n_shrunk = (ratio < 0.95).sum().item()
        print(f"  channels with >5% growth: {n_grew} | >5% shrinkage: {n_shrunk} | total D={ratio.numel()}")

    # ── (b) Raw ||g(t)|| under DIFFERENT weighting schemes ──
    print("\n=== (b) ||g(t)|| under DIFFERENT weighting schemes (same model state) ===")
    print("  if decay is intrinsic to the model, ||g(t)|| should look similar across schemes")
    print("  if decay is a weighting artifact, removing geometric weighting should change the pattern")
    print(f"  {'scheme':<12s} | {'ρ':>5s} | ||g(t=0,1,2,3,4)||  | rel ||g(t=4)||/||g(t=0)||")
    with torch.no_grad():
        for scheme in ['geometric', 'uniform', 'harmonic', 'exp_decay', 'cosine']:
            for rho in [0.1, 0.5, 0.9]:
                kv_full = policy._build_kv(vis, s_enc)
                y_full = policy._y_embed(B, dev, gt, masks_full)
                wL = alt_weights(scheme, a['L_inner'], rho, dev, y_full.dtype)
                z_L = torch.zeros_like(y_full); z_H = torch.zeros_like(y_full)
                g_norms = []
                for t in range(a['L_inner']):
                    g_t = policy.g(z_L + z_H + y_full, kv_full)
                    g_norms.append(g_t.norm().item())
                    z_L = z_L + wL[t] * g_t
                ratio = g_norms[-1] / g_norms[0] if g_norms[0] > 0 else float('nan')
                gs = "  ".join(f'{g:>7.1f}' for g in g_norms)
                print(f"  {scheme:<12s} | {rho:>5.2f} | {gs}  | {ratio:>5.3f}")

    # ── (c) Acc under different weighting schemes ──
    print("\n=== (c) all-mask top-1 ACC under different weighting schemes ===")
    print(f"  {'scheme':<12s} | {'ρ':>5s} | acc at t=0..4")
    with torch.no_grad():
        for scheme in ['geometric', 'uniform', 'harmonic', 'exp_decay', 'cosine']:
            for rho in [0.1, 0.5, 0.9]:
                kv_full = policy._build_kv(vis, s_enc)
                y_full = policy._y_embed(B, dev, gt, masks_full)
                wL = alt_weights(scheme, a['L_inner'], rho, dev, y_full.dtype)
                z_L = torch.zeros_like(y_full); z_H = torch.zeros_like(y_full)
                accs = []
                for t in range(a['L_inner']):
                    g_t = policy.g(z_L + z_H + y_full, kv_full)
                    z_L = z_L + wL[t] * g_t
                    logits, _, _ = policy._sample_heads(z_L)
                    cur = [logits[l][..., :K].argmax(-1) for l in range(len(seq_lens))]
                    correct = sum((cur[l] == gt[l]).sum().item() for l in range(len(seq_lens)))
                    total = sum(gt[l].numel() for l in range(len(seq_lens)))
                    accs.append(100 * correct / total)
                acc_str = " ".join(f't{i}:{a_:>4.1f}%' for i, a_ in enumerate(accs))
                print(f"  {scheme:<12s} | {rho:>5.2f} | {acc_str}")


if __name__ == '__main__':
    main()
