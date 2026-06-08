#!/usr/bin/env python3
"""Empirical test of g's dynamics under WEIGHTLESS iteration.

Cauchy/decaying weights guarantee z_L stays bounded REGARDLESS of g's properties — that's
the design rationale. But it ALSO hides whether g itself is naturally well-behaved.

To find out, replace the convex weighting with constant w=1 (no Cauchy guarantee) and
observe ||z_L||, ||g(t)||, and the direction of g(t) under several update rules:

  (A) Pure Cauchy (current): z_L^{t+1} = z_L^t + w[t]·g(z_L^t + y, kv), Σw=1
  (B) Constant w=1:           z_L^{t+1} = z_L^t + g(z_L^t + y, kv)            — unbounded?
  (C) Direct overwrite:       z_L^{t+1} = g(z_L^t + y, kv)                     — iteration of g(·+y, kv)
  (D) Damped fixed point:     z_L^{t+1} = z_L^t + α·(g(z_L^t + y, kv) - z_L^t) — gradient toward fp
  (E) Implicit/Newton-style:  not implemented; would require jacobian

If (B) diverges → g is non-contractive in this direction, Cauchy is essential.
If (B) converges → g is naturally contractive, Cauchy adds nothing.
If (C) settles to a fixed point → g really has a fixed point (Banach).
If (C) cycles or diverges → no fixed point near the origin.
"""
import os, sys, glob, math, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from babygroot_strm import ActionVQVAE1d, VQ1d_EMA, STRMPolicyVAE, LayerAggregator, ScaleNorm
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset, EMBODIMENTS, EMBODIMENT_ID
from babygroot_strm.cond_vae import CondActionVQVAE1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='data/ckpts/oxe_policy_v5_clamp.pt')
    ap.add_argument('--oxe-root', default='data/oxe')
    ap.add_argument('--t5-cache', default='data/cache/t5_text_cache.pt')
    ap.add_argument('--image-var', default='data/cache/image_var_global.pt')
    ap.add_argument('--vae-dir', default='data/ckpts')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--n-iters', type=int, default=20, help="run more iterations than L_inner to see asymptotic behavior")
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
    kv_full = policy._build_kv(vis, s_enc)
    y_full = policy._y_embed(B, dev, gt, masks_full)
    z_H = torch.zeros_like(y_full)

    def trajectory(update_fn, n=args.n_iters, label='', verbose_steps=(0,1,2,3,4,9,14,19)):
        """Run iteration; record ||z||, ||g||, cos(g_t,g_{t-1}), ||z_t - z_{t-1}||/||z_{t-1}||."""
        z_L = torch.zeros_like(y_full); g_prev = None; z_prev = None
        history = []
        for t in range(n):
            g_t = policy.g(z_L + z_H + y_full, kv_full)
            z_new = update_fn(z_L, g_t, t)
            rec = {
                't': t,
                'z_norm': z_new.norm().item(),
                'g_norm': g_t.norm().item(),
                'cos_gg': (F.cosine_similarity(g_t.flatten(1), g_prev.flatten(1), dim=1).mean().item()
                           if g_prev is not None else float('nan')),
                'rel_z':  ((z_new - z_prev).norm() / max(z_prev.norm().item(), 1e-6)).item()
                           if z_prev is not None else float('nan'),
            }
            history.append(rec)
            z_prev = z_new.clone(); g_prev = g_t.clone(); z_L = z_new
        print(f"\n  {label}")
        print(f"    t  | {'||z||':>9s} {'||g||':>9s} {'cos(g_t,g_t-1)':>15s} {'||Δz||/||z||':>14s}")
        for r in history:
            if r['t'] in verbose_steps:
                print(f"    {r['t']:>2d} | {r['z_norm']:>9.2f} {r['g_norm']:>9.2f} "
                      f"{r['cos_gg']:>15.4f} {r['rel_z']:>14.5f}")
        # summary: is it diverging, converging, or stable?
        z_norms = [r['z_norm'] for r in history]
        g_norms = [r['g_norm'] for r in history]
        if z_norms[-1] > 5 * z_norms[len(z_norms)//2]:
            verdict = "DIVERGING"
        elif z_norms[-1] < 0.2 * z_norms[len(z_norms)//2]:
            verdict = "COLLAPSING"
        elif abs(z_norms[-1] - z_norms[-5]) / max(z_norms[-5], 1e-6) < 0.02:
            verdict = "CONVERGED (||z|| stable in last 5 iters)"
        else:
            verdict = "STILL CHANGING"
        print(f"    → verdict: {verdict}")

    # Pre-compute Cauchy weights for the reference scheme
    rho_ref = 0.5
    wL_cauchy = policy._weights(torch.tensor(rho_ref, device=dev), args.n_iters, dev, y_full.dtype)

    with torch.no_grad():
        # (A) Cauchy reference
        trajectory(lambda z, g, t: z + wL_cauchy[t] * g,
                   label=f"(A) CAUCHY weighting (ρ=0.5, n={args.n_iters}) — bounded by design")

        # (B) Constant w=1 (no normalization) — this is what Cauchy guards against
        trajectory(lambda z, g, t: z + g,
                   label="(B) CONSTANT w=1 (no Cauchy) — does ||z|| blow up?")

        # (C) Direct overwrite: z = g(z + y, kv); pure fixed-point iteration
        trajectory(lambda z, g, t: g,
                   label="(C) DIRECT OVERWRITE z = g(z + y, kv) — true fixed-point iter")

        # (D) Damped: z = z + α(g - z) with α=0.5
        trajectory(lambda z, g, t: z + 0.5 * (g - z),
                   label="(D) DAMPED z = z + 0.5·(g - z)")

        # (E) Heavy momentum: z = 0.9 z + 0.1 g (low pass)
        trajectory(lambda z, g, t: 0.9 * z + 0.1 * g,
                   label="(E) MOMENTUM z = 0.9 z + 0.1 g")

    print("\nInterpretation:")
    print("  If (B) DIVERGES → g is not contractive on its own. Cauchy is doing real work.")
    print("  If (B) CONVERGES → g is accidentally contractive. Cauchy redundant for this g.")
    print("  Compare (A) vs (C): does Cauchy give the same fixed point that direct iteration finds?")


if __name__ == '__main__':
    main()
