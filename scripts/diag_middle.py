#!/usr/bin/env python3
"""Why does the MIDDLE of (ρ_L, ρ_H) space fail?

Hypotheses to discriminate:
  M1. Weighting-form artifact: geometric specifically creates problematic z_L distributions
      in the middle. Test: same grid but with LINEAR weighting (the user's proposed
      reparameterization). If bimodal disappears → form-specific.
  M2. The head was only trained on z_L distributions from low-ρ regime, so anywhere it
      hasn't seen z_L statistics is OOD. Both endpoints are coincidentally well-handled.
  M3. Head SATURATION: at certain ρ, the head's pre-softmax logits flatten or explode,
      producing high-entropy garbage predictions.
  M4. Interaction between ρ_L and ρ_H: the cross-product is what matters, not each alone.

Direct probes:
  * z_L magnitude vs ρ — is it monotonic?
  * head logit entropy vs ρ — does the head get confused at middle ρ?
  * top-1 prediction CHANGE rate from one ρ to neighboring ρ — sharp transitions?
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
from scripts.diag_weighting import alt_weights


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

    c = torch.load(args.ckpt, map_location=dev, weights_only=False); a = c['args']
    print(f"loaded {args.ckpt} step={c['step']}")
    print(f"trained ρ_L={c['policy']['rho_L_raw'].item():.4f}  ρ_H={c['policy']['rho_H_raw'].item():.4f}")
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
            ac = actions[mask]; pv = prevs[mask]; vg = var_globals[emb]; nT = ac.shape[1]
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
            h = e['hidden'].float(); t = min(h.shape[1], T); out[:, b, :t, :] = h[:, :t, :]
        t5s = out.to(dev); tagg = text_agg([t5s[l] for l in range(t5s.shape[0])])
        ttok = text_proj(tagg)
        idx = torch.tensor([EMBODIMENT_ID.get(r, len(EMBODIMENTS)) for r in emb_robots], device=dev)
        etok = emb_id_emb(idx).unsqueeze(1)
        vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
        s_enc = torch.zeros(states.shape[0], a['dim'], device=dev)
        for emb in set(emb_robots):
            mask = torch.tensor([r == emb for r in emb_robots], device=dev)
            if mask.sum() == 0: continue
            s_enc[mask] = state_encoders[emb](states[mask])

    gt = all_codes; B = vis.shape[0]
    masks_full = [torch.ones(B, T_l, dtype=torch.bool, device=dev) for T_l in seq_lens]
    kv_full = policy._build_kv(vis, s_enc)
    y_full = policy._y_embed(B, dev, gt, masks_full)

    def run_inner(rL_value, scheme='geometric', n=a['L_inner']):
        """Run one inner loop with given ρ_L (or slope for linear). Returns final z_L + diagnostics."""
        wL = alt_weights(scheme, n, rL_value, dev, y_full.dtype)
        z_L = torch.zeros_like(y_full); z_H = torch.zeros_like(y_full)
        for t in range(n):
            g_t = policy.g(z_L + z_H + y_full, kv_full)
            z_L = z_L + wL[t] * g_t
        logits, mu, logprec = policy._sample_heads(z_L)
        # diagnostics
        z_norm = z_L.norm().item()
        # logit entropy (averaged across positions)
        ent = 0.0
        for l in range(len(seq_lens)):
            lp = F.log_softmax(logits[l][..., :K], dim=-1)
            p = lp.exp()
            ent += -(p * lp).sum(-1).mean().item()
        ent /= len(seq_lens)
        # acc + loss
        correct = 0; total = 0; loss = 0
        for l in range(len(seq_lens)):
            lp = F.log_softmax(logits[l][..., :K], dim=-1)
            ce = -lp.gather(-1, gt[l].unsqueeze(-1)).squeeze(-1).mean()
            loss += ce.item()
            cur = logits[l][..., :K].argmax(-1)
            correct += (cur == gt[l]).sum().item()
            total += gt[l].numel()
        return {'z_norm': z_norm, 'entropy': ent, 'acc': 100*correct/total, 'loss': loss/len(seq_lens),
                'mu_norm': mu.norm().item(), 'logprec_mean': logprec.mean().item()}

    # ═══ (1) Sweep ρ_L for GEOMETRIC weighting — refine the bimodality picture ═══
    print("\n=== (1) GEOMETRIC weighting: ρ_L sweep at finer resolution ===")
    print(f"  {'ρ':>6s} | {'loss':>6s} {'acc%':>5s} | {'||z||':>8s} {'entropy':>8s} {'||mu||':>8s} {'⟨logprec⟩':>10s}")
    with torch.no_grad():
        for rL in [0.0001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
            r = run_inner(rL, 'geometric')
            print(f"  {rL:>6.4f} | {r['loss']:>6.3f} {r['acc']:>5.1f} | {r['z_norm']:>8.1f} "
                  f"{r['entropy']:>8.3f} {r['mu_norm']:>8.1f} {r['logprec_mean']:>10.3f}")

    # ═══ (2) Same sweep with LINEAR weighting (user's proposal) ═══
    print("\n=== (2) LINEAR weighting: slope sweep (centered, slope=0 → uniform) ===")
    print(f"  {'slope':>6s} | {'loss':>6s} {'acc%':>5s} | {'||z||':>8s} {'entropy':>8s} {'weights':>40s}")
    with torch.no_grad():
        for slope in [-2.0, -1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0, 2.0]:
            r = run_inner(slope, 'linear')
            ws = alt_weights('linear', a['L_inner'], slope, dev, y_full.dtype)
            ws_str = "[" + ",".join(f"{w.item():.2f}" for w in ws) + "]"
            print(f"  {slope:>6.2f} | {r['loss']:>6.3f} {r['acc']:>5.1f} | {r['z_norm']:>8.1f} "
                  f"{r['entropy']:>8.3f} {ws_str:>40s}")
    print("  (slope=0 → uniform; slope<0 → early-step weighted; slope>0 → late-step weighted)")

    # ═══ (3) z_L statistics: is the middle OOD in some measurable way? ═══
    print("\n=== (3) z_L distribution statistics across ρ_L (geometric) ===")
    print("  measures: variance and 'sparsity' (% near-zero channels) of z_L")
    print(f"  {'ρ_L':>6s} | {'var(z_L)':>10s} {'%near-zero':>11s} {'kurtosis':>10s}")
    with torch.no_grad():
        for rL in [0.0001, 0.05, 0.2, 0.5, 0.7, 0.9]:
            wL = alt_weights('geometric', a['L_inner'], rL, dev, y_full.dtype)
            z_L = torch.zeros_like(y_full); z_H = torch.zeros_like(y_full)
            for t in range(a['L_inner']):
                g_t = policy.g(z_L + z_H + y_full, kv_full)
                z_L = z_L + wL[t] * g_t
            flat = z_L.flatten()
            var = flat.var().item()
            near_zero = (flat.abs() < 0.01).float().mean().item() * 100
            kurt = (((flat - flat.mean()) / flat.std()) ** 4).mean().item() - 3
            print(f"  {rL:>6.4f} | {var:>10.3f} {near_zero:>10.1f}% {kurt:>10.3f}")


if __name__ == '__main__':
    main()
