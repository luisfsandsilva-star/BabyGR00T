#!/usr/bin/env python3
"""Bimodality diagnoses: WHY does the loss surface have two basins (ρ≈0 and ρ≈0.9)
with a barrier at ρ≈0.5, when literature on iterative refinement says depth should help monotonically?

Hypotheses to test:
  H1. g() is trained as a one-shot predictor → adding more iterations corrupts it
  H2. g() outputs are nearly identical across iterations (converged → refinement futile)
  H3. The closed-form geometric weights create a "convex-combination" inversion — at ρ→0 only t=0
      contributes; at ρ→1 uniform avg. Middle ρ requires g() to be both one-shot AND refinement.
  H4. Gradient flow through recursion is dominated by t=0 (effectively, model only trains t=0)
  H5. The output head expects a specific z_L magnitude that only matches at certain ρ
"""
import os, sys, glob, math, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from babygroot_strm import (RevIN, ActionVQVAE1d, VQ1d_EMA, STRMPolicy, STRMPolicyVAE,
                            LayerAggregator, ScaleNorm)
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
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()
    dev = 'cuda'
    torch.manual_seed(0); random.seed(0)

    # ── load ckpt + caches (same as diag_rho.py) ──
    c = torch.load(args.ckpt, map_location=dev, weights_only=False); a = c['args']
    print(f"loaded {args.ckpt} step={c['step']}")
    sc = torch.load(os.path.join(args.vae_dir, 'oxe_shared_vae.pt'), map_location=dev, weights_only=False)
    shared_vae = CondActionVQVAE1d(action_dim=sc['action_dim'], n_embodiments=sc['n_embodiments'], k=sc['k']).to(dev)
    shared_vae.load_state_dict(sc['vae']); shared_vae.eval()
    for p in shared_vae.parameters(): p.requires_grad_(False)
    var_globals = {emb: sc['action_var_globals'][EMBODIMENT_ID.get(emb, -1)].to(dev).view(1, 1, -1)
                   for emb in EMBODIMENTS if EMBODIMENT_ID.get(emb, -1) in sc['action_var_globals']}
    K = shared_vae.vqs[0].K
    seq_lens = tuple(shared_vae.seq_lens)
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
    print(f"policy current ρ_L_raw={policy.rho_L_raw.item():.4f} (used as ρ_L), "
          f"ρ_H_raw={policy.rho_H_raw.item():.4f}")

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
    emb_ids = [b[5] for b in batch]
    emb_robots = [EMBODIMENTS[e] if e < len(EMBODIMENTS) else 'unknown' for e in emb_ids]

    # encode targets + modalities (no grad needed for diags)
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
            gt_codes, _ = shared_vae.encode_with_soft(xn, eid, tau=0.1)
            for l in range(len(seq_lens)): all_codes[l][mask] = gt_codes[l]
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
        # state
        s_enc = torch.zeros(states.shape[0], a['dim'], device=dev)
        for emb in set(emb_robots):
            mask = torch.tensor([r == emb for r in emb_robots], device=dev)
            if mask.sum() == 0: continue
            s_enc[mask] = state_encoders[emb](states[mask])

    gt = all_codes
    B = vis.shape[0]

    def make_masks(ratio):
        """All-masked if ratio=1.0; random mask of this approximate ratio otherwise."""
        out = []
        for T_l in seq_lens:
            m = torch.rand(B, T_l, device=dev) < ratio
            # always mask at least one position
            m[torch.arange(B, device=dev), torch.randint(0, T_l, (B,), device=dev)] = True
            out.append(m)
        return out

    masks_full = make_masks(1.0)

    # ════ H2: g(t) similarity — direction (cosine) AND magnitude (L2 distance) ════
    print("\n=== H2: similarity of g(t) vs g(t-1) — cosine AND L2 ===")
    print(f"  {'ρ_L':>5s} | {'cos[mean]':>10s} | {'L2(g_t-g_t-1)/||g_t-1||':>26s} | {'||g(0)||':>10s} {'||g(4)||':>10s}")
    with torch.no_grad():
        for rL in [0.001, 0.1, 0.3, 0.5, 0.7, 0.9]:
            kv_full = policy._build_kv(vis, s_enc)
            y_full = policy._y_embed(B, dev, gt, masks_full)            # all-MASKED y
            wL = policy._weights(torch.tensor(rL, device=dev), a['L_inner'], dev, y_full.dtype)
            z_L = torch.zeros_like(y_full); z_H = torch.zeros_like(y_full)
            g_prev = None; cosines = []; rel_l2 = []; g_norms = []
            for t in range(a['L_inner']):
                g_t = policy.g(z_L + z_H + y_full, kv_full)
                g_norms.append(g_t.norm().item())
                if g_prev is not None:
                    cos = F.cosine_similarity(g_t.flatten(1), g_prev.flatten(1), dim=1).mean().item()
                    cosines.append(cos)
                    # relative L2: ||g_t - g_t-1|| / ||g_t-1|| (0 → identical, large → very different)
                    rel = ((g_t - g_prev).norm(dim=(1, 2)) / g_prev.norm(dim=(1, 2)).clamp(min=1e-6)).mean().item()
                    rel_l2.append(rel)
                z_L = z_L + wL[t] * g_t
                g_prev = g_t.clone()
            mean_c = sum(cosines) / max(len(cosines), 1)
            mean_l = sum(rel_l2) / max(len(rel_l2), 1)
            print(f"  {rL:>5.3f} | {mean_c:>10.4f} | {mean_l:>26.4f} | {g_norms[0]:>10.2f} {g_norms[-1]:>10.2f}")
    print("  → cos~1 + L2 small ⇒ truly converged.  cos~1 + L2 large ⇒ same direction but growing/shrinking.")

    # STRMPolicyVAE splits z = [μ | ρ] and head reads only μ (dim/2 channels).
    # use _sample_heads which handles that correctly.
    def _vae_heads(z):
        logits, _, _ = policy._sample_heads(z)
        return logits

    # ════ H4: gradient ∂loss/∂g(t) at each inner step (rho_L=0.9 forced) ════
    print("\n=== H4: gradient ∂loss/∂g(t) magnitude at each inner step (ρ_L=0.9 forced) ===")
    print("  (does t=0 get all the gradient signal, starving t≥1?)")
    original_L = policy.rho_L_raw.data.clone()
    policy.rho_L_raw.data.fill_(0.9)
    kv_full = policy._build_kv(vis, s_enc)
    y_full = policy._y_embed(B, dev, gt, masks_full)
    wL = policy._weights(policy.rho_L_raw.clamp(1e-3, 1 - 1e-3), a['L_inner'], dev, y_full.dtype)
    z_L = torch.zeros_like(y_full); z_H = torch.zeros_like(y_full)
    g_outs = []
    for t in range(a['L_inner']):
        g_t = policy.g(z_L + z_H + y_full, kv_full)
        g_t.retain_grad()
        g_outs.append(g_t)
        z_L = z_L + wL[t] * g_t
    logits = _vae_heads(z_L)
    total_loss = 0
    for l in range(len(seq_lens)):
        lp = F.log_softmax(logits[l], dim=-1)
        ce = -lp.gather(-1, gt[l].unsqueeze(-1)).squeeze(-1)
        total_loss = total_loss + ce.mean()
    total_loss = total_loss / len(seq_lens)
    total_loss.backward()
    print(f"  loss = {total_loss.item():.4f}")
    print(f"  {'t':>3s} | {'weight':>8s} | {'||g(t)||':>10s} | {'||grad g(t)||':>14s} | {'effective signal w*||grad||':>28s}")
    for t, g_t in enumerate(g_outs):
        gn = g_t.norm().item()
        gradn = g_t.grad.norm().item() if g_t.grad is not None else 0.0
        print(f"  {t:>3d} | {wL[t].item():>8.4f} | {gn:>10.3f} | {gradn:>14.6f} | {wL[t].item()*gradn:>28.6f}")
    policy.rho_L_raw.data.copy_(original_L)

    # ════ H1: per-iteration acc swept over (mask_ratio × ρ_L × MAP/sample) ════
    print("\n=== H1: per-iteration ACC across mask ratios, ρ_L, and MAP vs sampling ===")
    print("  (mask 1.0 = all-masked inference; lower = partial GT visible at training time)")
    print("  (sample: use random eps each call; MAP: eps=0; only sample at FINAL step is the standard)")
    for mask_ratio in [0.25, 0.5, 0.75, 1.0]:
        masks_r = make_masks(mask_ratio)
        print(f"\n  ──── mask_ratio = {mask_ratio} ────")
        print(f"  {'ρ_L':>5s} {'mode':>10s} | acc per inner step (t=0..4)")
        for rL in [0.001, 0.1, 0.5, 0.9]:
            for use_sample in [False, True]:
                # use_sample True: sample at final t. False: MAP (eps=0)
                policy.eval()                                       # base mode = eval (eps=0)
                with torch.no_grad():
                    kv_full = policy._build_kv(vis, s_enc)
                    y_full = policy._y_embed(B, dev, gt, masks_r)
                    wL = policy._weights(torch.tensor(rL, device=dev), a['L_inner'], dev, y_full.dtype)
                    z_L = torch.zeros_like(y_full); z_H = torch.zeros_like(y_full)
                    accs = []
                    for t in range(a['L_inner']):
                        g_t = policy.g(z_L + z_H + y_full, kv_full)
                        z_L = z_L + wL[t] * g_t
                        # For sampling: at every step, but eps only random at the FINAL t.
                        # Simplest: temporarily set training=True only at the last step (so eps≠0 then).
                        if use_sample and t == a['L_inner'] - 1:
                            policy.train()
                            logits = _vae_heads(z_L)
                            policy.eval()
                        else:
                            logits = _vae_heads(z_L)
                        cur_codes = [logits[l][..., :K].argmax(-1) for l in range(len(seq_lens))]
                        correct = sum((cur_codes[l] == gt[l]).sum().item() for l in range(len(seq_lens)))
                        total = sum(gt[l].numel() for l in range(len(seq_lens)))
                        accs.append(100 * correct / total)
                acc_str = " ".join(f't{i}:{a_:>4.1f}%' for i, a_ in enumerate(accs))
                mode = 'sample@t=4' if use_sample else 'MAP'
                print(f"  {rL:>5.2f} {mode:>10s} | {acc_str}")
    print("\n  → look for: monotonic ↑ within a row (refinement helps), high ρ better at high mask (recursion needed at inference)")


if __name__ == '__main__':
    main()
