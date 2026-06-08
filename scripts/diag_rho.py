#!/usr/bin/env python3
"""ρ-collapse diagnostic.

Loads a (possibly mid-training) policy ckpt and a real batch from the OXE
loader, then probes:

  1. Gradient ∂L/∂rho_L_raw and ∂L/∂rho_H_raw — magnitude + sign + how it
     compares to gradients on other params.
  2. Loss as a function of forced ρ (ρ ∈ {0.0, 0.05, 0.1, 0.3, 0.5, 0.9}):
     does the optimizer's preferred ρ → 0 actually correspond to the
     minimum loss, or is it a sub-optimum it can't escape?
  3. Per-step inner-loop refinement magnitudes ||g(t)|| for t=0..L-1.
     If t=0 dwarfs t≥1, recursion adds noise and is correctly killed.
"""
import os, sys, glob, json, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import math, numpy as np, torch, torch.nn as nn
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

    # ── load ckpt + caches ──
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

    # ── rebuild policy modules ──
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
        m.load_state_dict(c[nm]); m.train()
    print(f"policy current ρ_L={torch.sigmoid(policy.rho_L_raw).item():.4f}  "
          f"ρ_H={torch.sigmoid(policy.rho_H_raw).item():.4f}")

    # ── grab one real batch from OXE ──
    specs = []
    for ds_dir in sorted(glob.glob(os.path.join(args.oxe_root, '*'))):
        if not os.path.isfile(os.path.join(ds_dir, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(ds_dir, chunk_len=16, lookback=16)
            if sp.chunk_index: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    random.seed(0)
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

    # encode targets
    def encode_targets():
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
            gt, _ = shared_vae.encode_with_soft(xn, eid, tau=0.1)
            for l in range(len(seq_lens)):
                all_codes[l][mask] = gt[l]
        return all_codes

    def encode_modalities():
        x = normalize_image(frames, var_global_img)
        vtok, _ = cnn(x); vtok = cnn_proj(vtok)
        B = vtok.shape[0]; T = a['max_text']
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
        return kv_norm(torch.cat([etok, vtok, ttok], dim=1))

    def encode_state():
        out = torch.zeros(states.shape[0], a['dim'], device=dev)
        for emb in set(emb_robots):
            mask = torch.tensor([r == emb for r in emb_robots], device=dev)
            if mask.sum() == 0: continue
            out[mask] = state_encoders[emb](states[mask])
        return out

    gt = encode_targets()
    vis = encode_modalities()
    s_enc = encode_state()

    # ════ DIAG 1: gradient on ρ — and the effective Δρ per step ════
    print("\n=== DIAG 1: gradient on ρ_L_raw / ρ_H_raw + EFFECTIVE Δρ per optimizer step ===")
    print(f"  (lr={a['lr']}, opt = MuSGD_LARS; check if sigmoid' makes Δρ vanishingly small)")
    for trial in range(3):
        # MUST recompute encodings per trial; backward consumes the graph
        gt = encode_targets()
        vis = encode_modalities()
        s_enc = encode_state()
        for p in policy.parameters(): p.requires_grad_(True)
        policy.zero_grad()
        loss, per, _ = policy.forward_loss(gt, vis, s_enc, n_inner=a['L_inner'],
                                            h_max=a['h_max'], mask_ratio_max=1.0)
        loss.backward()
        rL = torch.sigmoid(policy.rho_L_raw).item()
        rH = torch.sigmoid(policy.rho_H_raw).item()
        gL = policy.rho_L_raw.grad.item() if policy.rho_L_raw.grad is not None else 0.0
        gH = policy.rho_H_raw.grad.item() if policy.rho_H_raw.grad is not None else 0.0
        # gradient WRT ρ value via chain rule: dL/dρ = dL/draw · 1/(ρ·(1-ρ))
        dL_dρ_L = gL / max(rL * (1 - rL), 1e-12)
        dL_dρ_H = gH / max(rH * (1 - rH), 1e-12)
        # effective Δρ for ONE optimizer step (assuming plain SGD with lr * grad)
        # raw_new = raw - lr * grad; Δρ ≈ -lr * grad * ρ(1-ρ)
        d_rho_L = -a['lr'] * gL * (rL * (1 - rL))
        d_rho_H = -a['lr'] * gH * (rH * (1 - rH))
        other = [p for n, p in policy.named_parameters() if 'rho_' not in n and p.grad is not None]
        other_avg = sum(p.grad.abs().sum().item() for p in other) / sum(p.numel() for p in other)
        print(f"  trial {trial}: loss={loss.item():.3f}")
        print(f"     ρ_L={rL:.4f}  grad_raw={gL:+.3e}  grad_wrt_ρ={dL_dρ_L:+.3e}  Δρ_per_step={d_rho_L:+.3e}")
        print(f"     ρ_H={rH:.4f}  grad_raw={gH:+.3e}  grad_wrt_ρ={dL_dρ_H:+.3e}  Δρ_per_step={d_rho_H:+.3e}")
        print(f"     avg other-param grad: {other_avg:.3e}  (compare to grad_raw above)")
        print(f"     direction: ρ_L → {'UP (recover)' if gL<0 else 'DOWN (collapse)'}, "
              f"ρ_H → {'UP' if gH<0 else 'DOWN'}")

    # ════ DIAG 2: forced ρ → loss curve ════
    print("\n=== DIAG 2: loss vs FORCED ρ (frozen, both at same value) ===")
    rho_grid = [0.0001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    losses = []
    with torch.no_grad():
        original_L = policy.rho_L_raw.data.clone()
        original_H = policy.rho_H_raw.data.clone()
        for rho in rho_grid:
            logit = math.log(rho / (1 - rho))
            policy.rho_L_raw.data.fill_(logit)
            policy.rho_H_raw.data.fill_(logit)
            loss, _, _ = policy.forward_loss(gt, vis, s_enc, n_inner=a['L_inner'],
                                              h_max=a['h_max'], mask_ratio_max=1.0)
            losses.append((rho, loss.item()))
        policy.rho_L_raw.data.copy_(original_L)
        policy.rho_H_raw.data.copy_(original_H)
    print(f"  {'ρ':>7s} | {'loss':>7s}")
    for rho, l in losses: print(f"  {rho:>7.4f} | {l:>7.3f}")
    best_rho, best_loss = min(losses, key=lambda x: x[1])
    print(f"  → best ρ (over grid): {best_rho:.3f}  (loss={best_loss:.3f})")

    # ════ DIAG 3: inner-loop refinement magnitudes ════
    print("\n=== DIAG 3: ||g(t)|| in the inner loop ===")
    with torch.no_grad():
        original_L = policy.rho_L_raw.data.clone()
        policy.rho_L_raw.data.fill_(0.5)                      # ρ=0.5 (clamp_direct param)
        B = vis.shape[0]
        kv_full = policy._build_kv(vis, s_enc)
        masks_zero = [torch.zeros(B, T_l, dtype=torch.bool, device=dev) for T_l in seq_lens]
        y = policy._y_embed(B, dev, gt, masks_zero)
        rL_tensor = policy.rho_L_raw.clamp(1e-3, 1 - 1e-3)
        wL = policy._weights(rL_tensor, a['L_inner'], dev, y.dtype)
        z_L = torch.zeros_like(y); z_H = torch.zeros_like(y)
        print(f"  using forced ρ_L=0.5, weights={[round(w.item(),3) for w in wL]}")
        print(f"  ||y||={y.norm().item():.2f}  ||kv||={kv_full.norm().item():.2f}")
        for t in range(a['L_inner']):
            g_t = policy.g(z_L + z_H + y, kv_full)
            scaled = wL[t] * g_t
            print(f"  t={t}: ||g(t)||={g_t.norm().item():.4f}  "
                  f"weight={wL[t].item():.4f}  ||scaled||={scaled.norm().item():.4f}  "
                  f"new z_L norm={(z_L + scaled).norm().item():.4f}")
            z_L = z_L + scaled
        policy.rho_L_raw.data.copy_(original_L)

    # ════ DIAG 4: counterfactual loss with ρ_L and ρ_H FROZEN INDEPENDENTLY ════
    print("\n=== DIAG 4: independent ρ_L × ρ_H grid (current model state) ===")
    print(f"  {'ρ_L \\ ρ_H':>10s}  " + "  ".join(f"{r:>6.3f}" for r in [0.001, 0.1, 0.3, 0.5, 0.9]))
    with torch.no_grad():
        oL = policy.rho_L_raw.data.clone(); oH = policy.rho_H_raw.data.clone()
        for rL_v in [0.001, 0.1, 0.3, 0.5, 0.9]:
            row = f"  {rL_v:>10.3f}  "
            for rH_v in [0.001, 0.1, 0.3, 0.5, 0.9]:
                policy.rho_L_raw.data.fill_(rL_v); policy.rho_H_raw.data.fill_(rH_v)
                loss, _, _ = policy.forward_loss(gt, vis, s_enc, n_inner=a['L_inner'],
                                                  h_max=a['h_max'], mask_ratio_max=1.0)
                row += f"  {loss.item():>5.3f}"
            print(row)
        policy.rho_L_raw.data.copy_(oL); policy.rho_H_raw.data.copy_(oH)

    # ════ DIAG 5: HOW MUCH does g(t≥1) actually CHANGE the prediction vs g(t=0) alone? ════
    print("\n=== DIAG 5: prediction CHANGE per inner step (at ρ_L=0.5) ===")
    print("  (measures: do later g() calls REFINE the prediction, or just add noise?)")
    with torch.no_grad():
        oL = policy.rho_L_raw.data.clone()
        policy.rho_L_raw.data.fill_(0.5)
        B = vis.shape[0]
        kv_full = policy._build_kv(vis, s_enc)
        masks_full = [torch.ones(B, T_l, dtype=torch.bool, device=dev) for T_l in seq_lens]
        y_masked = policy._y_embed(B, dev, None, masks_full)   # all-masked y (real inference)
        rL_tensor = policy.rho_L_raw.clamp(1e-3, 1 - 1e-3)
        wL = policy._weights(rL_tensor, a['L_inner'], dev, y_masked.dtype)
        z_L = torch.zeros_like(y_masked); z_H = torch.zeros_like(y_masked)
        prev_codes = None
        for t in range(a['L_inner']):
            g_t = policy.g(z_L + z_H + y_masked, kv_full)
            z_L = z_L + wL[t] * g_t
            # snapshot predicted codes after this inner step
            cur_logits = policy._heads(z_L)
            cur_codes = [cur_logits[l][..., :K].argmax(-1) for l in range(len(seq_lens))]
            if prev_codes is not None:
                # what % of predicted codes CHANGED vs previous inner step?
                n_diff = sum((cc != pc).sum().item() for cc, pc in zip(cur_codes, prev_codes))
                n_total = sum(cc.numel() for cc in cur_codes)
                print(f"  t={t}: {100*n_diff/n_total:.1f}% of predicted codes changed from t={t-1}")
            prev_codes = cur_codes
        policy.rho_L_raw.data.copy_(oL)
    print("\nInterpretation:")
    print("  If DIAG 4 shows ρ=(0.001, 0.001) IS the min → architecture really doesn't want recursion.")
    print("  If DIAG 4 shows a sweet spot OTHER than (0,0) → optimizer is failing to find it.")
    print("  If DIAG 5 shows codes flipping randomly between iterations → g() output is noise; recursion can't refine.")


if __name__ == '__main__':
    main()
