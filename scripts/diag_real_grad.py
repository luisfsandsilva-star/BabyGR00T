#!/usr/bin/env python3
"""Real-architecture gradient diagnostic. Build the actual v10 stack (Bayesian VAE
+ 16 emb prefix + per-emb head, dim=768, depth=3), grab a real OXE batch, compute
gradients, and inspect:

  1. Per-parameter gradient norms — where is the gradient concentrated?
  2. Per-fusion-step Jacobian magnitudes — does the inner loop amplify gradients?
  3. Logprec values produced by g — where are they relative to the clamp?
  4. Compare moment-form fusion vs information-form for per-step gradient flow

Run on CPU; we just need ONE forward+backward — not training.
"""
import os, sys, glob, json, math, random
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
from babygroot_strm import (RevIN, ActionVQVAE1d, VQ1d_EMA, STRMPolicy, STRMPolicyVAE,
                            LayerAggregator, ScaleNorm)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset, EMBODIMENTS, EMBODIMENT_ID
from babygroot_strm.cond_vae import CondActionVQVAE1d

_DEV = 'cpu'                                                      # run on CPU; v11 owns GPU
torch.manual_seed(0); random.seed(0)


def build_fresh_v10_policy(dim=768, depth=3, n_emb_prefix=16, K=512):
    """Random-init STRMPolicyVAE in Bayesian mode, matching v10 spec."""
    pol = STRMPolicyVAE(
        seq_lens=(4,), k_codebook=K, dim=dim, heads=8,
        depth=depth, L_inner=5, H_outer=2,
        state_dim=dim, max_prefix=49 + 24 + 16 + n_emb_prefix,
        beta=1e-3, free_bits=0.1,
        update_mode='bayesian', alpha_parametrization='sigmoid',
        n_embodiments=12, per_emb_head=True,
    ).to(_DEV)
    return pol


def load_real_batch(batch_size=8):
    """Load one real OXE batch (small B for CPU)."""
    specs = []
    for ds_dir in sorted(glob.glob('/home/research/Projects/BBGr/BabyGR00T/data/oxe/*'))[:3]:
        if not os.path.isfile(os.path.join(ds_dir, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(ds_dir, chunk_len=16, lookback=16)
            if sp.chunk_index: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    idxs = [random.randint(0, len(ds) - 1) for _ in range(batch_size)]
    batch = [ds[i] for i in idxs]
    frames = torch.stack([
        (torch.from_numpy(np.asarray(b[0].convert('RGB').resize((224, 224))))
         .permute(2, 0, 1).float() / 255.) for b in batch])
    states  = torch.stack([b[1] for b in batch])
    actions = torch.stack([b[2] for b in batch])
    prevs   = torch.stack([b[3] for b in batch])
    tasks   = [b[4] for b in batch]
    emb_robots = [EMBODIMENTS[b[5]] if b[5] < len(EMBODIMENTS) else 'unknown' for b in batch]
    return frames, states, actions, prevs, tasks, emb_robots


def build_vis(frames, tasks, emb_robots, dim=768, n_emb_prefix=16):
    """Mimic the real encode_modalities path (frozen-ish, just zero-init for speed)."""
    # Use simple fresh modules for the encoder side — we only care about gradients in the policy
    cnn = EfficientCNN(dims=(24, 48, 96, 192), depths=(1, 1, 1, 1), expand=2,
                        out_dim=192, norm='scalenorm', pos_emb=True, img_size=224).to(_DEV)
    cnn_proj = nn.Linear(192, dim).to(_DEV)
    text_proj = nn.Linear(512, dim).to(_DEV)
    kv_norm = ScaleNorm(dim).to(_DEV)
    emb_id_emb = nn.Embedding(12, dim * n_emb_prefix).to(_DEV)

    img_var = torch.load('/home/research/Projects/BBGr/BabyGR00T/data/cache/image_var_global.pt',
                          map_location='cpu', weights_only=False)
    var_global_img = img_var['var_global']
    x = normalize_image(frames, var_global_img)
    vtok, _ = cnn(x); vtok = cnn_proj(vtok)
    # fake text: zeros (we're testing policy gradients, not text encoder)
    ttok = torch.zeros(vtok.shape[0], 24, dim)
    idx = torch.tensor([EMBODIMENT_ID.get(r, len(EMBODIMENTS)) for r in emb_robots])
    etok = emb_id_emb(idx).view(idx.shape[0], n_emb_prefix, dim)
    vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
    return vis


def build_state_enc(states, dim=768):
    se = nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim)).to(_DEV)
    return se(states)


def build_targets(actions, prevs, emb_robots, K=512):
    """Use the real shared VAE to get target codes."""
    sc = torch.load('/home/research/Projects/BBGr/BabyGR00T/data/ckpts/oxe_shared_vae.pt',
                     map_location='cpu', weights_only=False)
    shared_vae = CondActionVQVAE1d(action_dim=sc['action_dim'], n_embodiments=sc['n_embodiments'], k=sc['k']).to(_DEV)
    shared_vae.load_state_dict(sc['vae']); shared_vae.eval()
    var_globals = {emb: sc['action_var_globals'][EMBODIMENT_ID.get(emb, -1)].view(1, 1, -1)
                   for emb in EMBODIMENTS if EMBODIMENT_ID.get(emb, -1) in sc['action_var_globals']}
    all_codes = [torch.zeros(actions.shape[0], 4, dtype=torch.long)]
    for emb in set(emb_robots):
        mask = torch.tensor([r == emb for r in emb_robots])
        if mask.sum() == 0 or emb not in var_globals: continue
        ac = actions[mask]; pv = prevs[mask]
        vg = var_globals[emb]
        nT = ac.shape[1]
        m = pv.mean(dim=1, keepdim=True); S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
        lam = nT / (S + nT * vg)
        xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
        eid = torch.full((ac.shape[0],), EMBODIMENT_ID.get(emb, len(EMBODIMENTS)), dtype=torch.long)
        gt, _ = shared_vae.encode_with_soft(xn, eid, tau=0.1)
        all_codes[0][mask] = gt[0]
    return all_codes


# ──────────────────────────────────────────────────────────────────────
# Main diagnostic
# ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Setting up real v10 architecture on CPU ===")
    print("Loading real batch from OXE...")
    frames, states, actions, prevs, tasks, emb_robots = load_real_batch(batch_size=8)
    print(f"  batch: B=8, embodiments={emb_robots}")

    vis = build_vis(frames, tasks, emb_robots)
    s_enc = build_state_enc(states)
    gt = build_targets(actions, prevs, emb_robots)
    print(f"  vis: {vis.shape}, s_enc: {s_enc.shape}, gt[0]: {gt[0].shape}")

    # Fresh v10 policy
    print("\n=== Building fresh v10 policy (Bayesian, dim=768, depth=3) ===")
    policy = build_fresh_v10_policy()
    print(f"  policy params: {sum(p.numel() for p in policy.parameters())/1e6:.1f}M")

    # Forward + backward
    emb_id = torch.tensor([EMBODIMENT_ID.get(r, len(EMBODIMENTS)) for r in emb_robots])
    print(f"  emb_id: {emb_id.tolist()}")

    print("\n=== Forward + backward at fresh init ===")
    loss, _, _ = policy.forward_loss(gt, vis, s_enc, n_inner=5, h_max=2,
                                     mask_ratio_max=1.0, emb_id=emb_id, label_smoothing=0.05)
    loss.backward()
    print(f"  loss = {loss.item():.4f}")

    # ── (1) Top per-param gradient norms ──
    print("\n=== (1) Top 15 parameters by ||grad|| ===")
    grads = []
    for n, p in policy.named_parameters():
        if p.grad is not None:
            grads.append((n, p.data.norm().item(), p.grad.norm().item(),
                          p.grad.abs().max().item(), p.numel()))
    grads.sort(key=lambda x: x[2], reverse=True)
    print(f"  {'param':<45s} {'||W||':>10s} {'||grad||':>10s} {'max|g|':>10s} {'n':>8s}")
    for n, wn, gn, gm, num in grads[:15]:
        print(f"  {n:<45s} {wn:>10.2e} {gn:>10.2e} {gm:>10.2e} {num:>8d}")

    # ── (2) Inspect logprec output of g(z + c) at fresh init ──
    print("\n=== (2) Logprec channel statistics from g's output ===")
    # Run g manually on a fresh z, look at the logprec half of its output
    with torch.no_grad():
        B = vis.shape[0]
        kv = policy._build_kv(vis, s_enc)
        y = policy._y_embed(B, _DEV, gt, [torch.ones(B, 4, dtype=torch.bool)])
        z = torch.zeros_like(y)
        g_out = policy.g(z + y, kv)
        latent_dim = policy.latent_dim
        lp = g_out[..., latent_dim:]
        print(f"  g's logprec output: shape {tuple(lp.shape)}, dtype {lp.dtype}")
        print(f"    range: [{lp.min().item():.2f}, {lp.max().item():.2f}]  mean={lp.mean().item():.2f}  std={lp.std().item():.2f}")
        print(f"    % at clamp boundary (|lp|>4.5): {(lp.abs() > 4.5).float().mean().item()*100:.1f}%")
        print(f"    implied τ range: [{lp.min().exp().item():.2e}, {lp.clamp(max=5).max().exp().item():.2e}]")

    # ── (3) Gradient through iterated fusion: how does ||∂loss/∂g(t)|| scale with t? ──
    print("\n=== (3) Gradient through inner-loop iterations (forced rho, retain_grad) ===")
    # Manually run inner loop, retain grad on each g_t, backward, inspect
    policy.zero_grad()
    B = vis.shape[0]
    kv = policy._build_kv(vis, s_enc)
    y_full = policy._y_embed(B, _DEV, gt, [torch.ones(B, 4, dtype=torch.bool)])
    z = torch.zeros_like(y_full)
    g_outs = []
    for t in range(5):
        g_t = policy.g(z + y_full, kv)
        g_t.retain_grad()
        g_outs.append(g_t)
        z = policy._bayes_fuse(z, g_t, latent_dim)
    # head + loss
    logits, mu, lp = policy._sample_heads(z, emb_id=emb_id)
    K = logits[0].shape[-1]
    target_lp_log = F.log_softmax(logits[0][..., :K], dim=-1)
    target_lp_lossterm = -target_lp_log.gather(-1, gt[0].unsqueeze(-1)).squeeze(-1).mean()
    target_lp_lossterm.backward(retain_graph=True)
    print(f"  {'t':>3s} | {'||g(t)||':>12s} {'||grad g(t)||':>15s} {'||scaled||':>12s}")
    for t, g_t in enumerate(g_outs):
        gn = g_t.norm().item()
        gradn = g_t.grad.norm().item() if g_t.grad is not None else 0.0
        # the "scaled" is the contribution to ||g||'s update via this step
        print(f"  {t:>3d} | {gn:>12.4e} {gradn:>15.4e} {gn*gradn:>12.4e}")

    # ── (4) Compare moment-form vs info-form fusion gradient at fresh init ──
    print("\n=== (4) moment vs info form: gradient norm comparison at fresh init ===")
    # Reset network to clean state
    policy.zero_grad()
    z_p = torch.randn(B, 4, policy.dim, requires_grad=True)
    z_g_random = torch.randn(B, 4, policy.dim, requires_grad=True) * 0.5
    # moment fuse
    z_post_m = policy._bayes_fuse(z_p, z_g_random, latent_dim)
    loss_m = (z_post_m ** 2).sum()
    loss_m.backward(retain_graph=True)
    grad_zp_m = z_p.grad.norm().item()
    grad_zg_m = z_g_random.grad.norm().item()
    print(f"  moment: ||grad z_p|| = {grad_zp_m:.3e}, ||grad z_g|| = {grad_zg_m:.3e}")


if __name__ == '__main__':
    main()
