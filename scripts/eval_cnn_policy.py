#!/usr/bin/env python3
"""Action-space benchmark for the CNN+T5 policy on BridgeData V2.

Runs the policy ALL-MASKED (predict the whole 7-DoF action chunk from
vision+state alone — the real inference case), decodes predicted codes to
actions, and reports: codebook top-k accuracy, action MSE vs the VQ-VAE recon
floor, per-DoF MSE, and per-DoF predicted-vs-GT *std* (the "does it generate
real movement, not a constant?" check) + variance-explained.

Usage: python -m scripts.eval_cnn_policy [ckpt] [N] [n_eps]
"""
import os, sys, math, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import (RevIN, ActionVQVAE1d, VQ1d_EMA, STRMPolicyVAE,
                            LayerAggregator, ScaleNorm, load_lerobot_episodes)
from babygroot_strm.cnn_vision import EfficientCNN

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'data/ckpts/cnn_policy_v3pe_step4000.pt'
N      = int(sys.argv[2]) if len(sys.argv) > 2 else 96
n_eps  = int(sys.argv[3]) if len(sys.argv) > 3 else 64
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); random.seed(0)

c = torch.load(ckpt_path, map_location=dev, weights_only=False); a = c['args']
vck = torch.load(a['vae_ckpt'], map_location=dev, weights_only=False); adim = vck.get('action_dim', 7)
vae = ActionVQVAE1d(action_dim=adim, vq_cls=VQ1d_EMA).to(dev); revin = RevIN(adim).to(dev)
vae.load_state_dict(vck['vae']); revin.load_state_dict(vck['revin']); vae.eval(); revin.eval()
var_global = vck['action_var_global'].to(dev).view(1, 1, -1)   # precision prior (matches training)
seq_lens = tuple(vae.seq_lens); K = vae.vqs[0].K
t5 = torch.load(a['t5_cache'], map_location='cpu', weights_only=False)
t5e, t5dim, t5L = t5['embeddings'], t5['dim'], t5['n_layers']

# If the ckpt has held-out episodes per task, use them; otherwise fall back to the first n_eps.
heldout_by_task = c.get('heldout_eps_by_task')
if heldout_by_task:
    heldout_idx = sorted({e for v in heldout_by_task.values() for e in v})
    print(f"using HELDOUT episodes from ckpt: {len(heldout_idx)} eps across {len(heldout_by_task)} tasks")
    eps = load_lerobot_episodes(a['oxe_dataset_id'], camera_key=a['oxe_camera'],
                                load_video=True, episode_indices=heldout_idx)
else:
    eps = load_lerobot_episodes(a['oxe_dataset_id'], camera_key=a['oxe_camera'],
                                load_video=True, n_episodes=n_eps)
state_dim = int(eps[0][1].shape[-1]); img = a['img_size']; max_text = a['max_text']
n_vis = (img // 32) ** 2

cnn = EfficientCNN(dims=tuple(a['cnn_dims']), depths=tuple(a['cnn_depths']), expand=a['cnn_expand'],
                   out_dim=a['cnn_out_dim'], norm=a['cnn_norm'], pos_emb=a['cnn_pe'], img_size=img).to(dev)
text_agg = LayerAggregator(hidden_dim=t5dim, n_layers=t5L).to(dev)
cnn_proj = nn.Linear(a['cnn_out_dim'], a['dim']).to(dev)
text_proj = nn.Linear(t5dim, a['dim']).to(dev)
kv_norm = ScaleNorm(a['dim']).to(dev)
policy = STRMPolicyVAE(seq_lens=seq_lens, k_codebook=K, dim=a['dim'], heads=8, depth=a['depth'],
                       L_inner=a['L_inner'], H_outer=a['H_outer'], state_dim=state_dim,
                       max_prefix=n_vis + max_text + 16, beta=a['beta'], free_bits=a['free_bits']).to(dev)
for name, m in [('cnn', cnn), ('text_agg', text_agg), ('cnn_proj', cnn_proj),
                ('text_proj', text_proj), ('kv_norm', kv_norm), ('policy', policy)]:
    m.load_state_dict(c[name]); m.eval()
# Swap in Polyak/EMA shadow params if present and not disabled (NO_EMA=1).
if 'ema_params' in c and os.environ.get('NO_EMA', '0') != '1':
    n_swap = 0
    for nm, m in [('cnn', cnn), ('text_agg', text_agg), ('cnn_proj', cnn_proj),
                  ('text_proj', text_proj), ('kv_norm', kv_norm), ('policy', policy)]:
        sd = m.state_dict()
        for k in list(sd.keys()):
            full = f"{nm}.{k}"
            if full in c['ema_params']:
                sd[k] = c['ema_params'][full].to(sd[k].dtype).to(sd[k].device); n_swap += 1
        m.load_state_dict(sd)
    print(f"[ema] swapped in {n_swap} shadow params (decay={c.get('ema_decay','?')})")
print(f"loaded {os.path.basename(ckpt_path)} step={c['step']} | cnn_norm={a['cnn_norm']} pe={a['cnn_pe']} "
      f"h_max={a['h_max']} | ρ_L={torch.sigmoid(policy.rho_L_raw).item():.3f} ρ_H={torch.sigmoid(policy.rho_H_raw).item():.3f}")

# Skip ci=0 — we need a prev chunk for the precision normalization (same as training).
index = [(ei, ci) for ei, e in enumerate(eps) for ci in range(1, e[0].shape[0])]
picks = [random.choice(index) for _ in range(N)]
def f2t(p): p = p.convert('RGB').resize((img, img)); return torch.from_numpy(np.asarray(p)).permute(2, 0, 1).float() / 255.
fr = torch.stack([f2t(eps[e][2][ci][-1]) for e, ci in picks]).to(dev)
st = torch.stack([eps[e][1][ci] for e, ci in picks]).float().to(dev)
ac = torch.stack([eps[e][0][ci] for e, ci in picks]).float().to(dev)
pv = torch.stack([eps[e][0][ci - 1] for e, ci in picks]).float().to(dev)   # prev chunk (lookback)
tk = [eps[e][3] for e, ci in picks]

def decode_action(idxs):
    B = idxs[0].shape[0]; embs = []
    for vq, T_l, idx in zip(vae.vqs, vae.seq_lens, idxs):
        embs.append(vq.emb(idx).view(B, T_l, vq.D).permute(0, 2, 1))
    return vae.decode(embs).transpose(1, 2)              # (B, T, A) normalized

def t5_batch(tasks):
    out = torch.zeros(t5L, len(tasks), max_text, t5dim)
    for b, t in enumerate(tasks):
        e = t5e.get(t)
        if e is not None:
            h = e['hidden'].float(); n = min(h.shape[1], max_text); out[:, b, :n, :] = h[:, :n, :]
    return out.to(dev)

with torch.no_grad():
    vtok = cnn_proj(cnn(fr)[0]); t5s = t5_batch(tk)
    ttok = text_proj(text_agg([t5s[l] for l in range(t5L)]))
    vis = kv_norm(torch.cat([vtok, ttok], dim=1))
    # SAME normalization as training: precision λ from prev chunk + global prior (Gamma-conjugate),
    # NOT the old RevIN running-stats — those didn't match the training target distribution.
    nT = ac.shape[1]
    m = pv.mean(dim=1, keepdim=True)
    S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
    lam = nT / (S + nT * var_global)                        # precision (B,1,A), bounded
    x_norm = ((ac - m) * lam.sqrt()).transpose(1, 2)        # (B,A,T) in TRAINING space
    gt, _ = vae.encode_with_soft(x_norm, tau=0.1)           # GT codes (training-space)
    gt_recon = decode_action(gt)                            # VAE recon (B,T,A) — the floor
    actual = x_norm.transpose(1, 2)                         # actual normalized action (B,T,A)
    H = a['H_outer']; L = a['L_inner']
    logits = policy(None, vis, st, mask_list=None, n_outer=H, n_inner=L)[-1]
    pred_idx = [logits[l][..., :K].argmax(-1) for l in range(len(seq_lens))]
    pred = decode_action(pred_idx)                          # predicted action (B,T,A)

# codebook accuracy (all-masked)
TOPK = [1, 5]
acc = {k: 0 for k in TOPK}; tot = 0
for l in range(len(seq_lens)):
    for k in TOPK:
        topk = logits[l][..., :K].topk(k, -1).indices
        acc[k] += (topk == gt[l].unsqueeze(-1)).any(-1).sum().item()
    tot += gt[l].numel()
print(f"\n=== codebook acc (all-masked, N={N}) ===  top1={acc[1]/tot*100:.1f}%  top5={acc[5]/tot*100:.1f}%  (random top1={100/K:.2f}%)")

# action-space metrics (normalized space)
A = actual.shape[-1]; names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grip'][:A]
mse_pol = ((pred - actual) ** 2).mean().item()
mse_flo = ((gt_recon - actual) ** 2).mean().item()
print(f"\n=== action MSE (RevIN-normalized) ===  policy={mse_pol:.4f}   VAE-recon floor={mse_flo:.4f}   "
      f"(ratio {mse_pol/max(mse_flo,1e-9):.1f}× floor)")
print(f"\n=== per-DoF: does it generate real movement? ===")
print(f"  {'DoF':>5} | {'MSE':>7} | {'std_pred':>8} {'std_GT':>7} {'ratio':>6} | {'var_expl':>8}")
for d in range(A):
    p, g = pred[..., d], actual[..., d]
    mse = ((p - g) ** 2).mean().item()
    sp, sg = p.std().item(), g.std().item()
    ve = 1 - mse / max(g.var().item(), 1e-9)
    print(f"  {names[d]:>5} | {mse:7.4f} | {sp:8.4f} {sg:7.4f} {sp/max(sg,1e-9):6.2f} | {ve:8.2f}")
print(f"\n  std_pred/std_GT ≈ 1 → real movement; ≈ 0 → near-constant (no movement). "
      f"var_expl = 1 − MSE/var(GT) (1=perfect, 0=predicting the mean, <0=worse than mean).")
# how much movement is there at all (mean abs delta across the chunk)?
gt_motion = (actual[:, 1:] - actual[:, :-1]).abs().mean().item()
pred_motion = (pred[:, 1:] - pred[:, :-1]).abs().mean().item()
print(f"\n  within-chunk motion (mean |Δ| step-to-step):  GT={gt_motion:.4f}  policy={pred_motion:.4f}  "
      f"(ratio {pred_motion/max(gt_motion,1e-9):.2f})")
