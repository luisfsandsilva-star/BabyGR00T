#!/usr/bin/env python3
"""Conditioning diagnostic for the CNN+T5 small policy. For each modality
(vision, text, state) we measure how much the predicted action codes CHANGE
when we ablate that modality (zero / shuffle across batch). If Δ≈0, the
policy is ignoring that input — same failure mode the old PerceiverResampler
had (see babygroot-vision-collapse.md).

Usage: python -m scripts.diag_conditioning <ckpt> [N]
"""
import os, sys, math, random
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import (RevIN, ActionVQVAE1d, VQ1d_EMA, STRMPolicyVAE,
                            LayerAggregator, ScaleNorm, load_lerobot_episodes)
from babygroot_strm.cnn_vision import EfficientCNN

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'data/ckpts/cnn_policy_small_v2_finetune.pt'
N         = int(sys.argv[2]) if len(sys.argv) > 2 else 256
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0); random.seed(0)

c = torch.load(ckpt_path, map_location=dev, weights_only=False); a = c['args']
vck = torch.load(a['vae_ckpt'], map_location=dev, weights_only=False); adim = vck.get('action_dim', 7)
vae = ActionVQVAE1d(action_dim=adim, vq_cls=VQ1d_EMA).to(dev); revin = RevIN(adim).to(dev)
vae.load_state_dict(vck['vae']); revin.load_state_dict(vck['revin']); vae.eval(); revin.eval()
seq_lens = tuple(vae.seq_lens); K = vae.vqs[0].K
t5 = torch.load(a['t5_cache'], map_location='cpu', weights_only=False)
t5e, t5dim, t5L = t5['embeddings'], t5['dim'], t5['n_layers']

# Use heldout episodes for the diagnostic so we're testing GENERALIZATION conditioning, not memorization.
heldout_by_task = c.get('heldout_eps_by_task')
if heldout_by_task:
    heldout_idx = sorted({e for v in heldout_by_task.values() for e in v})
    print(f"using HELDOUT episodes: {len(heldout_idx)} across {len(heldout_by_task)} tasks")
    eps = load_lerobot_episodes(a['oxe_dataset_id'], camera_key=a['oxe_camera'],
                                load_video=True, episode_indices=heldout_idx)
else:
    eps = load_lerobot_episodes(a['oxe_dataset_id'], camera_key=a['oxe_camera'],
                                load_video=True, n_episodes=64)
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
if 'ema_params' in c and os.environ.get('NO_EMA', '0') != '1':
    for nm, m in [('cnn', cnn), ('text_agg', text_agg), ('cnn_proj', cnn_proj),
                  ('text_proj', text_proj), ('kv_norm', kv_norm), ('policy', policy)]:
        sd = m.state_dict()
        for k in list(sd.keys()):
            full = f"{nm}.{k}"
            if full in c['ema_params']:
                sd[k] = c['ema_params'][full].to(sd[k].dtype).to(sd[k].device)
        m.load_state_dict(sd)
    print(f"[ema] swapped in EMA shadow params (decay={c.get('ema_decay','?')})")

# Sample N chunks
index = [(ei, ci) for ei, e in enumerate(eps) for ci in range(1, e[0].shape[0])]
picks = [random.choice(index) for _ in range(N)]
def f2t(p): p = p.convert('RGB').resize((img, img)); return torch.from_numpy(np.asarray(p)).permute(2, 0, 1).float() / 255.
fr = torch.stack([f2t(eps[e][2][ci][-1]) for e, ci in picks]).to(dev)
st = torch.stack([eps[e][1][ci] for e, ci in picks]).float().to(dev)
tk = [eps[e][3] for e, ci in picks]

def t5_batch(tasks):
    out = torch.zeros(t5L, len(tasks), max_text, t5dim)
    for b, t in enumerate(tasks):
        e = t5e.get(t)
        if e is not None:
            h = e['hidden'].float(); n = min(h.shape[1], max_text); out[:, b, :n, :] = h[:, :n, :]
    return out.to(dev)

@torch.no_grad()
def predict(vtok, ttok, st_):
    """Run policy from already-projected modality tokens."""
    vis = kv_norm(torch.cat([vtok, ttok], dim=1))
    logits = policy(None, vis, st_, mask_list=None, n_outer=a['H_outer'], n_inner=a['L_inner'])[-1]
    return [logits[l][..., :K].argmax(-1) for l in range(len(seq_lens))]    # list of (B, T_l)

with torch.no_grad():
    vtok = cnn_proj(cnn(fr)[0])                                              # (B, n_vis, dim)
    t5s = t5_batch(tk)
    ttok = text_proj(text_agg([t5s[l] for l in range(t5L)]))                 # (B, max_text, dim)

base = predict(vtok, ttok, st)

# helpers
B = vtok.shape[0]
def shuf(x):
    perm = torch.randperm(B, device=x.device)
    while (perm == torch.arange(B, device=x.device)).any():
        perm = torch.randperm(B, device=x.device)
    return x[perm]

vtok_zero = torch.zeros_like(vtok)
ttok_zero = torch.zeros_like(ttok)
st_zero   = torch.zeros_like(st)
vtok_shuf = shuf(vtok)
ttok_shuf = shuf(ttok)
st_shuf   = shuf(st)

variants = {
    'real (baseline reference)':        (vtok,        ttok,        st),
    'zero VISION':                      (vtok_zero,   ttok,        st),
    'shuffled VISION (across batch)':   (vtok_shuf,   ttok,        st),
    'zero TEXT':                        (vtok,        ttok_zero,   st),
    'shuffled TEXT':                    (vtok,        ttok_shuf,   st),
    'zero STATE':                       (vtok,        ttok,        st_zero),
    'shuffled STATE':                   (vtok,        ttok,        st_shuf),
    'zero EVERYTHING (modal prior)':    (vtok_zero,   ttok_zero,   st_zero),
}

def code_change_pct(pred):
    """% of code positions that differ from the baseline prediction."""
    diff = total = 0
    for l in range(len(seq_lens)):
        diff += (pred[l] != base[l]).sum().item()
        total += base[l].numel()
    return 100.0 * diff / total

print(f"\n=== conditioning sensitivity (N={N} heldout chunks, % codes changed from real-input baseline) ===")
print(f"  {'variant':<38s} | {'Δcodes':>8s}  | meaning")
print(f"  {'-'*38} + {'-'*8} -+-{'-'*40}")
for name, (v, t, s) in variants.items():
    p = predict(v, t, s)
    d = code_change_pct(p)
    if name.startswith('real'): note = "must be 0 — sanity"
    elif name.startswith('zero EVERYTHING'): note = "policy's unconditional prior"
    elif 'zero' in name: note = "Δ=0 means modality unused; >>0 means it matters"
    elif 'shuffled' in name: note = "Δ=0 means modality unused; ≈zero-baseline means used"
    else: note = ""
    print(f"  {name:<38s} | {d:7.1f}% | {note}")
