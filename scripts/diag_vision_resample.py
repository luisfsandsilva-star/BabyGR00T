#!/usr/bin/env python3
"""Theory test: is the policy actually USING the CNN content, or does it just need
*a plausible vision vector* in the slot (vision ~ semi-constant necessary input)?

Fit a low-rank Gaussian to the REAL CNN vision-token outputs (on train frames),
sample SYNTHETIC vision maps from it (uncorrelated with the eval sample's codes),
feed them to the frozen policy in place of real vision, and compare:
  - masked-CE acc  vs true / zero / shuffle
  - agreement      = fraction of masked positions where the prediction MATCHES the
                     true-vision prediction (high ⇒ output insensitive to vision content).

If gauss ≈ true (and high agreement) ⇒ vision content is NOT used (decorative slot).
If gauss ≈ zero ⇒ real content carries the info. If gauss < zero ⇒ random vision misleads.

Run: .venv/bin/python scripts/diag_vision_resample.py [--n 128]
"""
import os, sys, argparse, importlib.util
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
_spec = importlib.util.spec_from_file_location('eval_holdout', os.path.join(THIS, 'eval_holdout.py'))
eh = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eh)   # CPU mode
sent = eh.sent
import numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt-path', default='data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt')
ap.add_argument('--n', type=int, default=128)
ap.add_argument('--ratios', type=float, nargs='+', default=[0.15, 0.5])
ap.add_argument('--nseed', type=int, default=3)
args = ap.parse_args()

ctx_h, ck = eh.build_holdout_context(args.ckpt_path, 'widowx', args.n, 98765, ctrl_range=None)
ctx_c, _  = eh.build_holdout_context(args.ckpt_path, 'widowx', args.n, 98765, ctrl_range=(40, 60))
mods = sent.build_policy_from_ckpt(ck, ctx_h); eh.apply_ema(mods, ck)
for m in mods: m.eval()
cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy = mods


@torch.no_grad()
def real_vtok(ctx):
    v, _ = cnn(ctx['img_n']); return cnn_proj(v)          # (N,49,512)

# ── fit low-rank Gaussian to TRAIN real vision tokens ──
Vtr = real_vtok(ctx_c)                                     # (N,49,512)
N, T, D = Vtr.shape
X = Vtr.reshape(N, T * D).double().numpy()
mu = X.mean(0); Xc = X - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)          # Xc = U diag(S) Vt
pcstd = S / np.sqrt(N)                                     # std along each principal component
print(f"fitted Gaussian on TRAIN vtok: N={N}, dim={T*D}, top-PC std {pcstd[:6].round(2)}\n", flush=True)

def sample_gauss(k, m, seed):
    rng = np.random.RandomState(seed)
    z = rng.randn(m, k) * pcstd[:k]
    fake = mu + z @ Vt[:k]
    return torch.tensor(fake.reshape(m, T, D), dtype=torch.float32)


@torch.no_grad()
def encode_with(ctx, vtok):
    a = ctx['args']; dim = a['dim']; Bf = ctx['Bf']; eid = ctx['eid_t']
    ttok = text_proj(text_agg([ctx['t5s'][l] for l in range(9)]))
    etok = emb_id_emb(eid).view(Bf, a['n_emb_prefix'], dim)
    vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
    s_enc = state_encoders[ctx['robot']](ctx['states'])
    return vis, s_enc, eid

def mask_at(Bf, T_l, ratio, seed):
    g = torch.Generator(); g.manual_seed(seed)
    noise = torch.rand(Bf, T_l, generator=g); m = noise < ratio
    m[torch.arange(Bf), noise.argmin(1)] = True
    return m

@torch.no_grad()
def eval_vtok(ctx, vtok, ratio, ref_pred=None):
    a = ctx['args']; Bf = ctx['Bf']; vis, s_enc, eid = encode_with(ctx, vtok)
    gt = ctx['gt_codes']; T_l = gt[0].shape[1]; tc = tm = 0; agree = 0; agn = 0
    preds_all = []
    for s in range(args.nseed):
        m = mask_at(Bf, T_l, ratio, s * 1000 + 1)
        logits = policy(gt, vis, s_enc, mask_list=[m], n_outer=a['H_outer'], n_inner=a['L_inner'], emb_id=eid)[-1][0]
        pred = logits.argmax(-1); preds_all.append(pred)
        tc += ((pred == gt[0]) & m).float().sum().item(); tm += m.float().sum().item()
        if ref_pred is not None:
            agree += ((pred == ref_pred[s]) & m).float().sum().item(); agn += m.float().sum().item()
    return 100 * tc / max(1, tm), (100 * agree / max(1, agn) if ref_pred is not None else None), preds_all


for label, ctx in [('TRAINED', ctx_c), ('HELDOUT', ctx_h)]:
    vt_true = real_vtok(ctx); Bf = ctx['Bf']
    perm = torch.randperm(Bf)
    variants = [
        ('true',        vt_true),
        ('zero',        torch.zeros_like(vt_true)),
        ('shuffle',     vt_true[perm]),
        ('gauss k=6',   sample_gauss(6,  Bf, 0)),
        ('gauss k=20',  sample_gauss(20, Bf, 1)),
        ('gauss k=64',  sample_gauss(min(64, N-1), Bf, 2)),
    ]
    # reference predictions = true-vision predictions (per ratio)
    print(f"== {label} ==   (acc | agreement-with-true)")
    for r in args.ratios:
        _, _, ref = eval_vtok(ctx, vt_true, r)
        cells = []
        for nm, vt in variants:
            acc, agr, _ = eval_vtok(ctx, vt, r, ref_pred=ref)
            cells.append(f"{nm}={acc:4.1f}" + ("" if agr is None else f"/{agr:3.0f}"))
        print(f"   r={r:.2f}:  " + "   ".join(cells))
    print()
print("agreement = % masked positions where prediction == true-vision prediction")
