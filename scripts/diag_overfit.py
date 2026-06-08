#!/usr/bin/env python3
"""Decompose WHAT is overfit: vision vs policy/sequence-model.

Grid over (domain × vision-ablation × mask-ratio), masked-CE acc:
  domain : TRAINED (len 40-60, start=16 windows in training) vs HELDOUT (len==32, never trained)
  vision : true | zero | shuffle (vtok replaced before kv_norm)
  ratio  : fraction of action-code positions masked (low = code-context easy; high = must use vision)

Reads:
  • vis=zero, LOW ratio  → policy predicting codes from code-context alone.
      HELDOUT≈TRAINED ⇒ sequence model GENERALIZES (policy not overfit).
  • (true − zero) at HIGH ratio → vision's real contribution.
      large on TRAINED but ~0/negative on HELDOUT ⇒ vision features OVERFIT.

Run: .venv/bin/python scripts/diag_overfit.py [--ckpt-path ...] [--n 128]
"""
import os, sys, argparse, importlib.util
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
_spec = importlib.util.spec_from_file_location('eval_holdout', os.path.join(THIS, 'eval_holdout.py'))
eh = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(eh)   # imports sentinel → CPU mode
sent = eh.sent
import numpy as np, torch, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt-path', default='data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt')
ap.add_argument('--n', type=int, default=128)
ap.add_argument('--ema', action='store_true', default=True)
ap.add_argument('--ratios', type=float, nargs='+', default=[0.15, 0.5, 0.9])
ap.add_argument('--nseed', type=int, default=3)
args = ap.parse_args()

ctx_h, ck = eh.build_holdout_context(args.ckpt_path, 'widowx', args.n, 98765, ctrl_range=None)
ctx_c, _  = eh.build_holdout_context(args.ckpt_path, 'widowx', args.n, 98765, ctrl_range=(40, 60))
mods = sent.build_policy_from_ckpt(ck, ctx_h)
nset = eh.apply_ema(mods, ck) if args.ema else 0
for m in mods: m.eval()
cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy = mods
print(f"loaded {os.path.basename(args.ckpt_path)} step={ck.get('step')} EMA={args.ema}({nset})  N={args.n}\n")


@torch.no_grad()
def encode(ctx, vision='true', vkeep=1.0, state=True, text=True):
    a = ctx['args']; dim = a['dim']; Bf = ctx['Bf']; eid = ctx['eid_t']
    vtok, _ = cnn(ctx['img_n']); vtok = cnn_proj(vtok)
    if vision == 'zero':      vtok = torch.zeros_like(vtok)
    elif vision == 'shuffle': vtok = vtok[torch.randperm(Bf)]
    elif vkeep < 1.0:                                  # keep a random subset of vision tokens, zero the rest
        Tv = vtok.shape[1]; k = max(1, int(round(Tv * vkeep)))
        g = torch.Generator(); g.manual_seed(0)
        keep = torch.zeros(Tv, dtype=torch.bool); keep[torch.randperm(Tv, generator=g)[:k]] = True
        vtok = vtok * keep.view(1, Tv, 1)
    ttok = text_proj(text_agg([ctx['t5s'][l] for l in range(9)]))
    if not text: ttok = torch.zeros_like(ttok)
    etok = emb_id_emb(eid).view(Bf, a['n_emb_prefix'], dim)
    vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
    s_enc = state_encoders[ctx['robot']](ctx['states'])
    if not state: s_enc = torch.zeros_like(s_enc)
    return vis, s_enc, eid


def mask_at(Bf, T_l, ratio, seed):
    g = torch.Generator(); g.manual_seed(seed)
    noise = torch.rand(Bf, T_l, generator=g)
    m = noise < ratio
    m[torch.arange(Bf), noise.argmin(1)] = True       # ≥1 masked per row
    return m


@torch.no_grad()
def acc(ctx, enc_kw, ratio):
    a = ctx['args']; Bf = ctx['Bf']
    vis, s_enc, eid = encode(ctx, **enc_kw)
    gt = ctx['gt_codes']; T_l = gt[0].shape[1]; tc = tm = 0
    for s in range(args.nseed):
        m = mask_at(Bf, T_l, ratio, s * 1000 + 1)
        logits = policy(gt, vis, s_enc, mask_list=[m], n_outer=a['H_outer'], n_inner=a['L_inner'], emb_id=eid)[-1][0]
        pred = logits.argmax(-1); tgt = gt[0]
        tc += ((pred == tgt) & m).float().sum().item(); tm += m.float().sum().item()
    return 100 * tc / max(1, tm)


ROWS = [
    ('full',                 dict()),
    ('vis=zero',             dict(vision='zero')),
    ('vis=shuffle',          dict(vision='shuffle')),
    ('vis keep50%',          dict(vkeep=0.5)),
    ('vis keep25%',          dict(vkeep=0.25)),
    ('state=zero',           dict(state=False)),
    ('text=zero',            dict(text=False)),
    ('code-only (v/s/t=0)',  dict(vision='zero', state=False, text=False)),
]
hdr = 'mask→ ' + '  '.join(f'r={r:.2f}' for r in args.ratios)
for label, ctx in [('TRAINED (in-train)', ctx_c), ('HELDOUT (len==32)', ctx_h)]:
    print(f'== {label} ==   {hdr}')
    for name, kw in ROWS:
        row = '  '.join(f'{acc(ctx, kw, r):6.1f}' for r in args.ratios)
        print(f'   {name:<20} {row}')
    print()
print('random baseline = 100/256 = 0.39%')
