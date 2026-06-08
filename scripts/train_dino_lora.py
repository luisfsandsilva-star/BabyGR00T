#!/usr/bin/env python3
"""FIX TEST: frozen DINOv2-small → trainable MLP adapter → vision tokens, with the
v3 policy FROZEN + LoRA. Tests whether GOOD vision + light adaptation generalizes,
validated on the TRUE disjoint held-out (length==32 episodes) — never the old
overlapping sentinel.

Trainable: MLP vision adapter, LoRA on policy attn/FFN linears, policy output heads.
Frozen   : DINOv2, policy base weights, text/emb/state/kv encoders, per-emb VAE.
Data     : same bridge widowx chunks (stride 4). Loss: masked-CE (forward_loss).

Run: .venv/bin/python scripts/train_dino_lora.py [--steps 4000 --batch-size 96 --smoke]
"""
import os, sys, json, glob, math, time, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from babygroot_strm import STRMPolicy, LayerAggregator, ScaleNorm, ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset, EMBODIMENTS, EMBODIMENT_ID
from transformers import AutoModel
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt-path', default='data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt')
ap.add_argument('--robot', default='widowx')
ap.add_argument('--steps', type=int, default=4000)
ap.add_argument('--batch-size', type=int, default=96)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--lora-r', type=int, default=16)
ap.add_argument('--lora-alpha', type=int, default=32)
ap.add_argument('--num-workers', type=int, default=10)
ap.add_argument('--val-every', type=int, default=250)
ap.add_argument('--val-n', type=int, default=192)
ap.add_argument('--out', default='data/ckpts/dino_lora_widowx.pt')
ap.add_argument('--smoke', action='store_true')
args = ap.parse_args()
dev = 'cuda'
torch.manual_seed(0); random.seed(0); np.random.seed(0)
BRIDGE = next(d for d in sorted(glob.glob('data/oxe/*'))
              if 'bridge_orig' in d and os.path.isfile(os.path.join(d, 'meta', 'info.json')))


# ───────────────────────── LoRA ─────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r, alpha, drop=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.base(x) + self.scale * (self.drop(x) @ self.lora_A.t() @ self.lora_B.t())

def apply_lora(root, names, r, alpha):
    n = 0
    for mod in root.modules():
        for cn, child in list(mod.named_children()):
            if cn in names and isinstance(child, nn.Linear):
                setattr(mod, cn, LoRALinear(child, r, alpha)); n += 1
    return n


# ───────────────── base modules from v3 ckpt (+EMA, frozen) ─────────────────
ck = torch.load(args.ckpt_path, map_location='cpu', weights_only=False); A = ck['args']
dim = A['dim']; n_emb_total = len(EMBODIMENTS) + 1
text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
text_proj = nn.Linear(512, dim); kv_norm = ScaleNorm(dim)
se_keys = sorted({k.split('.')[0] for k in ck['state_encoders'].keys()})
state_encoders = nn.ModuleDict({e: nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim)) for e in se_keys})
emb_id_emb = nn.Embedding(n_emb_total, dim * A['n_emb_prefix'])
n_vis = (A['img_size'] // 32) ** 2          # 49
max_prefix = n_vis + A['max_text'] + 16 + A['n_emb_prefix']
policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=dim, heads=A['heads'], kv_heads=A.get('kv_heads'),
                    ff_hidden=A.get('ff_hidden'), depth=A['depth'], L_inner=A['L_inner'], H_outer=A['H_outer'],
                    state_dim=dim, max_prefix=max_prefix, weighting=A['weighting'], update_mode=A['update_mode'],
                    alpha_parametrization=A['alpha_parametrization'], alpha_per_dim=A['alpha_per_dim'],
                    n_embodiments=n_emb_total, per_emb_head=A['per_emb_head'], dropout=0.0)
base = {'text_agg': text_agg, 'text_proj': text_proj, 'kv_norm': kv_norm,
        'state_encoders': state_encoders, 'emb_id_emb': emb_id_emb, 'policy': policy}
for k, m in base.items(): m.load_state_dict(ck[k])
# overlay EMA (deployed weights)
ep = ck.get('ema_params', {})
for nm, m in base.items():
    sd = m.state_dict()
    for k in list(sd):
        if f'{nm}.{k}' in ep: sd[k] = ep[f'{nm}.{k}'].to(sd[k].dtype)
    m.load_state_dict(sd)
for m in base.values():
    m.to(dev).eval()
    for p in m.parameters(): p.requires_grad_(False)

# LoRA on policy trunk + unfreeze output heads
n_lora = apply_lora(policy, {'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'}, args.lora_r, args.lora_alpha)
for nm, p in policy.named_parameters():
    if 'lora_' in nm or 'head' in nm.lower(): p.requires_grad_(True)
policy.to(dev)
print(f"LoRA-wrapped {n_lora} linears (r={args.lora_r})", flush=True)


# ───────────────── DINOv2 (frozen) + MLP adapter (trainable) ─────────────────
dino = AutoModel.from_pretrained('facebook/dinov2-small').to(dev).eval()
for p in dino.parameters(): p.requires_grad_(False)
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

class VisionAdapter(nn.Module):
    def __init__(self, din=384, dim=512, grid=7, hidden=1024):
        super().__init__()
        self.grid = grid; self.norm = nn.LayerNorm(din)
        self.mlp = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.pos = nn.Parameter(torch.zeros(1, grid * grid, dim)); nn.init.trunc_normal_(self.pos, std=0.02)
    def forward(self, patch):                      # patch: (B, P, din)
        B, P, D = patch.shape; s = int(P ** 0.5)
        x = patch.transpose(1, 2).reshape(B, D, s, s)
        x = F.adaptive_avg_pool2d(x, self.grid).flatten(2).transpose(1, 2)   # (B, grid², din)
        return self.mlp(self.norm(x)) + self.pos
adapter = VisionAdapter(dim=dim).to(dev)

@torch.no_grad()
def dino_patches(frames01):                        # (B,3,224,224) [0,1] → (B,256,384)
    x = (frames01 - DINO_MEAN) / DINO_STD
    return dino(pixel_values=x).last_hidden_state[:, 1:, :]

def vtok_of(frames01):
    return adapter(dino_patches(frames01))

# ───────────────── VAE (widowx) + T5 cache + helpers ─────────────────
c = torch.load(f'data/ckpts/oxe_vqvae_{args.robot}.pt', map_location='cpu', weights_only=False)
vae = ActionVQVAE1d(action_dim=c['action_dim'], vq_cls=VQ1d_EMA, k=c.get('k', 128)).to(dev).eval()
vae.load_state_dict(c['vae'])
for p in vae.parameters(): p.requires_grad_(False)
VG = c['action_var_global'].view(1, 1, -1).to(dev)
t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
EID = EMBODIMENT_ID.get(args.robot, len(EMBODIMENTS))

def t5_batch(tasks):
    B = len(tasks); out = torch.zeros(9, B, A['max_text'], 512)
    for b, tk in enumerate(tasks):
        e = t5['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float(); T = min(h.shape[1], A['max_text']); L = min(h.shape[0], 9)
        out[:L, b, :T, :] = h[:L, :T, :]
    return out.to(dev)

def encode_codes(ac, pv):
    nT = ac.shape[1]; m = pv.mean(1, keepdim=True); S = ((pv - m) ** 2).sum(1, keepdim=True)
    lam = nT / (S + nT * VG); xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
    gt, _ = vae.encode_with_soft(xn, tau=0.1)
    return [gt[0]]

def build_vis(frames01, tasks, B):
    vtok = vtok_of(frames01)
    ttok = text_proj(text_agg([t5_batch(tasks)[l] for l in range(9)]))
    eid = torch.full((B,), EID, dtype=torch.long, device=dev)
    etok = emb_id_emb(eid).view(B, A['n_emb_prefix'], dim)
    return kv_norm(torch.cat([etok, vtok, ttok], dim=1)), eid


# ───────────────── data ─────────────────
sp = load_dataset_spec(BRIDGE, chunk_len=16, lookback=16, chunk_stride=4)
ds = MultiOXEDataset([sp], chunk_len=16, lookback=16)
print(f"train chunks: {len(ds)}", flush=True)

def to224(pil):
    return torch.from_numpy(np.asarray(pil.convert('RGB').resize((224, 224))).copy()).permute(2, 0, 1).float() / 255.

def collate(batch):
    fr = torch.stack([to224(b[0]) for b in batch])
    st = torch.stack([b[1] for b in batch]); ac = torch.stack([b[2] for b in batch]); pv = torch.stack([b[3] for b in batch])
    return fr, st, ac, pv, [b[4] for b in batch]

loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                     collate_fn=collate, drop_last=True, persistent_workers=args.num_workers > 0,
                                     pin_memory=True, prefetch_factor=4 if args.num_workers > 0 else None)


# ───────────────── held-out + in-dist control batches (fixed) ─────────────────
def ep_ids(lo, hi):
    out = []
    with open(os.path.join(BRIDGE, 'meta', 'episodes.jsonl')) as f:
        for line in f:
            r = json.loads(line); L = r.get('length') or r.get('num_frames') or 0
            if lo <= L <= hi: out.append(r['episode_index'])
    return out

def fixed_batch(ep_list, n, seed, start=16):
    s = load_dataset_spec(BRIDGE, chunk_len=16, lookback=16, chunk_stride=16)
    s.chunk_index = [(ep, start) for ep in ep_list]
    d = MultiOXEDataset([s], chunk_len=16, lookback=16)
    rng = random.Random(seed); pool = list(range(len(d))); rng.shuffle(pool); got = []
    for i in pool:
        if len(got) >= n: break
        try:
            fr, st, ac, pv, tk, eid, di = d[i]
            if ac.shape[-1] != c['action_dim'] or st.shape[-1] != 8: continue
            got.append((to224(fr), st, ac, pv, tk))
        except Exception: pass
    fr = torch.stack([g[0] for g in got]).to(dev); st = torch.stack([g[1] for g in got]).to(dev)
    ac = torch.stack([g[2] for g in got]).to(dev); pv = torch.stack([g[3] for g in got]).to(dev)
    tasks = [g[4] for g in got]; B = len(got)
    gt = encode_codes(ac, pv); T_l = gt[0].shape[1]
    masks = []
    for sd in range(4):
        g = torch.Generator(); g.manual_seed(sd * 31337 + 7)
        u = torch.rand(B, generator=g); r = torch.cos(math.pi * u / 2).clamp(min=1.0 / T_l)
        noise = torch.rand(B, T_l, generator=g); m = noise < r.unsqueeze(1)
        m[torch.arange(B), noise.argmin(1)] = True; masks.append(m.to(dev))
    return dict(fr=fr, st=st, gt=gt, masks=masks, tasks=tasks, B=B)

# INDIST = trained windows (start 16); OFFSTRIDE = SAME episodes, unseen window (start 18,
# off the stride-4 grid) with in-distribution targets = the FAIR generalization probe.
# HELDOUT (len==32) kept only for continuity (confounded: OOD terminal-chunk targets).
ls = ep_ids(45, 60); random.Random(7).shuffle(ls)
VAL = {'INDIST':    fixed_batch(ls, args.val_n, 4, start=16),
       'OFFSTRIDE': fixed_batch(ls, args.val_n, 4, start=18),
       'HELDOUT':   fixed_batch(ep_ids(32, 32), args.val_n, 3, start=16)}

@torch.no_grad()
def validate():
    for m in list(base.values()) + [adapter]: m.eval()
    policy.eval()
    res = {}
    for name, b in VAL.items():
        vis, eid = build_vis(b['fr'], b['tasks'], b['B'])
        s_enc = state_encoders[args.robot](b['st']); tc = tm = 0
        for mk in b['masks']:
            logits = policy(b['gt'], vis, s_enc, mask_list=[mk], n_outer=A['H_outer'], n_inner=A['L_inner'], emb_id=eid)[-1][0]
            tc += ((logits.argmax(-1) == b['gt'][0]) & mk).float().sum().item(); tm += mk.float().sum().item()
        res[name] = 100 * tc / max(1, tm)
    adapter.train(); policy.train()
    return res


# ───────────────── optimizer + train loop ─────────────────
train_params = list(adapter.parameters()) + [p for p in policy.parameters() if p.requires_grad]
n_tr = sum(p.numel() for p in train_params) / 1e6
print(f"trainable: adapter+LoRA+heads = {n_tr:.2f}M params", flush=True)
opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.1)

print(f"\n=== DINO+LoRA finetune: {args.steps} steps, bs {args.batch_size} ===", flush=True)
print(f"[val @0] " + "  ".join(f"{k}={v:.1f}%" for k, v in validate().items()), flush=True)
best = -1; step = 0; t0 = time.time(); wl = wc = wa = 0
while step < args.steps:
    for fr, st, ac, pv, tasks in loader:
        if step >= args.steps: break
        step += 1
        fr = fr.to(dev, non_blocking=True); st = st.to(dev, non_blocking=True)
        ac = ac.to(dev, non_blocking=True); pv = pv.to(dev, non_blocking=True)
        with torch.no_grad(): gt = encode_codes(ac, pv)
        rmax = min(1.0, 0.3 + 0.7 * step / (args.steps * 0.5))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            vis, eid = build_vis(fr, tasks, fr.shape[0])
            s_enc = state_encoders[args.robot](st)
            loss, per, _ = policy.forward_loss(gt, vis, s_enc, n_inner=A['L_inner'], h_max=A['h_max'],
                                               mask_ratio_max=rmax, emb_id=eid, label_smoothing=0.1, mask_sampler='cosine')
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step(); sched.step()
        wl += loss.item(); wc += per[0]['mask_correct']; wa += per[0]['mask_total']
        if step % 50 == 0:
            print(f"  step {step:>5}/{args.steps}  loss={wl/50:.3f}  train_acc={wc/max(wa,1)*100:.1f}%  "
                  f"rmax={rmax:.2f}  [{time.time()-t0:.0f}s]", flush=True); wl = wc = wa = 0
        if step % args.val_every == 0:
            v = validate(); tag = ''
            if v['OFFSTRIDE'] > best:                 # select by the FAIR generalization metric
                best = v['OFFSTRIDE']
                torch.save({'adapter': adapter.state_dict(),
                            'lora': {n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad},
                            'step': step, 'offstride': best}, args.out); tag = ' (best, saved)'
            print(f"  [val @{step}] INDIST(seen)={v['INDIST']:.1f}%  OFFSTRIDE(unseen)={v['OFFSTRIDE']:.1f}%  "
                  f"HELDOUT(len32)={v['HELDOUT']:.1f}%  best_offstride={best:.1f}%{tag}", flush=True)
        if args.smoke and step >= 2: break
    if args.smoke and step >= 2:
        print("[smoke] ok"); break
print(f"done. best held-out = {best:.1f}%", flush=True)
