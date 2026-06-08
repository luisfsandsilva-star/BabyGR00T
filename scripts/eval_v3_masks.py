#!/usr/bin/env python3
"""Harder-mask eval for v3 best ckpt.

Loads the resumed_best (latest val_acc) ckpt and runs masked-CE accuracy on
64 fixed held-out widowx samples at multiple mask conditions:

  1. Cosine sampler   (matches current sentinel — average over 4 fixed seeds)
  2. Mask 1/4 (25%)   (easy: one masked code)
  3. Mask 2/4 (50%)
  4. Mask 3/4 (75%)
  5. Mask 4/4 (100%, FULL)  (hardest: all codes masked, true uncond gen)

Reports val_acc per condition. Full-mask is the regime closest to inference
when the policy must predict all 4 action codes from zero prior context.
"""
import os, sys, time, glob, random, math
os.environ['CUDA_VISIBLE_DEVICES'] = ''
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm,
                             ActionVQVAE1d, VQ1d_EMA)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image

torch.set_num_threads(2); random.seed(42); torch.manual_seed(42); np.random.seed(42)

CKPT = 'data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt'
# fallback to the original best if resumed best doesn't exist yet
if not os.path.exists(CKPT):
    CKPT = 'data/ckpts/oxe_policy_v14_widowx_v3_best.pt'
ROBOT = 'widowx'
N_EVAL_SAMPLES = 64
MASK_SEEDS_PER_BATCH = 4   # for cosine sampler (4 seeds → average)


def main():
    print(f"[eval] loading ckpt: {CKPT}", flush=True)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    args = ck['args']
    step = ck.get('step', '?')
    print(f"[eval]   step at save: {step}")

    # ── per-emb VAE (widowx) ──
    vae_c = torch.load(f'data/ckpts/oxe_vqvae_{ROBOT}.pt', map_location='cpu', weights_only=False)
    vae = ActionVQVAE1d(action_dim=vae_c['action_dim'], vq_cls=VQ1d_EMA, k=vae_c.get('k', 128))
    vae.load_state_dict(vae_c['vae']); vae.eval()
    var_global = vae_c['action_var_global'].view(1, 1, -1)
    adim = vae_c['action_dim']

    # ── caches ──
    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)

    # ── modules ──
    DIM = args['dim']; HEADS = args['heads']; KV = args.get('kv_heads', HEADS)
    FF = args.get('ff_hidden'); DEPTH = args['depth']
    L_I = args['L_inner']; H_O = args['H_outer']
    N_EMB = len(EMBODIMENTS) + 1
    IMG = args['img_size']; MAX_TEXT = args['max_text']; PFX = args['n_emb_prefix']
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=args['cnn_expand'],
                       out_dim=args['cnn_out_dim'], norm=args['cnn_norm'], pos_emb=args['cnn_pe'],
                       img_size=IMG, dropout=0.0,
                       n_embodiments=N_EMB if args.get('cnn_film_by_emb') else 0)
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
    cnn_proj = nn.Linear(args['cnn_out_dim'], DIM)
    text_proj = nn.Linear(512, DIM)
    kv_norm = ScaleNorm(DIM)
    se_keys = sorted({k.split('.')[0] for k in ck['state_encoders'].keys()})
    state_encoders = nn.ModuleDict({e: nn.Sequential(nn.Linear(8, DIM), nn.GELU(), nn.Linear(DIM, DIM))
                                     for e in se_keys})
    emb_id_emb = nn.Embedding(N_EMB, DIM * PFX)
    n_vis = (IMG // 32) ** 2
    max_prefix = n_vis + MAX_TEXT + 16 + PFX
    policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=DIM, heads=HEADS,
                        kv_heads=KV, ff_hidden=FF, depth=DEPTH, L_inner=L_I, H_outer=H_O,
                        state_dim=DIM, max_prefix=max_prefix,
                        weighting=args['weighting'], update_mode=args['update_mode'],
                        alpha_parametrization=args['alpha_parametrization'],
                        alpha_per_dim=args['alpha_per_dim'],
                        n_embodiments=N_EMB, per_emb_head=args['per_emb_head'], dropout=0.0,
                        g_input_noise=0.0)
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']
    for k, m in zip(keys, mods): m.load_state_dict(ck[k])
    for m in mods: m.eval()

    # ── data: same sampling rule as sentinel (seed=98765) ──
    print(f"[eval] building held-out batch of {N_EVAL_SAMPLES} widowx samples...")
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=16)
            if sp.chunk_index and sp.robot == ROBOT: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    rng = random.Random(98765)
    pool = list(range(len(ds))); rng.shuffle(pool)
    samples = []
    for idx in pool:
        if len(samples) >= N_EVAL_SAMPLES: break
        try:
            fr, st, ac, pv, tk, eid_, di = ds[idx]
            if ac.shape[-1] != adim or st.shape[-1] != 8: continue
            from PIL import Image
            pil = fr.convert('RGB').resize((args['img_size'], args['img_size']))
            ft = torch.from_numpy(np.asarray(pil).copy()).permute(2,0,1).float() / 255.
            samples.append((ft, st, ac, pv, tk))
        except Exception: pass
    Bf = len(samples)
    frames = torch.stack([s[0] for s in samples])
    states = torch.stack([s[1] for s in samples])
    actions = torch.stack([s[2] for s in samples])
    prevs = torch.stack([s[3] for s in samples])
    tasks = [s[4] for s in samples]
    print(f"[eval]   collected {Bf} samples")

    # build cached pieces
    t5s = torch.zeros(9, Bf, MAX_TEXT, 512)
    for b, tk in enumerate(tasks):
        e = t5['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float()
        t = min(h.shape[1], MAX_TEXT); L = min(h.shape[0], 9)
        t5s[:L, b, :t, :] = h[:L, :t, :]
    img_n = normalize_image(frames, img_var['var_global'])
    nT = actions.shape[1]
    m_pv = prevs.mean(dim=1, keepdim=True)
    S = ((prevs - m_pv) ** 2).sum(dim=1, keepdim=True)
    lam = nT / (S + nT * var_global)
    xn = ((actions - m_pv) * lam.sqrt()).transpose(1, 2)
    with torch.no_grad():
        cd, _ = vae.encode_with_soft(xn, tau=0.1)
    gt = [cd[0]]
    target = gt[0]
    eid_t = torch.tensor([EMBODIMENT_ID.get(ROBOT, len(EMBODIMENTS))] * Bf, dtype=torch.long)

    # context computation (cached for all eval conditions)
    with torch.no_grad():
        vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
        tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
        etok = emb_id_emb(eid_t).view(Bf, PFX, DIM)
        vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
        s_enc = state_encoders[ROBOT](states)

    # ── eval conditions ──
    T_l = gt[0].shape[1]  # = 4 for widowx
    print(f"\n[eval] running mask conditions (T_l={T_l}, B={Bf})...\n")

    def run_with_masks(masks_list, label):
        """Run policy with a list of mask tensors (each shape [B, T_l])."""
        total_correct = 0; total_masked = 0; total_loss = 0.0; n = 0
        all_logits_list = []
        for masks in masks_list:
            with torch.no_grad():
                all_logits = policy(gt, vis, s_enc, mask_list=[masks],
                                    n_outer=H_O, n_inner=L_I, emb_id=eid_t)
            logits = all_logits[-1][0]
            preds = logits.argmax(-1)
            correct = ((preds == target) & masks).float().sum().item()
            masked = masks.float().sum().item()
            total_correct += correct; total_masked += masked
            lp = F.log_softmax(logits, dim=-1)
            ce = -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            n_m = masks.float().sum(1).clamp(min=1)
            total_loss += ((ce * masks.float()).sum(1) / n_m).mean().item()
            n += 1
        acc = total_correct / max(1, total_masked)
        loss = total_loss / max(1, n)
        print(f"  {label:<30s} masked_acc = {acc*100:6.2f}%  masked_loss = {loss:.4f}  ({int(total_masked)} total masked positions)")
        return acc, loss

    results = {}

    # 1. cosine sampler (matches sentinel) — 4 fixed seeds
    fixed = []
    for seed in range(MASK_SEEDS_PER_BATCH):
        g = torch.Generator(); g.manual_seed(seed * 31337 + 7)
        u = torch.rand(Bf, generator=g)
        r = torch.cos(math.pi * u / 2).clamp(min=1.0 / T_l)
        noise = torch.rand(Bf, T_l, generator=g)
        m = noise < r.unsqueeze(1)
        m[torch.arange(Bf), noise.argmin(1)] = True
        fixed.append(m)
    results['cosine'] = run_with_masks(fixed, '1) cosine sampler (sentinel)')

    # 2-5. fixed-ratio: exactly k of T_l positions masked, sampled at random
    for k in range(1, T_l + 1):
        masks_list = []
        for seed in range(MASK_SEEDS_PER_BATCH):
            g = torch.Generator(); g.manual_seed(seed * 31337 + 7 + k * 101)
            # for each row, pick k random positions
            m = torch.zeros(Bf, T_l, dtype=torch.bool)
            for b in range(Bf):
                # use generator-based perm
                perm = torch.randperm(T_l, generator=g)
                m[b, perm[:k]] = True
            masks_list.append(m)
        results[f'mask_{k}_of_{T_l}'] = run_with_masks(masks_list, f'{k+1}) mask {k}/{T_l} ({100*k//T_l}%)')

    # ── summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY  (v3 best ckpt @ step {step}, {Bf} held-out widowx samples)")
    print(f"{'='*70}")
    print(f"  {'condition':<30s} {'val_acc':>10s} {'val_loss':>10s}")
    for label, (acc, loss) in results.items():
        print(f"  {label:<30s} {acc*100:>9.2f}% {loss:>10.4f}")
    rate_full = results.get(f'mask_{T_l}_of_{T_l}', (0,0))[0]
    rate_cos = results['cosine'][0]
    print(f"\n  full-mask vs cosine gap: {(rate_cos - rate_full)*100:+.2f}pp")
    print(f"  (cosine is what we've been tracking; full-mask is the inference regime)")


if __name__ == '__main__':
    main()
