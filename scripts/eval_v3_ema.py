#!/usr/bin/env python3
"""Same eval as eval_v3_masks.py but loads EMA params instead of raw training weights.
Reports val_acc + val_loss for RAW and EMA side by side at all mask conditions.
"""
import os, sys, glob, random, math
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

CKPT = sys.argv[1] if len(sys.argv) > 1 else 'data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt'
ROBOT = 'widowx'
N_EVAL_SAMPLES = 64
MASK_SEEDS_PER_BATCH = 4


def build_modules(args, ck):
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
    return mods, keys, (args, L_I, H_O, MAX_TEXT, PFX, DIM, IMG)


def main():
    print(f"[eval-ema] ckpt: {CKPT}", flush=True)
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    args = ck['args']
    step = ck.get('step', '?')
    print(f"[eval-ema]   step at save: {step}")
    has_ema = 'ema_params' in ck and ck['ema_params']
    print(f"[eval-ema]   ema_params present: {has_ema}  (n={len(ck.get('ema_params', {}))})")
    if not has_ema:
        print("[eval-ema] no EMA params — abort"); return

    # data setup
    vae_c = torch.load(f'data/ckpts/oxe_vqvae_{ROBOT}.pt', map_location='cpu', weights_only=False)
    vae = ActionVQVAE1d(action_dim=vae_c['action_dim'], vq_cls=VQ1d_EMA, k=vae_c.get('k', 128))
    vae.load_state_dict(vae_c['vae']); vae.eval()
    var_global = vae_c['action_var_global'].view(1, 1, -1)
    adim = vae_c['action_dim']
    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)

    # build batch
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

    def run_eval(weight_source):
        """weight_source: 'raw' or 'ema'"""
        mods, keys, dims = build_modules(args, ck)
        args_, L_I, H_O, MAX_TEXT, PFX, DIM, IMG = dims
        # load weights
        if weight_source == 'raw':
            for k, m in zip(keys, mods): m.load_state_dict(ck[k])
        else:  # ema
            # ema_params: {full_param_name: tensor}
            # we need to inject these into the right module's state_dict.
            # Names in ema_params are like 'cnn.stem.weight', 'policy.g.blocks.0.....'.
            for nm, m in zip(keys, mods):
                sd = m.state_dict()
                for pname in list(sd.keys()):
                    full = f"{nm}.{pname}"
                    if full in ck['ema_params']:
                        sd[pname] = ck['ema_params'][full].to(sd[pname].dtype)
                m.load_state_dict(sd, strict=False)
        for m in mods: m.eval()
        cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy = mods

        # build T5/img
        t5s = torch.zeros(9, Bf, MAX_TEXT, 512)
        for b, tk in enumerate(tasks):
            e = t5['embeddings'].get(tk)
            if e is None: continue
            h = e['hidden'].float()
            t = min(h.shape[1], MAX_TEXT); L = min(h.shape[0], 9)
            t5s[:L, b, :t, :] = h[:L, :t, :]
        img_n = normalize_image(frames, img_var['var_global'])

        with torch.no_grad():
            vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
            tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
            etok = emb_id_emb(eid_t).view(Bf, PFX, DIM)
            vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
            s_enc = state_encoders[ROBOT](states)

        T_l = gt[0].shape[1]
        results = {}

        def run_with_masks(masks_list, label):
            total_correct = 0; total_masked = 0; total_loss = 0.0; n = 0
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
            return total_correct / max(1, total_masked), total_loss / max(1, n)

        # cosine
        fixed = []
        for seed in range(MASK_SEEDS_PER_BATCH):
            g = torch.Generator(); g.manual_seed(seed * 31337 + 7)
            u = torch.rand(Bf, generator=g)
            r = torch.cos(math.pi * u / 2).clamp(min=1.0 / T_l)
            noise = torch.rand(Bf, T_l, generator=g)
            m = noise < r.unsqueeze(1)
            m[torch.arange(Bf), noise.argmin(1)] = True
            fixed.append(m)
        results['cosine'] = run_with_masks(fixed, 'cosine')

        # fixed-ratio
        for k in range(1, T_l + 1):
            masks_list = []
            for seed in range(MASK_SEEDS_PER_BATCH):
                g = torch.Generator(); g.manual_seed(seed * 31337 + 7 + k * 101)
                m = torch.zeros(Bf, T_l, dtype=torch.bool)
                for b in range(Bf):
                    perm = torch.randperm(T_l, generator=g)
                    m[b, perm[:k]] = True
                masks_list.append(m)
            results[f'mask_{k}_of_{T_l}'] = run_with_masks(masks_list, f'mask {k}/{T_l}')

        return results

    print(f"\n[eval-ema] running RAW...")
    raw = run_eval('raw')
    print(f"[eval-ema] running EMA...")
    ema = run_eval('ema')

    print(f"\n{'='*80}")
    print(f"RAW vs EMA comparison (step {step}, {Bf} held-out widowx samples)")
    print(f"{'='*80}")
    print(f"  {'condition':<22s} {'RAW val_acc':>12s} {'EMA val_acc':>12s} {'Δ':>8s} | "
          f"{'RAW loss':>10s} {'EMA loss':>10s}")
    for label in raw:
        r_acc, r_loss = raw[label]
        e_acc, e_loss = ema[label]
        d = (e_acc - r_acc) * 100
        print(f"  {label:<22s} {r_acc*100:>11.2f}% {e_acc*100:>11.2f}% {d:>+7.2f}pp | "
              f"{r_loss:>10.4f} {e_loss:>10.4f}")


if __name__ == '__main__':
    main()
