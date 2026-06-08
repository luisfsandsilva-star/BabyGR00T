#!/usr/bin/env python3
"""Live early-stop sentinel for in-progress policy training (ACC-based).

Sits beside the training process. Periodically (every CHECK_EVERY seconds):
  1. Reads the latest disk ckpt
  2. Runs a small held-out eval: same masked-CE-acc the model trains on
     (using a FIXED set of (sample, mask) pairs so signal is deterministic)
  3. Tracks best-val ACC (higher = better)
  4. Saves the best ckpt to a separate _best.pt path
  5. If best-val ACC hasn't improved by ABS_TOL over PATIENCE checks, sends
     SIGTERM to training PID (atexit handler will save a final ckpt)

Why ACC and not MSE?
  - ACC = CE accuracy on masked codes = exactly what the model trains on
  - MSE is downstream (depends on VAE-codebook geometry); model gradient flows
    through CE on code IDs, not through decoded action MSE.

T5 cache loaded ONCE at startup; each check loads only the latest policy ckpt.
Eval is fully deterministic across calls (fixed sample indices + fixed mask
seeds), so any ACC change reflects only policy weight changes.

Args:
  --ckpt-path / --train-pid / --robot
  --check-every (default 600s) / --patience (default 5) / --abs-tol (default 0.005)
"""
import os, sys, time, glob, random, argparse, signal
os.environ['CUDA_VISIBLE_DEVICES'] = ''
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import STRMPolicy, LayerAggregator, ScaleNorm, ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image
import torch.nn.functional as F
import math as _math

torch.set_num_threads(2); random.seed(42); torch.manual_seed(42); np.random.seed(42)
N_EVAL_SAMPLES = 64        # bigger batch → tighter ACC variance
MASK_SEEDS_PER_BATCH = 4   # average over 4 mask draws for variance reduction


def build_eval_context(ckpt_path, robot):
    print(f"[sentinel] loading T5 cache + per-emb VAE for {robot}...", flush=True)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ck['args']

    vae_path = f'data/ckpts/oxe_vqvae_{robot}.pt'
    c = torch.load(vae_path, map_location='cpu', weights_only=False)
    vae = ActionVQVAE1d(action_dim=c['action_dim'], vq_cls=VQ1d_EMA, k=c.get('k', 128))
    vae.load_state_dict(c['vae']); vae.eval()
    var_global = c['action_var_global'].view(1, 1, -1)
    adim = c['action_dim']

    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)

    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=16)
            if sp.chunk_index and sp.robot == robot: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    rng = random.Random(98765)  # different from any train seed
    pool = list(range(len(ds))); rng.shuffle(pool)
    samples = []
    for idx in pool:
        if len(samples) >= N_EVAL_SAMPLES: break
        try:
            fr, st, ac, pv, tk, eid, di = ds[idx]
            if ac.shape[-1] != adim or st.shape[-1] != 8: continue
            from PIL import Image
            pil = fr.convert('RGB').resize((args['img_size'], args['img_size']))
            frame_t = torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).float() / 255.
            samples.append((frame_t, st, ac, pv, tk))
        except Exception: pass
    print(f"[sentinel] collected {len(samples)} eval samples", flush=True)

    frames = torch.stack([s[0] for s in samples])
    states = torch.stack([s[1] for s in samples])
    actions = torch.stack([s[2] for s in samples])
    prevs = torch.stack([s[3] for s in samples])
    tasks = [s[4] for s in samples]
    Bf = len(samples)

    T_text = args['max_text']
    t5s = torch.zeros(9, Bf, T_text, 512)
    for b, tk in enumerate(tasks):
        e = t5['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float()
        t = min(h.shape[1], T_text); L = min(h.shape[0], 9)
        t5s[:L, b, :t, :] = h[:L, :t, :]

    img_n = normalize_image(frames, img_var['var_global'])

    nT = actions.shape[1]
    m_pv = prevs.mean(dim=1, keepdim=True)
    S = ((prevs - m_pv) ** 2).sum(dim=1, keepdim=True)
    lam = nT / (S + nT * var_global)
    xn = ((actions - m_pv) * lam.sqrt()).transpose(1, 2)
    with torch.no_grad():
        cd, _ = vae.encode_with_soft(xn, tau=0.1)
    gt_codes = [cd[0]]

    T_l = gt_codes[0].shape[1]
    fixed_masks = []
    for seed in range(MASK_SEEDS_PER_BATCH):
        g = torch.Generator(); g.manual_seed(seed * 31337 + 7)
        u = torch.rand(Bf, generator=g)
        r = torch.cos(_math.pi * u / 2).clamp(min=1.0 / T_l)
        noise = torch.rand(Bf, T_l, generator=g)
        m = noise < r.unsqueeze(1)
        m[torch.arange(Bf), noise.argmin(1)] = True
        fixed_masks.append(m)

    eid_t = torch.tensor([EMBODIMENT_ID.get(robot, len(EMBODIMENTS))] * Bf, dtype=torch.long)

    return dict(args=args, robot=robot, img_n=img_n, t5s=t5s, states=states,
                gt_codes=gt_codes, fixed_masks=fixed_masks, eid_t=eid_t, Bf=Bf)


def build_policy_from_ckpt(ck, ctx):
    args = ctx['args']
    n_emb_total = len(EMBODIMENTS) + 1
    dim = args['dim']
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=args['cnn_expand'],
                       out_dim=args['cnn_out_dim'], norm=args['cnn_norm'], pos_emb=args['cnn_pe'],
                       img_size=args['img_size'], dropout=0.0,
                       n_embodiments=n_emb_total if args.get('cnn_film_by_emb') else 0)
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
    cnn_proj = nn.Linear(args['cnn_out_dim'], dim)
    text_proj = nn.Linear(512, dim)
    kv_norm = ScaleNorm(dim)
    state_encoders_keys = sorted({k.split('.')[0] for k in ck['state_encoders'].keys()})
    state_encoders = nn.ModuleDict({
        e: nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
        for e in state_encoders_keys})
    emb_id_emb = nn.Embedding(n_emb_total, dim * args['n_emb_prefix'])
    n_vis = (args['img_size'] // 32) ** 2
    max_prefix = n_vis + args['max_text'] + 16 + args['n_emb_prefix']
    policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=dim, heads=args['heads'],
                        kv_heads=args.get('kv_heads'), ff_hidden=args.get('ff_hidden'),
                        depth=args['depth'], L_inner=args['L_inner'], H_outer=args['H_outer'],
                        state_dim=dim, max_prefix=max_prefix,
                        weighting=args['weighting'],
                        update_mode=args['update_mode'],
                        alpha_parametrization=args['alpha_parametrization'],
                        alpha_per_dim=args['alpha_per_dim'],
                        n_embodiments=n_emb_total,
                        per_emb_head=args['per_emb_head'], dropout=0.0)
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']
    for k, m in zip(keys, mods): m.load_state_dict(ck[k])
    for m in mods: m.eval()
    return mods


@torch.no_grad()
def run_eval(ctx, mods):
    cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy = mods
    args = ctx['args']; dim = args['dim']
    eid = ctx['eid_t']; Bf = ctx['Bf']; robot = ctx['robot']

    vtok, _ = cnn(ctx['img_n'], emb_id=eid if args.get('cnn_film_by_emb') else None)
    vtok = cnn_proj(vtok)
    tagg = text_agg([ctx['t5s'][l] for l in range(9)])
    ttok = text_proj(tagg)
    etok = emb_id_emb(eid).view(Bf, args['n_emb_prefix'], dim)
    vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
    s_enc = state_encoders[robot](ctx['states'])

    total_correct = 0; total_masked = 0; total_loss = 0.0; n_calls = 0
    for masks_l in ctx['fixed_masks']:
        masks = [masks_l]
        all_logits = policy(ctx['gt_codes'], vis, s_enc, mask_list=masks,
                            n_outer=args['H_outer'], n_inner=args['L_inner'], emb_id=eid)
        logits = all_logits[-1][0]
        target = ctx['gt_codes'][0]
        preds = logits.argmax(-1)
        correct = ((preds == target) & masks_l).float().sum().item()
        masked = masks_l.float().sum().item()
        total_correct += correct; total_masked += masked
        lp = F.log_softmax(logits, dim=-1)
        ce = -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        n_m = masks_l.float().sum(1).clamp(min=1)
        total_loss += ((ce * masks_l.float()).sum(1) / n_m).mean().item()
        n_calls += 1
    return total_correct / max(1, total_masked), total_loss / max(1, n_calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-path', required=True)
    ap.add_argument('--train-pid', type=int, required=True)
    ap.add_argument('--robot', required=True)
    ap.add_argument('--check-every', type=int, default=600)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--abs-tol', type=float, default=0.005)
    ap.add_argument('--best-ckpt', default=None)
    args = ap.parse_args()
    if args.best_ckpt is None:
        args.best_ckpt = args.ckpt_path.replace('.pt', '_best.pt')

    print(f"[sentinel] watching ckpt: {args.ckpt_path}")
    print(f"[sentinel]     train PID: {args.train_pid}")
    print(f"[sentinel]     robot:     {args.robot}")
    print(f"[sentinel]     check every {args.check_every}s, patience {args.patience}, abs_tol {args.abs_tol}")
    print(f"[sentinel]     METRIC: masked-CE accuracy (higher = better)")
    print(f"[sentinel]     best ckpt → {args.best_ckpt}\n", flush=True)

    ctx = build_eval_context(args.ckpt_path, args.robot)
    best_acc = -1.0; best_step = None; no_improve = 0
    last_mtime = 0
    while True:
        try: os.kill(args.train_pid, 0)
        except ProcessLookupError:
            print(f"[sentinel] training PID {args.train_pid} exited — sentinel done.", flush=True)
            return
        try: mtime = os.path.getmtime(args.ckpt_path)
        except FileNotFoundError: mtime = 0
        if mtime == last_mtime:
            time.sleep(args.check_every); continue
        last_mtime = mtime

        try:
            ck = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"[sentinel] couldn't load ckpt ({type(e).__name__}: {e}) — retry next round", flush=True)
            time.sleep(args.check_every); continue
        step = ck.get('step', '?')
        t0 = time.time()
        mods = build_policy_from_ckpt(ck, ctx)
        acc, val_loss = run_eval(ctx, mods)
        elapsed = time.time() - t0

        improved = acc > best_acc + args.abs_tol
        if improved:
            prev = best_acc
            best_acc = acc; best_step = step; no_improve = 0
            torch.save(ck, args.best_ckpt)
            print(f"[sentinel] step {step}: val_acc={acc*100:.2f}% val_loss={val_loss:.4f} "
                  f"← NEW BEST (was {prev*100:.2f}%)  saved [{elapsed:.0f}s]", flush=True)
        else:
            no_improve += 1
            print(f"[sentinel] step {step}: val_acc={acc*100:.2f}% val_loss={val_loss:.4f} "
                  f"(best={best_acc*100:.2f}% @ step {best_step}) [{no_improve}/{args.patience}] [{elapsed:.0f}s]",
                  flush=True)
            if no_improve >= args.patience:
                print(f"\n[sentinel] EARLY STOP: val_acc hasn't improved in {args.patience} ckpts. "
                      f"Best={best_acc*100:.2f}% @ step {best_step}.\n"
                      f"[sentinel] SIGTERM → PID {args.train_pid}...", flush=True)
                try:
                    os.kill(args.train_pid, signal.SIGTERM)
                    print(f"[sentinel] SIGTERM sent.", flush=True)
                except ProcessLookupError:
                    print(f"[sentinel] training already gone.", flush=True)
                return

        del mods
        import gc; gc.collect()
        time.sleep(args.check_every)


if __name__ == '__main__':
    main()
