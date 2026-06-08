#!/usr/bin/env python3
"""Find max safe batch size + optimal LR + g_input_noise for v14_widowx_v2.

PHASE 1 — BS SCAN
  Builds the full pipeline at progressively larger BS, runs 5 fwd+bwd steps,
  records peak GPU memory. Returns largest BS that stays under MEM_BUDGET.
  Uses AMP bf16 (matches training).

PHASE 2 — LR RANGE TEST (Smith 2015, "Cyclical Learning Rates")
  Uses the REAL training dataloader (fresh batch each step, same workers /
  prefetch / augmentation as training). Exponentially ramps lr from LR_LO to
  LR_HI over N_LR_STEPS. Pick lr ≈ (lr_at_min_loss) / 4 (Smith's rule).
  Exponential ramp lr(t) = lr_lo·(lr_hi/lr_lo)^(t/N) — better for LARS than linear.

PHASE 3 — g_input_noise GRID
  Bishop-1995 Tikhonov: σ noise added per g-call. Sweep {0, 0.01, 0.03, 0.10}
  for ~500 steps each at chosen LR+BS with real dataloader. Pick σ with lowest
  smoothed train loss at the END of the window.

Output: recommended (BS, LR, σ_noise). User signs off → full launch.
"""
import os, sys, time, json, math, argparse, gc
# Reduce fragmentation between phases (recommended by CUDA OOM messages)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm,
                             ActionVQVAE1d, VQ1d_EMA, MuSGD_LARS)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image
import glob, random

# ── target architecture (v3: ~12M policy, scaled down from 168M) ───
ROBOT = 'widowx'
DIM = 512; DEPTH = 2; HEADS = 8; KV_HEADS = 2; FF_HIDDEN = 2048   # ratio 4 (Llama-small)
L_INNER = 5; H_OUTER = 3; H_MAX = 3
N_EMB_PREFIX = 16; MAX_TEXT = 64; IMG_SIZE = 144
CNN_OUT_DIM = 192; CNN_EXPAND = 4; CNN_NORM = 'gn'; CNN_PE = True
DROPOUT = 0.2; CNN_DROPOUT = 0.2; LABEL_SMOOTHING = 0.10
N_EMB_TOTAL = len(EMBODIMENTS) + 1   # 11

# ── budgets ─────────────────────────────────────────────────────────
MEM_BUDGET_FRAC = 0.85   # leave 15% slack for activation spikes
BS_CANDIDATES = [256, 512, 1024, 2048, 4096]   # smaller model → much larger BS fits
N_WARM_STEPS = 5         # for BS scan
N_LR_STEPS = 200         # for LR range test
LR_LO = 1e-5
LR_HI = 3e-1
NOISE_GRID = [0.0, 0.01, 0.03, 0.10]   # σ for Bishop input-noise
N_NOISE_STEPS = 500                     # steps per noise candidate
USE_AMP = True                          # bf16 autocast


def build_pipeline(per_emb_head=False, g_input_noise=0.0):
    """Build all modules + optimizer fresh."""
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=CNN_EXPAND,
                       out_dim=CNN_OUT_DIM, norm=CNN_NORM, pos_emb=CNN_PE,
                       img_size=IMG_SIZE, dropout=CNN_DROPOUT, n_embodiments=0)
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9)
    cnn_proj = nn.Linear(CNN_OUT_DIM, DIM)
    text_proj = nn.Linear(512, DIM)
    kv_norm = ScaleNorm(DIM)
    state_encoders = nn.ModuleDict({ROBOT: nn.Sequential(
        nn.Linear(8, DIM), nn.GELU(), nn.Linear(DIM, DIM))})
    emb_id_emb = nn.Embedding(N_EMB_TOTAL, DIM * N_EMB_PREFIX)
    n_vis = (IMG_SIZE // 32) ** 2
    max_prefix = n_vis + MAX_TEXT + 16 + N_EMB_PREFIX
    policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=DIM, heads=HEADS,
                        kv_heads=KV_HEADS, ff_hidden=FF_HIDDEN,
                        depth=DEPTH, L_inner=L_INNER, H_outer=H_OUTER,
                        state_dim=DIM, max_prefix=max_prefix,
                        weighting='clamp_direct', update_mode='damped',
                        alpha_parametrization='sigmoid', alpha_per_dim=True,
                        n_embodiments=N_EMB_TOTAL,
                        per_emb_head=per_emb_head, dropout=DROPOUT,
                        g_input_noise=g_input_noise)
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders,
            emb_id_emb, policy]
    for m in mods: m.train().cuda()
    return mods


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _amp_ctx():
    return torch.autocast('cuda', dtype=torch.bfloat16) if USE_AMP else _NullCtx()


# ── REAL dataloader helpers (fresh batch each step, matches training) ──
NW_DEFAULT = 24
PF_DEFAULT = 4
_AUG_KW = dict(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4),
               blur_sigma=(0.0, 1.5), crop_keep=(0.70, 1.0))
def _frame_to_tensor(pil):
    import random as _r
    pil = pil.convert('RGB')
    from babygroot_strm import augment
    pil = augment._apply_visual_params(pil, augment._sample_visual_params(_r, **_AUG_KW))
    pil = pil.resize((IMG_SIZE, IMG_SIZE))
    return torch.from_numpy(np.asarray(pil).copy()).permute(2, 0, 1).float() / 255.

def _collate(batch):
    frames = torch.stack([_frame_to_tensor(b[0]) for b in batch])
    states = torch.stack([b[1] for b in batch])
    actions = torch.stack([b[2] for b in batch])
    prevs = torch.stack([b[3] for b in batch])
    tasks = [b[4] for b in batch]
    return frames, states, actions, prevs, tasks

def _build_real_loader(ds, bs, nw=NW_DEFAULT, pf=PF_DEFAULT):
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=nw,
                      pin_memory=True, persistent_workers=True, prefetch_factor=pf,
                      collate_fn=_collate, drop_last=True)


def _iter_forever(loader):
    """Wrap dataloader to restart on epoch exhaustion. Otherwise the for loop
    exits after one epoch (94 batches at 48k chunks / BS=512) and we silently
    short-circuit to far fewer steps than requested."""
    while True:
        for batch in loader: yield batch


def _one_train_step(mods, opt, frames, states, actions, prevs, tasks,
                     ctx_static, vae, var_global, eid_pre):
    """One forward+backward+opt step, returns loss (or None on NaN)."""
    cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy = mods
    Bf = frames.shape[0]
    eid = eid_pre[:Bf]
    frames = frames.cuda(non_blocking=True); states = states.cuda(non_blocking=True)
    actions = actions.cuda(non_blocking=True); prevs = prevs.cuda(non_blocking=True)
    # T5 hiddens
    t5s = torch.zeros(9, Bf, MAX_TEXT, 512, device='cuda')
    for b, tk in enumerate(tasks):
        e = ctx_static['t5']['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float().cuda()
        t = min(h.shape[1], MAX_TEXT); L = min(h.shape[0], 9)
        t5s[:L, b, :t, :] = h[:L, :t, :]
    # GT codes
    m_pv = prevs.mean(dim=1, keepdim=True)
    S = ((prevs - m_pv) ** 2).sum(dim=1, keepdim=True)
    lam = actions.shape[1] / (S + actions.shape[1] * var_global.cuda())
    xn = ((actions - m_pv) * lam.sqrt()).transpose(1, 2)
    with torch.no_grad():
        cd, _ = vae.encode_with_soft(xn, tau=0.1)
    gt = [cd[0]]
    # forward + backward
    from babygroot_strm.perimg_norm import normalize_image
    img_n = normalize_image(frames, ctx_static['img_var']['var_global'].cuda())
    opt.zero_grad(set_to_none=True)
    with _amp_ctx():
        vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
        tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
        etok = emb_id_emb(eid).view(Bf, N_EMB_PREFIX, DIM)
        vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
        s_enc = state_encoders[ROBOT](states)
        loss, per, _ = policy.forward_loss(gt, vis, s_enc,
            n_inner=L_INNER, n_outer=H_OUTER, h_max=H_MAX,
            mask_ratio_max=1.0, emb_id=eid,
            label_smoothing=0.10, mask_sampler='cosine')
    lv = loss.item()
    if not math.isfinite(lv): return None
    loss.backward(); opt.step()
    return lv


def build_one_batch(B, dataset, ctx_static, vae, var_global, adim, n_steps_data=None):
    """Construct a single (frames, states, actions, prevs, tasks) batch from disk."""
    rng = random.Random(1234)
    pool = list(range(len(dataset))); rng.shuffle(pool)
    samples = []
    for idx in pool:
        if len(samples) >= B: break
        try:
            fr, st, ac, pv, tk, eid, di = dataset[idx]
            if ac.shape[-1] != adim or st.shape[-1] != 8: continue
            from PIL import Image
            pil = fr.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            ft = torch.from_numpy(np.asarray(pil).copy()).permute(2,0,1).float() / 255.
            samples.append((ft, st, ac, pv, tk))
        except Exception: pass
    frames = torch.stack([s[0] for s in samples]).cuda()
    states = torch.stack([s[1] for s in samples]).cuda()
    actions = torch.stack([s[2] for s in samples]).cuda()
    prevs = torch.stack([s[3] for s in samples]).cuda()
    tasks = [s[4] for s in samples]
    Bf = len(samples)
    t5s = torch.zeros(9, Bf, MAX_TEXT, 512).cuda()
    for b, tk in enumerate(tasks):
        e = ctx_static['t5']['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float().cuda()
        t = min(h.shape[1], MAX_TEXT); L = min(h.shape[0], 9)
        t5s[:L, b, :t, :] = h[:L, :t, :]
    img_n = normalize_image(frames, ctx_static['img_var']['var_global'].cuda())
    # gt codes
    nT = actions.shape[1]
    m_pv = prevs.mean(dim=1, keepdim=True)
    S = ((prevs - m_pv) ** 2).sum(dim=1, keepdim=True)
    lam = nT / (S + nT * var_global.cuda())
    xn = ((actions - m_pv) * lam.sqrt()).transpose(1, 2)
    with torch.no_grad():
        cd, _ = vae.encode_with_soft(xn, tau=0.1)
    gt_codes = [cd[0]]
    eid_t = torch.tensor([EMBODIMENT_ID.get(ROBOT, len(EMBODIMENTS))] * Bf, dtype=torch.long).cuda()
    return img_n, t5s, states, gt_codes, eid_t, Bf


def fwd_loss(mods, img_n, t5s, states, gt_codes, eid_t, Bf):
    cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy = mods
    with _amp_ctx():
        vtok, _ = cnn(img_n)
        vtok = cnn_proj(vtok)
        tagg = text_agg([t5s[l] for l in range(9)])
        ttok = text_proj(tagg)
        etok = emb_id_emb(eid_t).view(Bf, N_EMB_PREFIX, DIM)
        vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
        s_enc = state_encoders[ROBOT](states)
        loss, per, _ = policy.forward_loss(gt_codes, vis, s_enc, n_inner=L_INNER, h_max=H_MAX,
                                            n_outer=H_OUTER, mask_ratio_max=1.0, emb_id=eid_t,
                                            label_smoothing=LABEL_SMOOTHING,
                                            mask_sampler='cosine')
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-bs', action='store_true', help='skip BS scan; use --bs')
    ap.add_argument('--bs', type=int, default=None, help='force BS for LR test')
    ap.add_argument('--skip-lr', action='store_true', help='skip LR range test')
    args = ap.parse_args()

    print(f"[finder] loading data context...", flush=True)
    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)
    ctx_static = dict(t5=t5, img_var=img_var)
    c = torch.load(f'data/ckpts/oxe_vqvae_{ROBOT}.pt', map_location='cpu', weights_only=False)
    vae = ActionVQVAE1d(action_dim=c['action_dim'], vq_cls=VQ1d_EMA, k=c.get('k', 128)).cuda().eval()
    vae.load_state_dict(c['vae'])
    var_global = c['action_var_global'].view(1, 1, -1)
    adim = c['action_dim']
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=4)
            if sp.chunk_index and sp.robot == ROBOT: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    print(f"[finder] dataset ready: {len(ds)} chunks", flush=True)

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    budget_gb = total_mem * MEM_BUDGET_FRAC
    print(f"[finder] GPU total {total_mem:.1f}GB, budget {budget_gb:.1f}GB ({int(MEM_BUDGET_FRAC*100)}%)\n", flush=True)

    # ────────────────────────── PHASE 1: BS SCAN ──────────────────────────
    chosen_bs = args.bs
    if not args.skip_bs and chosen_bs is None:
        print(f"{'='*70}\nPHASE 1 — BS SCAN\n{'='*70}", flush=True)
        last_ok = None
        for bs in BS_CANDIDATES:
            print(f"\n[BS={bs}]", flush=True)
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                mods = build_pipeline()
                opt = MuSGD_LARS([p for m in mods for p in m.parameters() if p.requires_grad],
                                  lr=1e-4, momentum=0.95, weight_decay=2e-3)
                img_n, t5s, st, gtc, eid_t, Bf = build_one_batch(bs, ds, ctx_static, vae, var_global, adim)
                for step in range(N_WARM_STEPS):
                    opt.zero_grad(set_to_none=True)
                    loss = fwd_loss(mods, img_n, t5s, st, gtc, eid_t, Bf)
                    loss.backward(); opt.step()
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                print(f"  ✓ BS={bs} OK, peak={peak_gb:.2f}GB ({peak_gb/total_mem*100:.0f}% of total)", flush=True)
                if peak_gb < budget_gb:
                    last_ok = bs
                else:
                    print(f"  → BS={bs} fits but exceeds {int(MEM_BUDGET_FRAC*100)}% budget — stopping.", flush=True)
                    break
                del mods, opt, img_n, t5s, st, gtc, eid_t, loss
            except torch.cuda.OutOfMemoryError:
                print(f"  ✗ BS={bs} OOM — stopping.", flush=True)
                # aggressive cleanup of the half-allocated state
                try: del mods, opt, img_n, t5s, st, gtc, eid_t, loss
                except Exception: pass
                break
            except Exception as e:
                print(f"  ✗ BS={bs} ERROR ({type(e).__name__}): {e}", flush=True); break
            torch.cuda.empty_cache(); gc.collect()
        chosen_bs = last_ok
        # full reset between phases: everything from Phase 1 must be gone
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        free_gb = (torch.cuda.mem_get_info()[0]) / 1024**3
        print(f"\n[finder] chosen BS = {chosen_bs}  (free GPU after cleanup: {free_gb:.2f}GB)\n", flush=True)
    if chosen_bs is None:
        print("[finder] no BS chosen, exiting"); return

    # ────────────────────────── PHASE 2: LR RANGE TEST (real loader) ──────────────────────────
    if not args.skip_lr:
        print(f"{'='*70}\nPHASE 2 — LR RANGE TEST at BS={chosen_bs} (real dataloader)\n"
              f"  exponential ramp lr: {LR_LO:.1e} → {LR_HI:.1e} over {N_LR_STEPS} steps\n"
              f"  workers={NW_DEFAULT}, prefetch={PF_DEFAULT}\n"
              f"{'='*70}", flush=True)
        torch.cuda.empty_cache()
        mods = build_pipeline()
        params = [p for m in mods for p in m.parameters() if p.requires_grad]
        opt = MuSGD_LARS(params, lr=LR_LO, momentum=0.95, weight_decay=2e-3)
        eid_pre = torch.tensor([EMBODIMENT_ID.get(ROBOT, len(EMBODIMENTS))] * chosen_bs,
                                dtype=torch.long).cuda()
        loader = _build_real_loader(ds, chosen_bs)
        log_lo = math.log(LR_LO); log_hi = math.log(LR_HI)
        rows = []
        smooth = None; smooth_beta = 0.9
        best_loss = float('inf'); best_lr = LR_LO
        diverge_lr = None
        t_start = time.time()
        step = 0
        for batch in _iter_forever(loader):
            if step >= N_LR_STEPS: break
            lr = math.exp(log_lo + (log_hi - log_lo) * step / max(1, N_LR_STEPS - 1))
            for g in opt.param_groups: g['lr'] = lr
            try:
                frames, states, actions, prevs, tasks = batch
                lv = _one_train_step(mods, opt, frames, states, actions, prevs, tasks,
                                     ctx_static, vae, var_global, eid_pre)
                if lv is None:
                    print(f"  step={step:3d} lr={lr:.2e} loss=NaN/Inf — DIVERGED", flush=True)
                    diverge_lr = lr; break
                smooth = lv if smooth is None else smooth_beta * smooth + (1 - smooth_beta) * lv
                if smooth < best_loss: best_loss = smooth; best_lr = lr
                if smooth > 1.5 * best_loss and step > 20:
                    print(f"  step={step:3d} lr={lr:.2e} loss={lv:.4f} smooth={smooth:.4f} — DIVERGED (>1.5x best)", flush=True)
                    diverge_lr = lr; break
                rows.append((step, lr, lv, smooth))
                if step % 10 == 0 or step == N_LR_STEPS - 1:
                    print(f"  step={step:3d} lr={lr:.2e} loss={lv:.4f} smooth={smooth:.4f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at step {step}, lr {lr:.2e} — stopping LR test", flush=True); break
            step += 1
        elapsed = time.time() - t_start
        recommended_lr = best_lr / 4.0
        if diverge_lr is not None:
            recommended_lr = min(recommended_lr, diverge_lr / 4.0)
        print(f"\n[finder] LR range test done in {elapsed:.0f}s ({step} steps)")
        print(f"  best smoothed loss = {best_loss:.4f} at lr = {best_lr:.2e}")
        if diverge_lr is not None: print(f"  divergence onset    = {diverge_lr:.2e}")
        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  RECOMMENDED lr = {recommended_lr:.2e}  (best/4)  │")
        print(f"  │  RECOMMENDED BS = {chosen_bs}                       │")
        print(f"  └─────────────────────────────────────────────┘\n")
        with open('/tmp/lr_range_test.csv', 'w') as f:
            f.write("step,lr,loss,smooth\n")
            for r in rows: f.write(f"{r[0]},{r[1]:.6e},{r[2]:.6f},{r[3]:.6f}\n")
        print(f"  detailed curve → /tmp/lr_range_test.csv")
        del mods, opt, loader
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    else:
        recommended_lr = 2e-3

    # ────────────────────────── PHASE 3: g_input_noise GRID (real loader) ──────────────────────────
    print(f"\n{'='*70}\nPHASE 3 — g_input_noise SWEEP at BS={chosen_bs}, lr={recommended_lr:.2e} (real loader)\n"
          f"  grid {NOISE_GRID}, {N_NOISE_STEPS} steps each\n{'='*70}", flush=True)
    noise_results = []
    eid_pre = torch.tensor([EMBODIMENT_ID.get(ROBOT, len(EMBODIMENTS))] * chosen_bs,
                            dtype=torch.long).cuda()
    for sigma in NOISE_GRID:
        print(f"\n[σ={sigma:.3f}]", flush=True)
        torch.cuda.empty_cache()
        mods = build_pipeline(g_input_noise=sigma)
        params = [p for m in mods for p in m.parameters() if p.requires_grad]
        opt = MuSGD_LARS(params, lr=recommended_lr, momentum=0.95, weight_decay=2e-3)
        loader = _build_real_loader(ds, chosen_bs)
        smooth = None; smooth_beta = 0.95
        last_window = []
        t_start = time.time()
        diverged = False
        step = 0
        for batch in _iter_forever(loader):
            if step >= N_NOISE_STEPS: break
            try:
                frames, states, actions, prevs, tasks = batch
                lv = _one_train_step(mods, opt, frames, states, actions, prevs, tasks,
                                     ctx_static, vae, var_global, eid_pre)
                if lv is None:
                    print(f"  step={step} NaN — DIVERGED"); diverged = True; break
                smooth = lv if smooth is None else smooth_beta * smooth + (1 - smooth_beta) * lv
                if step >= N_NOISE_STEPS - 50: last_window.append(lv)
                if step % 50 == 0 or step == N_NOISE_STEPS - 1:
                    print(f"  step={step:3d} loss={lv:.4f} smooth={smooth:.4f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at step {step}"); break
            step += 1
        elapsed = time.time() - t_start
        final = sum(last_window) / max(1, len(last_window))
        noise_results.append((sigma, final, smooth, diverged, elapsed))
        del mods, opt, loader
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"  {'σ':>6s} {'final_loss(last50)':>20s} {'smoothed':>12s} {'time':>8s}")
    best_sigma, best_loss = None, float('inf')
    for sigma, final, smooth, div, el in noise_results:
        flag = ' DIV' if div else ''
        print(f"  {sigma:>6.3f} {final:>20.4f} {smooth:>12.4f} {el:>7.0f}s{flag}")
        if not div and final < best_loss: best_loss = final; best_sigma = sigma

    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  FINAL RECOMMENDATION                                    │")
    print(f"  │    BS               = {chosen_bs:<35d}│")
    print(f"  │    LR               = {recommended_lr:<35.2e}│")
    print(f"  │    g_input_noise σ  = {best_sigma if best_sigma is not None else 0:<35.3f}│")
    print(f"  └──────────────────────────────────────────────────────────┘\n")


if __name__ == '__main__':
    main()
