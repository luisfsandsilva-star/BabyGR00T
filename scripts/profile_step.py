#!/usr/bin/env python3
"""Time per training step broken down by phase.
Runs 20 real training steps using the same dataloader and model as train_oxe,
prints mean time spent in: data load wait, CNN forward, T5+vision concat,
policy forward, backward, optimizer step.

Goal: find the actual bottleneck. If data load wait dominates → fix dataloader.
If policy forward dominates → look at grad_checkpoint / kernel launches /
H·L iteration count. If backward dominates → grad_checkpoint recomputation.
"""
import os, sys, time, glob, random
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import torch, torch.nn as nn, numpy as np
from torch.utils.data import DataLoader
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm,
                             ActionVQVAE1d, VQ1d_EMA, MuSGD_LARS)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm import augment

ROBOT='widowx'; DIM=512; HEADS=8; KV=2; FF=2048; DEPTH=2
N_EMB=len(EMBODIMENTS)+1; PFX=16; IMG=144; MAX_TEXT=64
L_INNER=5; H_OUTER=3
BS = 512
N_WARM = 5    # warmup steps (excluded from timing)
N_TIME = 15   # timed steps
NW = int(os.environ.get('NW', '24'))  # dataloader workers (default 24 from 64-CPU machine)
PF = int(os.environ.get('PF', '4'))   # prefetch_factor
torch.set_num_threads(4); random.seed(0); torch.manual_seed(0)


def build_pipeline(grad_ckpt=True):
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=4,
                       out_dim=192, norm='gn', pos_emb=True, img_size=IMG,
                       dropout=0.2, n_embodiments=0).cuda()
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9).cuda()
    cnn_proj = nn.Linear(192, DIM).cuda()
    text_proj = nn.Linear(512, DIM).cuda()
    kv_norm = ScaleNorm(DIM).cuda()
    se = nn.ModuleDict({ROBOT: nn.Sequential(
        nn.Linear(8, DIM), nn.GELU(), nn.Linear(DIM, DIM))}).cuda()
    emb_id = nn.Embedding(N_EMB, DIM*PFX).cuda()
    n_vis=(IMG//32)**2
    policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=DIM, heads=HEADS,
        kv_heads=KV, ff_hidden=FF, depth=DEPTH, L_inner=L_INNER, H_outer=H_OUTER,
        state_dim=DIM, max_prefix=n_vis+MAX_TEXT+16+PFX,
        weighting='clamp_direct', update_mode='damped',
        alpha_parametrization='sigmoid', alpha_per_dim=True,
        n_embodiments=N_EMB, per_emb_head=False, dropout=0.2,
        g_input_noise=0.01, grad_checkpoint=grad_ckpt).cuda()
    for m in [cnn, text_agg, cnn_proj, text_proj, kv_norm, se, emb_id, policy]:
        m.train()
    return cnn, text_agg, cnn_proj, text_proj, kv_norm, se, emb_id, policy


def build_dataset():
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=4)
            if sp.chunk_index and sp.robot == ROBOT: specs.append(sp)
        except Exception: pass
    return MultiOXEDataset(specs, chunk_len=16, lookback=16)

_AUG_KW = dict(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4),
               blur_sigma=(0.0, 1.5), crop_keep=(0.70, 1.0))

def _frame_to_tensor(pil):
    import random as _r
    pil = pil.convert('RGB')
    pil = augment._apply_visual_params(pil, augment._sample_visual_params(_r, **_AUG_KW))
    pil = pil.resize((IMG, IMG))
    return torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.

def _collate(batch):
    frames = torch.stack([_frame_to_tensor(b[0]) for b in batch])
    states = torch.stack([b[1] for b in batch])
    actions = torch.stack([b[2] for b in batch])
    prevs = torch.stack([b[3] for b in batch])
    tasks = [b[4] for b in batch]
    embs = [EMBODIMENTS[b[5]] if b[5] < len(EMBODIMENTS) else 'unknown' for b in batch]
    return frames, states, actions, prevs, tasks, embs, torch.tensor([b[6] for b in batch])


def main():
    print(f"[prof] building model + dataset...", flush=True)
    mods = build_pipeline(grad_ckpt=True)
    cnn, text_agg, cnn_proj, text_proj, kv_norm, se, emb_id, policy = mods
    params = [p for m in mods for p in m.parameters() if p.requires_grad]
    opt = MuSGD_LARS(params, lr=2e-3, momentum=0.95, weight_decay=2e-3)
    ds = build_dataset()
    print(f"[prof] dataset: {len(ds)} chunks", flush=True)

    vae_c = torch.load(f'data/ckpts/oxe_vqvae_{ROBOT}.pt', map_location='cpu', weights_only=False)
    vae = ActionVQVAE1d(action_dim=vae_c['action_dim'], vq_cls=VQ1d_EMA,
                         k=vae_c.get('k', 128)).cuda().eval()
    vae.load_state_dict(vae_c['vae'])
    var_global = vae_c['action_var_global'].view(1, 1, -1).cuda()

    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)['var_global'].cuda()

    loader = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=NW,
                        pin_memory=True, persistent_workers=True, prefetch_factor=PF,
                        collate_fn=_collate)
    print(f"[prof] dataloader: BS={BS}, workers={NW}, pin_memory=True, prefetch={PF}", flush=True)

    t_data = []; t_targets = []; t_fwd = []; t_bwd = []; t_opt = []; t_total = []
    eid_pre = torch.tensor([EMBODIMENT_ID.get(ROBOT, len(EMBODIMENTS))] * BS,
                            dtype=torch.long).cuda()

    print(f"[prof] running {N_WARM} warmup + {N_TIME} timed steps", flush=True)
    step = 0; t_iter_start = time.time(); t_data_start = time.time()
    for frames, states, actions, prevs, tasks, embs, _ in loader:
        step += 1
        torch.cuda.synchronize(); t1 = time.time()
        t_data.append(t1 - t_data_start)

        frames = frames.cuda(non_blocking=True)
        states = states.cuda(non_blocking=True)
        actions = actions.cuda(non_blocking=True)
        prevs = prevs.cuda(non_blocking=True)
        Bf = frames.shape[0]
        eid = eid_pre[:Bf]

        # build T5 hiddens per batch (on CPU then to GPU)
        t5s = torch.zeros(9, Bf, MAX_TEXT, 512, device='cuda')
        for b, tk in enumerate(tasks):
            e = t5['embeddings'].get(tk)
            if e is None: continue
            h = e['hidden'].float().cuda()
            t = min(h.shape[1], MAX_TEXT); L = min(h.shape[0], 9)
            t5s[:L, b, :t, :] = h[:L, :t, :]

        # GT codes
        m_pv = prevs.mean(dim=1, keepdim=True)
        S = ((prevs - m_pv) ** 2).sum(dim=1, keepdim=True)
        lam = actions.shape[1] / (S + actions.shape[1] * var_global)
        xn = ((actions - m_pv) * lam.sqrt()).transpose(1, 2)
        with torch.no_grad():
            cd, _ = vae.encode_with_soft(xn, tau=0.1)
        gt = [cd[0]]
        torch.cuda.synchronize(); t2 = time.time()
        t_targets.append(t2 - t1)

        img_n = normalize_image(frames, img_var)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
            tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
            etok = emb_id(eid).view(Bf, PFX, DIM)
            vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
            s_enc = se[ROBOT](states)
            loss, per, _ = policy.forward_loss(gt, vis, s_enc,
                n_inner=L_INNER, n_outer=H_OUTER, h_max=H_OUTER,
                mask_ratio_max=1.0, emb_id=eid,
                label_smoothing=0.10, mask_sampler='cosine')
        torch.cuda.synchronize(); t3 = time.time()
        t_fwd.append(t3 - t2)

        loss.backward()
        torch.cuda.synchronize(); t4 = time.time()
        t_bwd.append(t4 - t3)

        opt.step()
        torch.cuda.synchronize(); t5_t = time.time()
        t_opt.append(t5_t - t4)
        t_total.append(t5_t - t_iter_start)

        if step <= N_WARM:
            print(f"  warmup step {step}: total={t5_t - t_iter_start:.2f}s  loss={loss.item():.3f}", flush=True)
        else:
            print(f"  step {step}: total={t5_t - t_iter_start:.2f}s "
                  f"data={t1-t_data_start:.2f} tgt={t2-t1:.2f} "
                  f"fwd={t3-t2:.2f} bwd={t4-t3:.2f} opt={t5_t-t4:.2f}  loss={loss.item():.3f}", flush=True)
        t_iter_start = time.time(); t_data_start = time.time()
        if step >= N_WARM + N_TIME: break

    # summarize (skip warmup)
    def mean(xs): return sum(xs[N_WARM:]) / max(1, len(xs[N_WARM:]))
    print(f"\n{'='*60}\nMEAN per-step breakdown (last {N_TIME} steps):")
    print(f"  data wait:    {mean(t_data):.3f}s")
    print(f"  build target: {mean(t_targets):.3f}s")
    print(f"  forward:      {mean(t_fwd):.3f}s")
    print(f"  backward:     {mean(t_bwd):.3f}s")
    print(f"  optimizer:    {mean(t_opt):.3f}s")
    print(f"  TOTAL:        {mean(t_total):.3f}s")

    # repeat with grad_checkpoint=False
    print(f"\n{'='*60}\nRe-test with grad_checkpoint=False (faster bwd, more memory)")
    del mods, opt; torch.cuda.empty_cache()
    import gc; gc.collect()
    mods = build_pipeline(grad_ckpt=False)
    cnn, text_agg, cnn_proj, text_proj, kv_norm, se, emb_id, policy = mods
    params = [p for m in mods for p in m.parameters() if p.requires_grad]
    opt = MuSGD_LARS(params, lr=2e-3, momentum=0.95, weight_decay=2e-3)
    t_total2 = []; t_iter_start = time.time(); step = 0
    for frames, states, actions, prevs, tasks, embs, _ in loader:
        step += 1
        frames = frames.cuda(non_blocking=True); states = states.cuda(non_blocking=True)
        actions = actions.cuda(non_blocking=True); prevs = prevs.cuda(non_blocking=True)
        Bf = frames.shape[0]; eid = eid_pre[:Bf]
        t5s = torch.zeros(9, Bf, MAX_TEXT, 512, device='cuda')
        for b, tk in enumerate(tasks):
            e = t5['embeddings'].get(tk)
            if e is None: continue
            h = e['hidden'].float().cuda()
            t = min(h.shape[1], MAX_TEXT); L = min(h.shape[0], 9)
            t5s[:L, b, :t, :] = h[:L, :t, :]
        m_pv = prevs.mean(dim=1, keepdim=True)
        S = ((prevs - m_pv) ** 2).sum(dim=1, keepdim=True)
        lam = actions.shape[1] / (S + actions.shape[1] * var_global)
        xn = ((actions - m_pv) * lam.sqrt()).transpose(1, 2)
        with torch.no_grad():
            cd, _ = vae.encode_with_soft(xn, tau=0.1)
        gt = [cd[0]]
        img_n = normalize_image(frames, img_var)
        opt.zero_grad(set_to_none=True)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
                tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
                etok = emb_id(eid).view(Bf, PFX, DIM)
                vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
                s_enc = se[ROBOT](states)
                loss, per, _ = policy.forward_loss(gt, vis, s_enc,
                    n_inner=L_INNER, n_outer=H_OUTER, h_max=H_OUTER,
                    mask_ratio_max=1.0, emb_id=eid,
                    label_smoothing=0.10, mask_sampler='cosine')
            loss.backward(); opt.step()
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at step {step} without grad_checkpoint"); break
        torch.cuda.synchronize()
        dt = time.time() - t_iter_start; t_total2.append(dt)
        if step <= N_WARM:
            print(f"  warmup step {step}: total={dt:.2f}s loss={loss.item():.3f}", flush=True)
        else:
            print(f"  step {step}: total={dt:.2f}s loss={loss.item():.3f}", flush=True)
        t_iter_start = time.time()
        if step >= N_WARM + N_TIME: break
    if len(t_total2) > N_WARM:
        m2 = sum(t_total2[N_WARM:]) / max(1, len(t_total2[N_WARM:]))
        print(f"\n  TOTAL (no grad_ckpt): {m2:.3f}s (vs {mean(t_total):.3f}s with) → {mean(t_total)/m2:.2f}× speedup")


if __name__ == '__main__':
    main()
