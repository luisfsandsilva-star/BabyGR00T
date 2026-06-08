#!/usr/bin/env python3
"""Train the VAE-TRM policy with the NEW conditioning stack:
  • trainable EfficientCNN over the raw frame  → 49 vision tokens
  • frozen-T5 cache → reused LayerAggregator (13 layers) → text tokens
  • fuse [CNN ⊕ T5] as ONE sequence (each projected by its own weights),
    passed as the policy's `vis` KV → policy cross-attends to it (+ state).

Frames are loaded with video into RAM (small subsets only — overfit/sweep).
T5 is read from the precomputed cache (scripts/cache_t5.py); never run live.
"""
import os, sys, time, math, random, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np
import torch, torch.nn as nn
from babygroot_strm import (RevIN, ActionVQVAE1d, VQ1d_EMA, STRMPolicyVAE,
                            LayerAggregator, MuSGD_LARS, ScaleNorm, load_lerobot_episodes)
from babygroot_strm.cnn_vision import EfficientCNN


def _tuple(s): return tuple(int(x) for x in str(s).split(','))


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear: y = base(x) + (α/r)·(x A B)."""
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        self.r = r; self.scale = (alpha / r) if r > 0 else 0.0
        if r > 0:
            self.A = nn.Parameter(torch.zeros(base.in_features, r))
            self.B = nn.Parameter(torch.zeros(r, base.out_features))
            nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
    def forward(self, x):
        y = self.base(x)
        return y + self.scale * (x @ self.A @ self.B) if self.r > 0 else y


def apply_lora(module, r=8, alpha=16):
    """Recursively wrap every nn.Linear (not already LoRALinear) in a module."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, LoRALinear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            apply_lora(child, r=r, alpha=alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oxe-dataset-id', default='IPEC-COMMUNITY/bridge_orig_lerobot')
    ap.add_argument('--oxe-camera', default='observation.images.image_0')
    ap.add_argument('--n-eps-cap', type=int, default=24)
    ap.add_argument('--t5-cache', default='data/cache/t5_text_cache.pt')
    ap.add_argument('--vae-ckpt', default='data/ckpts/oxe_vqvae_1800ep_16k.pt')
    ap.add_argument('--img-size', type=int, default=224)
    ap.add_argument('--max-text', type=int, default=16)
    # CNN config (swept)
    ap.add_argument('--cnn-dims', type=_tuple, default=(64, 128, 256, 384))
    ap.add_argument('--cnn-depths', type=_tuple, default=(2, 2, 4, 2))
    ap.add_argument('--cnn-expand', type=int, default=3)
    ap.add_argument('--cnn-out-dim', type=int, default=384)
    ap.add_argument('--cnn-norm', default='scalenorm', choices=['scalenorm', 'layernorm'])
    ap.add_argument('--cnn-pe', action='store_true')
    # policy
    ap.add_argument('--dim', type=int, default=1728)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--L-inner', type=int, default=5)
    ap.add_argument('--H-outer', type=int, default=2)
    ap.add_argument('--h-max', type=int, default=4)
    ap.add_argument('--beta', type=float, default=1e-3)
    ap.add_argument('--free-bits', type=float, default=0.1)
    # train
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=9.5e-4)
    ap.add_argument('--ckpt-path', default='data/ckpts/cnn_policy.pt')
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--ckpt-every', type=int, default=500)
    ap.add_argument('--resume', action='store_true',
                    help="Resume modules + step from --ckpt-path if it exists.")
    ap.add_argument('--dropout-prob', type=float, default=0.0,
                    help="Per-sample probability of independently zeroing each modality "
                         "(state, vision-tokens, text-tokens) during training. Forces the model "
                         "to be robust across modalities rather than over-relying on any one.")
    ap.add_argument('--task-balanced', action='store_true',
                    help="Load 4 SimplerEnv-task episodes (from --task-file) + uniform-per-task "
                         "weighted sampling. Overrides --n-eps-cap.")
    ap.add_argument('--task-file', default='data/cache/sim_task_episodes.pt')
    ap.add_argument('--train-frac', type=float, default=0.8,
                    help="Per-task fraction to use for training (rest = held-out for eval).")
    ap.add_argument('--mix-general', type=int, default=0,
                    help="If >0, also load that many GENERAL Bridge episodes (first N by index) "
                         "and add as a 5th sampling group (~50%% of batches → diversity / regularization).")
    ap.add_argument('--general-weight', type=float, default=0.5,
                    help="Sampling-weight share for the general group when --mix-general>0 "
                         "(rest is split uniformly across the 4 task groups).")
    ap.add_argument('--cnn-dropout', type=float, default=0.0,
                    help="Standard Dropout2d inside the CNN blocks (light — modality dropout is the heavy one).")
    ap.add_argument('--weight-decay', type=float, default=1e-4,
                    help="Optimizer weight decay.")
    ap.add_argument('--lora-r', type=int, default=0,
                    help="If >0, freeze the policy and add LoRA adapters of rank r to every nn.Linear in it.")
    ap.add_argument('--lora-alpha', type=float, default=16.0)
    ap.add_argument('--freeze-cnn', action='store_true',
                    help="Freeze the CNN vision encoder (preserve pretrained visual features).")
    ap.add_argument('--freeze-text', action='store_true',
                    help="Freeze the text aggregator + text projection (preserve pretrained text path).")
    ap.add_argument('--state-noise', type=float, default=0.0,
                    help="Stddev of Gaussian noise added to the proprio state during training.")
    ap.add_argument('--strong-aug', action='store_true',
                    help="Use much stronger photometric/blur/crop ranges on the input frames.")
    ap.add_argument('--ema-decay', type=float, default=0.0,
                    help="If >0, maintain a Polyak/EMA shadow over all trainable params "
                         "(θ_ema ← decay·θ_ema + (1-decay)·θ) and save it as 'ema_params' "
                         "alongside the live weights. Inference can swap in θ_ema. "
                         "Typical: 0.999 (half-life ~693 steps).")
    # annealed ρ revival (VQ-VAE dead-code-style) for ρ_L / ρ_H
    ap.add_argument('--revive', action='store_true')
    ap.add_argument('--revive-thresh', type=float, default=0.02)
    ap.add_argument('--revive-patience', type=int, default=200)
    ap.add_argument('--revive-to', type=float, default=0.1)
    ap.add_argument('--revive-cooldown', type=int, default=400)
    ap.add_argument('--revive-decay', type=float, default=0.7,
                    help="<1 anneals the revive peak each revival → smooth death "
                         "if the loop keeps getting rejected; revives stop once "
                         "the peak drops below --revive-thresh.")
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── frozen VQ-VAE codebook ──
    vck = torch.load(args.vae_ckpt, map_location=dev, weights_only=False)
    adim = vck.get('action_dim', 7)
    vae = ActionVQVAE1d(action_dim=adim, vq_cls=VQ1d_EMA).to(dev)
    revin = RevIN(adim).to(dev)
    vae.load_state_dict(vck['vae']); revin.load_state_dict(vck['revin'])
    vae.eval(); revin.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    seq_lens = tuple(vae.seq_lens); K = vae.vqs[0].K
    var_global = vck['action_var_global'].to(dev).view(1, 1, -1)  # per-dim global variance (precision prior)

    # ── T5 cache ──
    t5 = torch.load(args.t5_cache, map_location='cpu', weights_only=False)
    t5_emb, t5_dim, t5_layers = t5['embeddings'], t5['dim'], t5['n_layers']

    # ── episodes WITH video (RAM) ──
    def _classify(p):
        p = p.lower()
        if 'spoon' in p and any(w in p for w in ['towel','cloth']): return 'spoon_on_towel'
        if 'carrot' in p and 'plate' in p: return 'carrot_on_plate'
        if 'eggplant' in p and any(w in p for w in ['basket','container','box','bowl','tray','bin']): return 'eggplant_in_basket'
        if 'green' in p and 'yellow' in p and any(b in p for b in ['block','cube','rect','box']) and any(o in p for o in ['on','top','stack']): return 'stack_green_on_yellow'
        return None
    per_task_index = None; heldout_eps_by_task = None
    if args.task_balanced:
        te = torch.load(args.task_file, map_location='cpu', weights_only=False)
        tasks2eps = te['episodes_per_task']
        rng = random.Random(0)
        train_eps_by_task, heldout_eps_by_task = {}, {}
        for tname, eps_list in tasks2eps.items():
            ll = sorted(set(eps_list))
            if not ll: continue
            rng.shuffle(ll); n_tr = max(1, int(len(ll) * args.train_frac))
            train_eps_by_task[tname] = ll[:n_tr]; heldout_eps_by_task[tname] = ll[n_tr:]
        all_train_eps = [e for v in train_eps_by_task.values() for e in v]
        print(f"  task-balanced: {len(all_train_eps)} train eps over {len(train_eps_by_task)} tasks")
        for k, v in train_eps_by_task.items():
            print(f"    {k:24s}: train={len(v):4d}  heldout={len(heldout_eps_by_task[k]):3d}")
        all_eps_set = set(all_train_eps)
        if args.mix_general > 0:
            general_eps = sorted(set(range(args.mix_general)) - all_eps_set)
            all_eps_set |= set(general_eps)
            print(f"  + {len(general_eps)} general episodes (first {args.mix_general} not already in task set)")
        eps = load_lerobot_episodes(args.oxe_dataset_id, camera_key=args.oxe_camera,
                                    load_video=True, episode_indices=sorted(all_eps_set))
        per_task_index = {}
        for ei, e in enumerate(eps):
            t = _classify(e[3])
            if t is None:                       # not a sim-task → general bucket (only if mixing)
                if args.mix_general > 0: per_task_index.setdefault('general', []).extend([(ei, ci) for ci in range(1, e[0].shape[0])])
                continue
            per_task_index.setdefault(t, []).extend([(ei, ci) for ci in range(1, e[0].shape[0])])
        print(f"  chunks per group: {[(k, len(v)) for k,v in per_task_index.items()]}")
        index = [(ei, ci) for ei, e in enumerate(eps) for ci in range(1, e[0].shape[0])]
    else:
        eps = load_lerobot_episodes(args.oxe_dataset_id, camera_key=args.oxe_camera,
                                    load_video=True, n_episodes=args.n_eps_cap)
        index = [(ei, ci) for ei, e in enumerate(eps) for ci in range(1, e[0].shape[0])]
    state_dim = int(eps[0][1].shape[-1])
    print(f"{len(eps)} eps, {len(index)} chunks, state_dim={state_dim}, K={K}, seq_lens={seq_lens}", flush=True)

    from babygroot_strm import augment
    _AUG_KW = (dict(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4),
                    blur_sigma=(0.0, 1.5), crop_keep=(0.70, 1.0)) if args.strong_aug else {})
    def frame_to_tensor(pil, aug=False):
        pil = pil.convert('RGB')
        if aug:                                      # photometric jitter + blur + crop (+optional stronger)
            pil = augment._apply_visual_params(pil, augment._sample_visual_params(random, **_AUG_KW))
        pil = pil.resize((args.img_size, args.img_size))
        return torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.0

    # ── modules ──
    cnn = EfficientCNN(dims=args.cnn_dims, depths=args.cnn_depths, expand=args.cnn_expand,
                       out_dim=args.cnn_out_dim, norm=args.cnn_norm, pos_emb=args.cnn_pe,
                       img_size=args.img_size, dropout=args.cnn_dropout).to(dev)
    text_agg  = LayerAggregator(hidden_dim=t5_dim, n_layers=t5_layers).to(dev)
    cnn_proj  = nn.Linear(args.cnn_out_dim, args.dim).to(dev)
    text_proj = nn.Linear(t5_dim, args.dim).to(dev)
    kv_norm   = ScaleNorm(args.dim).to(dev)      # bound the fused-KV scale (ScaleNorm — learns faster than LN)
    n_vis = (args.img_size // 32) ** 2
    policy = STRMPolicyVAE(seq_lens=seq_lens, k_codebook=K, dim=args.dim, heads=8,
                           depth=args.depth, L_inner=args.L_inner, H_outer=args.H_outer,
                           state_dim=state_dim, max_prefix=n_vis + args.max_text + 16,
                           beta=args.beta, free_bits=args.free_bits).to(dev)
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, policy]
    keys_for_load = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'policy']
    # Load pretrained weights FIRST (before LoRA/freeze changes the parameter set).
    _resumed_step = 0; _loaded_ck = None
    if args.resume and os.path.exists(args.ckpt_path):
        _loaded_ck = torch.load(args.ckpt_path, map_location=dev, weights_only=False)
        for k, m in zip(keys_for_load, mods): m.load_state_dict(_loaded_ck[k])
        _resumed_step = _loaded_ck.get('step', 0)
        print(f"resumed weights from {args.ckpt_path} @ step {_resumed_step}", flush=True)
    # Optional freezing of vision / text paths to preserve pretrained features.
    if args.freeze_cnn:
        for p in cnn.parameters(): p.requires_grad_(False)
    if args.freeze_text:
        for p in text_agg.parameters(): p.requires_grad_(False)
        for p in text_proj.parameters(): p.requires_grad_(False)
    # Optional LoRA: freeze policy completely, then add low-rank adapters to every Linear.
    if args.lora_r > 0:
        for p in policy.parameters(): p.requires_grad_(False)
        apply_lora(policy, r=args.lora_r, alpha=args.lora_alpha)
        policy.to(dev)
    trainable = [p for m in mods for p in m.parameters() if p.requires_grad]
    opt = MuSGD_LARS(trainable, lr=args.lr, momentum=0.95, weight_decay=args.weight_decay)
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for m in mods for p in m.parameters())
    print(f"params: CNN {sum(p.numel() for p in cnn.parameters())/1e6:.2f}M  "
          f"policy {sum(p.numel() for p in policy.parameters())/1e6:.2f}M  "
          f"n_vis_tok={n_vis}  | TRAINABLE {n_train/1e6:.2f}M / {n_total/1e6:.2f}M "
          f"({100*n_train/max(n_total,1):.1f}%)", flush=True)

    if per_task_index is not None:
        SIM_TASKS = ['spoon_on_towel','carrot_on_plate','eggplant_in_basket','stack_green_on_yellow']
        sim_present = [t for t in SIM_TASKS if per_task_index.get(t)]
        has_general = 'general' in per_task_index and per_task_index['general']
        if has_general and args.mix_general > 0:
            wg = args.general_weight; ws = (1 - wg) / max(1, len(sim_present))
            sample_groups = sim_present + ['general']; sample_weights = [ws]*len(sim_present) + [wg]
        else:
            sample_groups = sim_present; sample_weights = [1.0/len(sim_present)] * len(sim_present)
        print(f"  sampling weights: {[(g, round(w,3)) for g,w in zip(sample_groups, sample_weights)]}")
    def get_batch(B):
        if per_task_index is not None:                  # weighted sampling by group
            grps = random.choices(sample_groups, weights=sample_weights, k=B)
            picks = [random.choice(per_task_index[g]) for g in grps]
        else:
            picks = [random.choice(index) for _ in range(B)]
        fr = torch.stack([frame_to_tensor(eps[ei][2][ci][-1], aug=True) for ei, ci in picks]).to(dev)
        st = torch.stack([eps[ei][1][ci] for ei, ci in picks]).float().to(dev)
        ac = torch.stack([eps[ei][0][ci] for ei, ci in picks]).float().to(dev)
        pv = torch.stack([eps[ei][0][ci - 1] for ei, ci in picks]).float().to(dev)   # prev chunk = lookback
        tk = [eps[ei][3] for ei, ci in picks]
        return fr, st, ac, pv, tk

    def t5_layers_batch(tasks):
        B, T = len(tasks), args.max_text
        out = torch.zeros(t5_layers, B, T, t5_dim)
        for b, tk in enumerate(tasks):
            e = t5_emb.get(tk)
            if e is None: continue
            h = e['hidden'].float(); t = min(h.shape[1], T)
            out[:, b, :t, :] = h[:, :t, :]
        return out.to(dev)

    def encode_vis(frames, tasks, mod_dropout=0.0):
        vtok, _ = cnn(frames)                              # (B,49,out_dim)
        vtok = cnn_proj(vtok)                              # (B,49,dim)
        t5s = t5_layers_batch(tasks)                       # (L,B,T,t5_dim)
        tagg = text_agg([t5s[l] for l in range(t5s.shape[0])])   # (B,T,t5_dim)
        ttok = text_proj(tagg)                             # (B,T,dim)
        if mod_dropout > 0.0:                              # independent per-sample modality dropout
            B = vtok.shape[0]
            vk = (torch.rand(B, 1, 1, device=vtok.device) >= mod_dropout).float()
            tk_ = (torch.rand(B, 1, 1, device=vtok.device) >= mod_dropout).float()
            vtok = vtok * vk; ttok = ttok * tk_
        return kv_norm(torch.cat([vtok, ttok], dim=1))     # (B,49+T,dim)

    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'policy']
    ema_state = None        # populated after resume if --ema-decay>0

    def save(step):
        os.makedirs(os.path.dirname(args.ckpt_path) or '.', exist_ok=True)
        tmp = args.ckpt_path + '.tmp'
        out = ({k: m.state_dict() for k, m in zip(keys, mods)}
               | {'step': step, 'opt': opt.state_dict(), 'args': vars(args),
                  'heldout_eps_by_task': heldout_eps_by_task})
        if ema_state is not None:
            out['ema_params'] = {k: v.detach().cpu() for k, v in ema_state.items()}
            out['ema_decay'] = args.ema_decay
        torch.save(out, tmp)
        os.replace(tmp, args.ckpt_path)

    step = 0; t0 = time.perf_counter(); wl = wc = wa = wn = 0
    warmup = 150
    revive_dead = {'L': 0, 'H': 0}; revive_cd = {'L': 0, 'H': 0}
    revive_peak = {'L': args.revive_to, 'H': args.revive_to}
    # weights already loaded above (before LoRA/freeze); restore step + opt state if not LoRA
    if _loaded_ck is not None:
        step = _resumed_step
        if 'opt' in _loaded_ck and args.lora_r == 0:
            try: opt.load_state_dict(_loaded_ck['opt'])
            except Exception as e: print(f"  [warn] opt state not loaded ({e})")
    # EMA / Polyak shadow over all trainable params. Initialized from current weights;
    # resumed from ckpt if it has them; otherwise starts fresh from the loaded base.
    if args.ema_decay > 0:
        ema_state = {}
        for nm, m in zip(keys, mods):
            for pname, p in m.named_parameters():
                if p.requires_grad:
                    ema_state[f"{nm}.{pname}"] = p.detach().clone()
        if _loaded_ck is not None and 'ema_params' in _loaded_ck:
            ld = _loaded_ck['ema_params']; n_match = 0
            for k in list(ema_state.keys()):
                if k in ld and ld[k].shape == ema_state[k].shape:
                    ema_state[k] = ld[k].to(dev); n_match += 1
            print(f"  [ema] resumed {n_match}/{len(ema_state)} params from ckpt (decay={args.ema_decay})", flush=True)
        else:
            print(f"  [ema] initialized {len(ema_state)} params from current weights (decay={args.ema_decay})", flush=True)
    while step < args.steps:
        step += 1
        fr, st, ac, pv, tk = get_batch(args.batch_size)
        if args.state_noise > 0.0:                              # Gaussian noise on proprio
            st = st + torch.randn_like(st) * args.state_noise
        if args.dropout_prob > 0.0:                            # state dropout (per-sample)
            st = st * (torch.rand(st.shape[0], 1, device=dev) >= args.dropout_prob).float()
        with torch.no_grad():
            # normalize target by PREV chunk via DIRECT precision (Gamma-prior at global var)
            nT = ac.shape[1]
            m = pv.mean(dim=1, keepdim=True)
            S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
            lam = nT / (S + nT * var_global)            # precision (B,1,A), bounded
            xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
            gt, _ = vae.encode_with_soft(xn, tau=0.1)
        rmax = min(1.0, 0.3 + 0.7 * step / (args.steps * 0.5))
        with torch.autocast('cuda', dtype=torch.bfloat16):   # bf16: fp32 exponent range, no overflow
            vis = encode_vis(fr, tk, mod_dropout=args.dropout_prob)
            loss, per, _ = policy.forward_loss(gt, vis, st, n_inner=args.L_inner,
                                               h_max=args.h_max, mask_ratio_max=rmax)
        loss.backward()
        for g in opt.param_groups: g['lr'] = args.lr * min(1.0, step / warmup)
        opt.step(); opt.zero_grad(set_to_none=True)
        if ema_state is not None:                        # Polyak/EMA update on trainable params
            d = args.ema_decay
            with torch.no_grad():
                for nm, m in zip(keys, mods):
                    for pname, p in m.named_parameters():
                        if p.requires_grad:
                            ema_state[f"{nm}.{pname}"].mul_(d).add_(p.detach(), alpha=1 - d)
        # annealed ρ revival: if a loop saturates near 0, reset it to the
        # high-gradient zone (+ clear its momentum) so it can re-prove useful;
        # the revive peak anneals so persistent rejection dies off smoothly.
        if args.revive:
            with torch.no_grad():
                for nm, raw in (('L', policy.rho_L_raw), ('H', policy.rho_H_raw)):
                    if revive_cd[nm] > 0:
                        revive_cd[nm] -= 1; continue
                    revive_dead[nm] = revive_dead[nm] + 1 if torch.sigmoid(raw).item() < args.revive_thresh else 0
                    if revive_dead[nm] >= args.revive_patience:
                        revive_dead[nm] = 0; revive_cd[nm] = args.revive_cooldown
                        rt = revive_peak[nm]
                        if rt <= args.revive_thresh:
                            continue                       # annealed out → let it stay dead
                        raw.data.fill_(math.log(rt / (1 - rt))); opt.state.pop(raw, None)
                        revive_peak[nm] = rt * args.revive_decay
                        print(f"  [revive] ρ_{nm} → {rt:.3f} (next {revive_peak[nm]:.3f}) @ step {step}", flush=True)
        wl += loss.item(); wn += 1
        for l in range(len(seq_lens)):
            wc += per[l]['mask_correct']; wa += per[l]['mask_total']
        if step % args.log_every == 0 or step == 1:
            print(f"  step {step:>5}/{args.steps}  loss={wl/wn:.3f}  "
                  f"acc={wc/max(wa,1)*100:.1f}%  rmax={rmax:.2f}  "
                  f"ρ_L={torch.sigmoid(policy.rho_L_raw).item():.3f} "
                  f"ρ_H={torch.sigmoid(policy.rho_H_raw).item():.3f}  "
                  f"[{time.perf_counter()-t0:.0f}s]", flush=True)
            wl = wc = wa = wn = 0
        if step % args.ckpt_every == 0:
            save(step)
    save(step)
    print(f"saved {args.ckpt_path} @ step {step}", flush=True)


if __name__ == '__main__':
    main()
