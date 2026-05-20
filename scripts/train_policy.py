#!/usr/bin/env python3
"""Train the S-TRM v3 policy on cached InternVL3 features.

Pipeline:
  cached hidden (B, 25, N_tok, 896)
    → LayerAggregator         (B, N_tok, 896)
    → PerceiverResampler      (B, 128, dim)
    → STRMPolicy              split-stream Parcae recurrence with deep
                                supervision over H outer cycles
  CQ-VAE (frozen) provides per-level (hard, soft) targets via encode_with_soft.

Default recipe:
  --depth 2 --dim 768 --L-inner 5 --H-outer 4 --h-max 12
  --gamma-L 0.7 --gamma-H 0.7    (additive geometric-decay TRM updates)
  --mask-curriculum --mask-curriculum-init 0.3 --mask-curriculum-frac 0.5
  --no-snce --tau-anneal-frac 0.4 --lr 9.5e-4

Usage:
  python -m scripts.train_policy --steps 25000
"""
import os, sys, time, signal, random, math, argparse

THIS = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(THIS))

import numpy as np
import torch

import torch.nn.functional as F

from babygroot_strm import (RevIN, ActionRQUNet1d, ActionVQVAE1d, VQ1d_EMA,
                            LayerAggregator, PerceiverResampler,
                            STRMPolicy, MuSGD_LARS,
                            load_so101_episodes, load_lerobot_episodes,
                            make_loader, cosine_snce_tau,
                            NUM_RESAMPLER_LATENTS, VIS_HIDDEN_DIM,
                            SEQ_LENS_1D)


def main():
    ap = argparse.ArgumentParser()
    # Schedule
    ap.add_argument('--steps', type=int, default=25000)
    ap.add_argument('--lr', type=float, default=9.5e-4)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=2)
    # Logging / ckpt
    ap.add_argument('--log-every', type=int, default=200)
    ap.add_argument('--probe-every', type=int, default=1000)
    ap.add_argument('--n-probe', type=int, default=64)
    ap.add_argument('--ckpt-every', type=int, default=1000)
    ap.add_argument('--ckpt-path', type=str, default='so101_strm.pt')
    ap.add_argument('--vae-ckpt', type=str, default='so101_vae_revin.pt')
    ap.add_argument('--cache-dir', type=str, default='vision_cache')
    # Architecture (v3 defaults)
    ap.add_argument('--depth',  type=int,   default=2)
    ap.add_argument('--dim',    type=int,   default=768)
    ap.add_argument('--L-inner', type=int,  default=5)
    ap.add_argument('--H-outer', type=int,  default=4)
    ap.add_argument('--rho-L', type=float, default=0.1,
                    help="Initial inner-loop decay rate; closed-form weights "
                         "a_t = ρ_L^(t/(L-1)). Single learnable scalar in (0,1).")
    ap.add_argument('--rho-H', type=float, default=0.1,
                    help="Initial outer-loop decay rate; closed-form weights "
                         "a_h = ρ_H^(h/(H-1)). Single learnable scalar in (0,1).")
    ap.add_argument('--h-max',  type=int,   default=12,
                    help="Train with H ~ U{1..h_max} per call. Unlocks "
                         "test-time scaling beyond the training depth.")
    # Loss / curriculum
    ap.add_argument('--no-snce', action='store_true', default=True,
                    help="Use plain CE on hard targets. Default for v3.")
    ap.add_argument('--snce',     dest='no_snce', action='store_false',
                    help="Use SNCE soft-target CE instead of plain CE.")
    ap.add_argument('--tau-min', type=float, default=0.1)
    ap.add_argument('--tau-max', type=float, default=2.0)
    ap.add_argument('--tau-anneal-frac', type=float, default=0.4)
    ap.add_argument('--mask-curriculum', action='store_true', default=True)
    ap.add_argument('--no-mask-curriculum', dest='mask_curriculum',
                    action='store_false')
    ap.add_argument('--mask-curriculum-init', type=float, default=0.3)
    ap.add_argument('--mask-curriculum-frac', type=float, default=0.5)
    ap.add_argument('--no-augment', action='store_true')
    # MSE-decode auxiliary loss (v4 — corrected recipe).
    # The post-mortem on the original v4 (β=0.1, argmax-STE, final-cycle only)
    # was: argmax-STE early in training is gradient-noisy, and final-cycle-only
    # MSE biases the model against iterative refinement (best-H collapsed to 1).
    # The corrected recipe keeps the action-space signal but smooths it:
    #   - expectation mode (gradient and forward agree exactly)
    #   - all H cycles (preserves H-cycle deep supervision)
    #   - smaller β (don't dominate the codebook CE)
    ap.add_argument('--mse-decode-weight', type=float, default=0.0,
                    help="If > 0, add β·MSE on the action-space decode of the "
                         "policy's prediction. Frozen CQ-VAE decoder provides "
                         "the action-space gradient. v4 default-when-on: 0.05.")
    ap.add_argument('--mse-decode-mode', choices=['expectation', 'argmax'],
                    default='expectation',
                    help="How to build e_pred from logits. 'expectation' = "
                         "softmax @ codebook (smooth); 'argmax' = STE forward "
                         "= codebook[argmax], backward = soft-mixture gradient.")
    ap.add_argument('--mse-decode-cycles', choices=['all', 'final'],
                    default='all',
                    help="Apply the aux loss to every H cycle (preserves deep "
                         "supervision) or just the last (the v4 anti-pattern).")
    # Dataset selection (v5 — OXE migration).
    # 'so101' : the original 78-episode SO-101 set (pavelsimo + supplements).
    # 'oxe'   : a single LeRobot OXE dataset. Default = BridgeData V2
    #           (IPEC-COMMUNITY/bridge_orig_lerobot, WidowX 7-DoF + 8-dim state).
    #           The CQ-VAE must be retrained on the dataset's action distribution
    #           (--action-dim must match: 7 for Bridge, 6 for SO-101).
    ap.add_argument('--dataset', choices=['so101', 'oxe'], default='so101')
    ap.add_argument('--oxe-dataset-id', type=str,
                    default='IPEC-COMMUNITY/bridge_orig_lerobot')
    ap.add_argument('--oxe-camera', type=str, default='observation.images.image_0')
    ap.add_argument('--state-dim', type=int, default=None,
                    help="State dim for the StateEncoder. If unset, inferred "
                         "from the first episode (Bridge=8, SO-101=6).")
    # Resume
    ap.add_argument('--resume', type=str, default=None,
                    help="Path to ckpt. Restores weights + optimizer + step + best_acc.")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Frozen action codebook + RevIN (CQ-VAE or VQ-VAE) ──
    print(f"Loading frozen VAE from {args.vae_ckpt} ...", flush=True)
    ck = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    action_dim = ck.get('action_dim', 6)
    vae_kind   = ck.get('kind', 'cqvae')
    if vae_kind == 'vqvae':
        vae = ActionVQVAE1d(action_dim=action_dim, vq_cls=VQ1d_EMA).to(device)
    else:
        vae = ActionRQUNet1d(action_dim=action_dim, vq_cls=VQ1d_EMA).to(device)
    revin = RevIN(action_dim).to(device)
    vae.load_state_dict(ck['vae']); revin.load_state_dict(ck['revin']); vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    revin.eval()
    seq_lens = tuple(vae.seq_lens)
    K = vae.vqs[0].K
    print(f"  VAE: kind={vae_kind}  levels={seq_lens}  K={K}  action_dim={action_dim}",
          flush=True)

    # ── Load episodes early so we can infer state_dim from data ──
    # IMPORTANT: cap to the number of episodes that actually have vision-cache
    # files. The cache's meta.json carries that number (set when cache_vision
    # was run). Without this cap, ChunkDataset indexes all 53k Bridge eps and
    # crashes the first time DataLoader samples one that wasn't cached.
    import json
    cache_meta_path = os.path.join(args.cache_dir, 'meta.json')
    cache_n_episodes = None
    if os.path.exists(cache_meta_path):
        with open(cache_meta_path) as f:
            cache_n_episodes = int(json.load(f).get('n_episodes') or 0)
        if cache_n_episodes > 0:
            print(f"  cache contains {cache_n_episodes} episodes — capping "
                  f"dataset load to match.", flush=True)

    print(f"Loading episodes ({args.dataset}, cached features) ...", flush=True)
    if args.dataset == 'oxe':
        episodes = load_lerobot_episodes(args.oxe_dataset_id,
                                          camera_key=args.oxe_camera,
                                          load_video=False,
                                          n_episodes=cache_n_episodes)
    else:
        episodes = load_so101_episodes(load_video=False)
        if cache_n_episodes:
            episodes = episodes[:cache_n_episodes]
    inferred_state_dim = int(episodes[0][1].shape[-1])
    state_dim = args.state_dim if args.state_dim is not None else inferred_state_dim
    print(f"  state_dim = {state_dim}  (inferred {inferred_state_dim}; "
          f"CLI override = {args.state_dim})", flush=True)

    # ── Vision pipeline + S-TRM policy ──
    aggregator = LayerAggregator(hidden_dim=VIS_HIDDEN_DIM, n_layers=25).to(device)
    resampler  = PerceiverResampler(input_dim=VIS_HIDDEN_DIM, dim=args.dim,
                                    num_latents=NUM_RESAMPLER_LATENTS).to(device)
    policy = STRMPolicy(
        seq_lens=seq_lens, k_codebook=K,
        dim=args.dim, heads=8, depth=args.depth,
        L_inner=args.L_inner, H_outer=args.H_outer,
        rho_L=args.rho_L, rho_H=args.rho_H,
        max_prefix=NUM_RESAMPLER_LATENTS + 16, state_dim=state_dim,
    ).to(device)

    n_pol = sum(p.numel() for p in policy.parameters()) / 1e6
    n_agg = sum(p.numel() for p in aggregator.parameters()) / 1e6
    n_res = sum(p.numel() for p in resampler.parameters()) / 1e6
    h_str = f"H~U{{1..{args.h_max}}}" if args.h_max else f"H={args.H_outer} fixed"
    loss_kind = "CE" if args.no_snce else "SNCE"
    mc = (f"mask-curriculum {args.mask_curriculum_init}→1.0 over "
          f"{args.mask_curriculum_frac*100:.0f}%"
          if args.mask_curriculum else "uniform [1/T, 1.0]")
    print(f"  Aggregator: {n_agg:.2f}M  Resampler: {n_res:.2f}M  "
          f"Policy: {n_pol:.2f}M", flush=True)
    print(f"  TRM: depth={args.depth}  dim={args.dim}  L={args.L_inner}  "
          f"{h_str}  ρ_L0={args.rho_L} ρ_H0={args.rho_H} (learnable)  "
          f"additive closed-form updates (a_t=ρ^(t/(n-1)))", flush=True)
    aux_str = ""
    if args.mse_decode_weight > 0:
        aux_str = (f" + β={args.mse_decode_weight}·MSE_decode("
                   f"{args.mse_decode_mode}, {args.mse_decode_cycles})")
    print(f"  Optim: lr={args.lr}  loss={loss_kind}{aux_str}  "
          f"tau {args.tau_max}→{args.tau_min} over {args.tau_anneal_frac*100:.0f}%  |  {mc}",
          flush=True)

    # ── Data loader ──
    loader = make_loader(args.cache_dir, episodes,
                         batch_size=args.batch_size,
                         num_workers=args.num_workers,
                         shuffle=True, lru_size=2,
                         augment=not args.no_augment,
                         dropout=0.1 if not args.no_augment else 0.0)
    print(f"  {len(loader.dataset)} samples, {len(loader)} batches/epoch", flush=True)

    # ── Optimizer ──
    trainable = (list(aggregator.parameters()) + list(resampler.parameters())
                 + list(policy.parameters()))
    opt = MuSGD_LARS(trainable, lr=args.lr, momentum=0.95, weight_decay=1e-4,
                     nesterov=True, ns_steps=5)
    warmup_steps = min(200, args.steps // 10)

    # ── Resume ──
    resume_step = 0; resume_best_acc = 0.0
    if args.resume:
        print(f"Resuming from {args.resume} ...", flush=True)
        rck = torch.load(args.resume, map_location=device, weights_only=False)
        aggregator.load_state_dict(rck['aggregator'])
        resampler.load_state_dict(rck['resampler'])
        policy.load_state_dict(rck['policy'])
        if 'opt' in rck:
            opt.load_state_dict(rck['opt'])
            print("  optimizer state restored")
        else:
            print("  WARNING: no optimizer state — using fresh momentum")
        resume_step = rck.get('step', 0)
        resume_best_acc = rck.get('best_acc', 0.0)
        print(f"  resuming at step {resume_step}/{args.steps}  "
              f"best_acc={resume_best_acc*100:.1f}%")

    def vis_pipeline(hidden):
        layer_list = [hidden[:, l] for l in range(hidden.shape[1])]
        return resampler(aggregator(layer_list))

    def encode_codes(action, tau):
        with torch.no_grad():
            x = revin(action, 'norm').transpose(1, 2)
            return vae.encode_with_soft(x, tau=tau)

    def mask_ratio_max_at(step):
        if not args.mask_curriculum:
            return 1.0
        frac = max(1, int(args.steps * args.mask_curriculum_frac))
        progress = min(1.0, step / frac)
        return args.mask_curriculum_init + (1.0 - args.mask_curriculum_init) * \
               0.5 * (1 - math.cos(math.pi * progress))

    def run_probe(n_probe):
        """All-masked accuracy on n_probe random samples (final-cycle preds)."""
        policy.eval(); aggregator.eval(); resampler.eval()
        correct_per_lvl = [0] * len(seq_lens)
        total_per_lvl   = [0] * len(seq_lens)
        idxs = random.sample(range(len(loader.dataset)),
                             min(n_probe, len(loader.dataset)))
        with torch.no_grad():
            for idx in idxs:
                s = loader.dataset[idx]
                hidden = s['hidden'].unsqueeze(0).to(device)
                state  = s['state'].unsqueeze(0).to(device)
                action = s['action'].unsqueeze(0).to(device)
                vis = vis_pipeline(hidden)
                indices, _ = encode_codes(action, tau=0.1)
                all_logits = policy(None, vis, state, mask_list=None)
                final = all_logits[-1]
                for l in range(len(seq_lens)):
                    # head emits K+1 logits (last = MASK); compare real codes only
                    preds = final[l][..., :K].argmax(-1)
                    correct_per_lvl[l] += (preds == indices[l]).sum().item()
                    total_per_lvl[l]   += seq_lens[l]
        policy.train(); aggregator.train(); resampler.train()
        return [c / max(t, 1) for c, t in zip(correct_per_lvl, total_per_lvl)]

    def save_checkpoint(step, best_acc, path):
        torch.save({
            'aggregator': aggregator.state_dict(),
            'resampler':  resampler.state_dict(),
            'policy':     policy.state_dict(),
            'opt':        opt.state_dict(),
            'step': step, 'best_acc': best_acc,
            'seq_lens': list(seq_lens), 'k': K, 'vae_kind': vae_kind,
            'L_inner': args.L_inner, 'H_outer': args.H_outer,
            'depth': args.depth, 'dim': args.dim,
            'rho_L': args.rho_L, 'rho_H': args.rho_H,
            'action_dim': action_dim, 'state_dim': state_dim,
            'dataset': args.dataset,
            'oxe_dataset_id': args.oxe_dataset_id if args.dataset == 'oxe' else None,
        }, path)

    # ── Train ──
    print(f"\nTraining {args.steps} steps  ({loss_kind}) ...", flush=True)
    policy.train(); aggregator.train(); resampler.train()
    n_lvls = len(seq_lens)
    win_correct = [0] * n_lvls; win_total = [0] * n_lvls
    win_loss = 0.0; win_mse_decode = 0.0; win_steps = 0
    best_acc = resume_best_acc
    t0 = time.perf_counter(); step = resume_step

    def _save_on_exit(signum, frame):
        print(f"\n[signal {signum}] saving checkpoint at step {step}...", flush=True)
        save_checkpoint(step, best_acc, args.ckpt_path); sys.exit(0)
    signal.signal(signal.SIGINT, _save_on_exit)
    signal.signal(signal.SIGTERM, _save_on_exit)

    loader_iter = iter(loader)
    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader); batch = next(loader_iter)
        step += 1

        hidden = batch['hidden'].to(device, non_blocking=True)
        action = batch['action'].to(device, non_blocking=True)
        state  = batch['state'].to(device,  non_blocking=True)

        tau = cosine_snce_tau(step - 1, args.steps, args.tau_max, args.tau_min,
                              anneal_frac=args.tau_anneal_frac)
        indices, soft = encode_codes(action, tau)
        soft_for_loss = None if args.no_snce else soft
        vis = vis_pipeline(hidden)
        rmax = mask_ratio_max_at(step)

        loss, per_lvl, all_logits = policy.forward_loss(
            indices, vis, state,
            soft_targets=soft_for_loss,
            h_max=args.h_max,
            mask_ratio_max=rmax,
        )

        # Auxiliary loss: MSE on the action-space decode of e_pred.
        # Two modes for e_pred (`--mse-decode-mode`):
        #   'expectation': e_pred = softmax(logits) @ codebook (smooth mixture)
        #   'argmax':      forward = codebook[argmax],
        #                  backward = soft-mixture gradient (STE)
        # Cycle scope (`--mse-decode-cycles`): 'all' adds it at every H cycle
        # (preserves H-cycle deep supervision); 'final' only at the last cycle.
        # The CQ-VAE decoder is frozen — its parameters don't update, but
        # gradients pass through to the policy's logits.
        if args.mse_decode_weight > 0:
            vqs = vae.vqs    # coarsest-first; CQ-VAE→3 entries, VQ-VAE→1 entry
            a_target = revin(action, 'norm').transpose(1, 2)
            cycles = (all_logits if args.mse_decode_cycles == 'all'
                      else [all_logits[-1]])
            mse_acc = 0.0
            for cycle_logits in cycles:
                e_in = []
                for l, lg in enumerate(cycle_logits):
                    E = vqs[l].emb.weight                        # (K, D_l)
                    # policy head emits K+1 (last = MASK); the VAE codebook has
                    # only the K real codes, so softmax over the real columns.
                    lg_real = lg[..., :E.shape[0]]               # (B, T_l, K)
                    soft = torch.softmax(lg_real, dim=-1)        # (B, T_l, K)
                    e_soft = soft @ E                            # (B, T_l, D_l)
                    if args.mse_decode_mode == 'argmax':
                        idx = lg_real.argmax(-1)
                        e_hard = E[idx]
                        e_pred = e_soft + (e_hard - e_soft).detach()
                    else:
                        e_pred = e_soft
                    e_in.append(e_pred.transpose(1, 2))          # (B, D_l, T_l)
                a_pred = vae.decode(e_in)
                mse_acc = mse_acc + F.mse_loss(a_pred, a_target)
            mse_decode = mse_acc / len(cycles)
            loss = loss + args.mse_decode_weight * mse_decode
            win_mse_decode += mse_decode.item()

        (loss / args.grad_accum).backward()

        for l in range(n_lvls):
            win_correct[l] += per_lvl[l]['mask_correct']
            win_total[l]   += per_lvl[l]['mask_total']
        win_loss += loss.item(); win_steps += 1

        if step % args.grad_accum == 0:
            lr_scale = min(1.0, (step / args.grad_accum + 1) / max(warmup_steps, 1))
            for g in opt.param_groups:
                g['lr'] = args.lr * lr_scale
            opt.step(); opt.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            cur_lr = opt.param_groups[0]['lr']
            mem = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0.0
            per = [(win_correct[l] / win_total[l]) if win_total[l] > 0 else 0.0
                   for l in range(n_lvls)]
            mean_acc = (sum(win_correct) / sum(win_total)) if sum(win_total) > 0 else 0.0
            with torch.no_grad():
                rL, rH = policy._rhos()
            mse_dec_str = (f"  mse_dec={win_mse_decode/win_steps:.3f}"
                           if args.mse_decode_weight > 0 else "")
            print(f"  step {step:>6}/{args.steps}  loss={win_loss/win_steps:.3f}{mse_dec_str}  "
                  f"acc={mean_acc*100:.1f}% [L0/L1/L2={'/'.join(f'{a*100:.1f}' for a in per)}]  "
                  f"tau={tau:.3f}  rmax={rmax:.2f}  lr={cur_lr:.1e}  "
                  f"ρ_L={rL.item():.3f} ρ_H={rH.item():.3f}  "
                  f"mem={mem:.1f}GB  [{elapsed:.0f}s]", flush=True)
            win_correct = [0] * n_lvls; win_total = [0] * n_lvls
            win_loss = 0.0; win_mse_decode = 0.0; win_steps = 0

        if step % args.probe_every == 0 or step == args.steps:
            per_lvl_p = run_probe(args.n_probe)
            mean_p = sum(per_lvl_p) / len(per_lvl_p)
            best_acc = max(best_acc, mean_p)
            print(f"  probe: mean={mean_p*100:.1f}% "
                  f"[{'/'.join(f'{p*100:.1f}' for p in per_lvl_p)}]  "
                  f"(best={best_acc*100:.1f}%)", flush=True)

        if step % args.ckpt_every == 0 or step == args.steps:
            save_checkpoint(step, best_acc, args.ckpt_path)

    save_checkpoint(args.steps, best_acc, args.ckpt_path)
    print(f"\nDone. Best probe: {best_acc*100:.1f}%  Saved: {args.ckpt_path}",
          flush=True)


if __name__ == '__main__':
    main()
