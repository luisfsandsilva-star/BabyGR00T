#!/usr/bin/env python3
"""Train the single-bottleneck VQ-VAE on action chunks.

Direct comparison baseline for the 3-level CQ-VAE — identical encoder, a
single codebook at the bottleneck (4 tokens × 256 ch × K=128), no
decoder skip connections. See babygroot_strm/vqvae.py for the design.

Usage (Bridge V2, full action stream):
  python -m scripts.train_vqvae \\
      --dataset oxe --action-dim 7 \\
      --steps 8000 --batch-size 32 \\
      --ckpt-path oxe_vqvae.pt

Usage (SO-101, original 78-ep set):
  python -m scripts.train_vqvae --steps 5000 --batch-size 32
"""
import os, sys, time, argparse

THIS = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(THIS))

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from babygroot_strm import (RevIN, ActionVQVAE1d, VQ1d_EMA,
                            load_so101_episodes, load_lerobot_episodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--ckpt-path', type=str, default='vqvae.pt')
    ap.add_argument('--log-every', type=int, default=200)
    ap.add_argument('--norm-reg', choices=['eps', 'global'], default='global',
                    help="precision regularizer: 'eps' (tiny, ~1/var, blows up on still chunks) "
                         "or 'global' (Gamma-prior at global variance — bounded).")
    ap.add_argument('--action-dim', type=int, default=6,
                    help="Bridge=7, SO-101=6. Must match the dataset.")
    ap.add_argument('--dataset', choices=['so101', 'oxe'], default='so101')
    ap.add_argument('--oxe-dataset-id', type=str,
                    default='IPEC-COMMUNITY/bridge_orig_lerobot')
    ap.add_argument('--n-eps-cap', type=int, default=None,
                    help="Cap dataset episodes. Default = all available.")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    print(f"Loading episodes ({args.dataset}, no video, cap={args.n_eps_cap}) ...")
    if args.dataset == 'oxe':
        eps = load_lerobot_episodes(args.oxe_dataset_id, load_video=False,
                                     n_episodes=args.n_eps_cap)
    else:
        eps = load_so101_episodes(load_video=False)
        if args.n_eps_cap:
            eps = eps[:args.n_eps_cap]
    # pair each chunk with its PREVIOUS chunk (within episode) — RevIN lookback
    cur_list, prev_list = [], []
    for ep in eps:
        ac = ep[0]                                                  # (n_ch, T, A)
        if ac.shape[0] < 2:
            continue
        cur_list.append(ac[1:]); prev_list.append(ac[:-1])
    actions = torch.cat(cur_list, dim=0); prevs = torch.cat(prev_list, dim=0)
    var_global = actions.var(dim=(0, 1)).clamp(min=1e-8).to(device).view(1, 1, -1)  # per-dim global variance
    print(f"  {len(actions)} (chunk, prev-chunk) pairs, action_dim={actions.shape[-1]}")
    print(f"  global std (per-dim): {[round(x,4) for x in var_global.sqrt().flatten().tolist()]}  | norm-reg={args.norm_reg}")

    vae   = ActionVQVAE1d(action_dim=args.action_dim, vq_cls=VQ1d_EMA).to(device)
    revin = RevIN(args.action_dim).to(device)
    n_p = sum(p.numel() for p in vae.parameters()) / 1e6
    print(f"  VQ-VAE params: {n_p:.2f}M  "
          f"bottleneck: {vae.bottleneck_T} tokens × {vae.vq.D} ch, K={vae.vq.K}")

    opt = torch.optim.AdamW(list(vae.parameters()) + list(revin.parameters()),
                            lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(actions, prevs), batch_size=args.batch_size,
                        shuffle=True, drop_last=True)
    print(f"  {len(loader)} batches/epoch")

    vae.train(); revin.train()
    t0 = time.perf_counter(); step = 0
    win_recon, win_vq, win_steps = 0.0, 0.0, 0
    while step < args.steps:
        for (action, prev) in loader:
            if step >= args.steps:
                break
            action = action.to(device, non_blocking=True)
            prev = prev.to(device, non_blocking=True)
            # RevIN-style, but compute PRECISION directly (Gamma-posterior mean) and MULTIPLY:
            #   λ = n / (Σ(prev−m)² + reg)   — bounded since reg sits in the denominator.
            # reg='eps' (tiny → ~1/var, blows up on still chunks) vs 'global' (n·var_global → prior at global precision).
            n = action.shape[1]
            m = prev.mean(dim=1, keepdim=True)
            S = ((prev - m) ** 2).sum(dim=1, keepdim=True)
            reg = 1e-6 if args.norm_reg == 'eps' else n * var_global
            lam = n / (S + reg)                                  # precision (B,1,A)
            x = ((action - m) * lam.sqrt()).transpose(1, 2)
            embs, vql, _ = vae.encode(x)
            recon = vae.decode(embs)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + vql
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            win_recon += recon_loss.item(); win_vq += float(vql); win_steps += 1
            step += 1
            if step % args.log_every == 0 or step == 1:
                elapsed = time.perf_counter() - t0
                print(f"  step {step:>6}/{args.steps}  recon={win_recon/win_steps:.4f}  "
                      f"vq={win_vq/win_steps:.4f}  [{elapsed:.0f}s]", flush=True)
                win_recon, win_vq, win_steps = 0.0, 0.0, 0

    torch.save({'kind': 'vqvae',
                'vae': vae.state_dict(), 'revin': revin.state_dict(),
                'action_dim': args.action_dim,
                'action_var_global': var_global.squeeze().cpu(), 'norm_reg': args.norm_reg,
                'seq_lens': vae.seq_lens, 'k': vae.vq.K},
               args.ckpt_path)
    print(f"\nSaved {args.ckpt_path}")


if __name__ == '__main__':
    main()
