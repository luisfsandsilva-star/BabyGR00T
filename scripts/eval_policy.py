#!/usr/bin/env python3
"""H-scaling evaluation for an S-TRM checkpoint.

Reports per-level top-1 / top-5 / top-10 codebook accuracy AND action-space MSE
(after CQ-VAE decode), all on the cold all-masked single forward — the
hardest evaluation regime. The L_inner used at training time is fixed; H is
swept across H_GRID to confirm test-time scaling holds.

mse_policy = pred_decoded vs gt_decoded         (isolates policy error)
mse_full   = pred_decoded vs raw GT             (deployment-level error)
mse_vae    = gt_decoded   vs raw GT             (VAE reconstruction floor)

Usage:
  python -m scripts.eval_policy [ckpt_path] [n_samples]
"""
import os, sys, time, random, argparse

THIS = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(THIS))

import torch

from babygroot_strm import (RevIN, ActionRQUNet1d, VQ1d_EMA,
                            LayerAggregator, PerceiverResampler,
                            STRMPolicy,
                            load_so101_episodes, load_lerobot_episodes,
                            make_loader,
                            NUM_RESAMPLER_LATENTS, VIS_HIDDEN_DIM, SEQ_LENS_1D)


H_GRID = [1, 2, 4, 6, 8, 12]
TOPK   = [1, 5, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', nargs='?', default='so101_strm.pt')
    ap.add_argument('n_samples', nargs='?', type=int, default=64)
    ap.add_argument('--vae-ckpt', default='so101_vae_revin.pt')
    ap.add_argument('--cache-dir', default='vision_cache')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading {args.ckpt} ...")
    c = torch.load(args.ckpt, map_location=device, weights_only=False)
    trained_L     = c.get('L_inner', 5)
    trained_H     = c.get('H_outer', 4)
    trained_depth = c.get('depth',   2)
    trained_dim   = c.get('dim',     768)
    rho1 = c.get('rho1', 0.75); rho2 = c.get('rho2', 0.65); rho_H = c.get('rho_H', 0.85)
    state_dim = c.get('state_dim', c.get('action_dim', 6))
    print(f"  trained: depth={trained_depth} dim={trained_dim} L={trained_L} "
          f"H={trained_H} ρ1={rho1} ρ2={rho2} ρ_H={rho_H}  state_dim={state_dim}")

    print(f"Loading frozen CQ-VAE from {args.vae_ckpt} ...")
    vck = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    action_dim = vck.get('action_dim', 6)
    vae   = ActionRQUNet1d(action_dim=action_dim, vq_cls=VQ1d_EMA).to(device)
    revin = RevIN(action_dim).to(device)
    vae.load_state_dict(vck['vae']); revin.load_state_dict(vck['revin']); vae.eval(); revin.eval()

    aggregator = LayerAggregator(hidden_dim=VIS_HIDDEN_DIM, n_layers=25).to(device)
    resampler  = PerceiverResampler(input_dim=VIS_HIDDEN_DIM, dim=trained_dim,
                                    num_latents=NUM_RESAMPLER_LATENTS).to(device)
    policy = STRMPolicy(
        seq_lens=tuple(SEQ_LENS_1D), k_codebook=vae.vq1.K,
        dim=trained_dim, heads=8, depth=trained_depth,
        L_inner=trained_L, H_outer=trained_H,
        rho1_target=rho1, rho2_target=rho2, rho_H_target=rho_H,
        max_prefix=NUM_RESAMPLER_LATENTS + 16, state_dim=state_dim,
    ).to(device)
    aggregator.load_state_dict(c['aggregator'])
    resampler.load_state_dict(c['resampler'])
    policy.load_state_dict(c['policy'])
    policy.eval(); aggregator.eval(); resampler.eval()
    print(f"  step={c.get('step', '?')}  best_acc={c.get('best_acc', 0)*100:.1f}%")

    # Pick the right episode loader from the ckpt's dataset tag.
    ds_tag = c.get('dataset', 'so101')
    oxe_id = c.get('oxe_dataset_id') or 'IPEC-COMMUNITY/bridge_orig_lerobot'
    print(f"Loading episodes ({ds_tag}) ...")
    if ds_tag == 'oxe':
        episodes = load_lerobot_episodes(oxe_id, load_video=False)
    else:
        episodes = load_so101_episodes(load_video=False)
    loader = make_loader(args.cache_dir, episodes,
                         batch_size=1, num_workers=0, shuffle=True,
                         lru_size=2, augment=False)

    def vis_pipeline(hidden):
        layers = [hidden[:, l] for l in range(hidden.shape[1])]
        return resampler(aggregator(layers))

    def decode_action(idxs):
        i_L0, i_L1, i_L2 = idxs
        B = i_L0.shape[0]
        e3 = vae.vq3.emb(i_L0).view(B, SEQ_LENS_1D[0], vae.vq3.D).permute(0, 2, 1)
        e2 = vae.vq2.emb(i_L1).view(B, SEQ_LENS_1D[1], vae.vq2.D).permute(0, 2, 1)
        e1 = vae.vq1.emb(i_L2).view(B, SEQ_LENS_1D[2], vae.vq1.D).permute(0, 2, 1)
        return vae.decode([e3, e2, e1]).transpose(1, 2)

    def topk_correct(logits_l, gt_l, k):
        topk = logits_l.topk(k, dim=-1).indices
        return (topk == gt_l.unsqueeze(-1)).any(-1).sum().item()

    random.seed(42); torch.manual_seed(42)
    N = min(args.n_samples, len(loader.dataset))
    sample_ids = random.sample(range(len(loader.dataset)), N)

    print(f"Pre-encoding {N} samples (vis + GT codes + GT-decoded action) ...")
    cached = []
    with torch.no_grad():
        for idx in sample_ids:
            s = loader.dataset[idx]
            hidden = s['hidden'].unsqueeze(0).to(device)
            state  = s['state'].unsqueeze(0).to(device)
            action = s['action'].unsqueeze(0).to(device)
            vis = vis_pipeline(hidden)
            x_norm = revin(action, 'norm').transpose(1, 2)
            gt, _ = vae.encode_with_soft(x_norm, tau=0.1)
            gt_recon = decode_action(gt)
            action_norm_chunk = x_norm.transpose(1, 2)
            cached.append({
                'vis': vis.detach(), 'state': state.detach(),
                'gt': gt, 'gt_recon': gt_recon,
                'action_norm': action_norm_chunk,
            })

    def eval_at(L, H):
        topk_correct_per_lvl = {l: {k: 0 for k in TOPK} for l in range(len(SEQ_LENS_1D))}
        total_per_lvl = [0] * len(SEQ_LENS_1D)
        mse_pol = mse_full = mse_vae = 0.0; n_chunks = 0
        with torch.no_grad():
            for s in cached:
                all_logits = policy(None, s['vis'], s['state'], mask_list=None,
                                    n_outer=H, n_inner=L)
                final = all_logits[-1]
                for l, T_l in enumerate(SEQ_LENS_1D):
                    for k in TOPK:
                        topk_correct_per_lvl[l][k] += topk_correct(final[l], s['gt'][l], k)
                    total_per_lvl[l] += T_l
                pred_idx = [final[l].argmax(-1) for l in range(len(SEQ_LENS_1D))]
                pred_recon = decode_action(pred_idx)
                mse_pol  += ((pred_recon - s['gt_recon']) ** 2).mean().item()
                mse_full += ((pred_recon - s['action_norm']) ** 2).mean().item()
                mse_vae  += ((s['gt_recon'] - s['action_norm']) ** 2).mean().item()
                n_chunks += 1
        topk_acc = {l: {k: topk_correct_per_lvl[l][k] / max(total_per_lvl[l], 1) for k in TOPK}
                    for l in range(len(SEQ_LENS_1D))}
        return {'topk': topk_acc,
                'mse_policy': mse_pol / n_chunks,
                'mse_full':   mse_full / n_chunks,
                'mse_vae':    mse_vae / n_chunks}

    print(f"\nH-scaling eval on {N} samples (L fixed at trained={trained_L})\n")
    print("=" * 92)
    print(f"  {'H':>3}  {'time':>5}   "
          f"{'L0 top1/5/10':>16}  {'L1 top1/5/10':>16}  {'L2 top1/5/10':>16}  "
          f"{'mse_pol':>8}  {'mse_full':>8}")
    print("-" * 92)

    results = []
    for H in H_GRID:
        t0 = time.perf_counter()
        r = eval_at(trained_L, H)
        elapsed = time.perf_counter() - t0
        cells = ["/".join(f"{r['topk'][l][k]*100:.0f}" for k in TOPK)
                 for l in range(len(SEQ_LENS_1D))]
        marker = "  ←train" if H == trained_H else ""
        print(f"  {H:>3}  {elapsed:>5.1f}s   "
              f"{cells[0]:>16}  {cells[1]:>16}  {cells[2]:>16}  "
              f"{r['mse_policy']:>8.4f}  {r['mse_full']:>8.4f}{marker}")
        results.append((H, r))

    print("=" * 92)
    print(f"\n  VAE recon MSE (gt_decoded vs raw gt_action): "
          f"{results[0][1]['mse_vae']:.4f}  (lower bound for mse_full)")
    best = min(results, key=lambda x: x[1]['mse_policy'])
    H, r = best
    print(f"\n  Best by mse_policy: H={H}, mse_pol={r['mse_policy']:.4f}, "
          f"mse_full={r['mse_full']:.4f}")
    print(f"     top-1 mean: {sum(r['topk'][l][1] for l in range(3))/3*100:.1f}%   "
          f"top-5 mean: {sum(r['topk'][l][5] for l in range(3))/3*100:.1f}%   "
          f"top-10 mean: {sum(r['topk'][l][10] for l in range(3))/3*100:.1f}%")


if __name__ == '__main__':
    main()
