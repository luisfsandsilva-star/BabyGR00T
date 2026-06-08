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
import os, sys, time, math, random, argparse

THIS = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(THIS))

import torch

from babygroot_strm import (RevIN, ActionRQUNet1d, ActionVQVAE1d, VQ1d_EMA,
                            LayerAggregator, PerceiverResampler,
                            STRMPolicy, STRMPolicyVAE,
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
    ap.add_argument('--diagnostics', action='store_true',
                    help="Print interpretability diagnostics at the trained "
                         "(L,H): majority-class baseline + lift, codebook usage "
                         "& entropy (GT vs predicted), prediction confidence, and "
                         "per-action-dimension MSE vs the VAE-recon floor.")
    ap.add_argument('--rho-h-sweep', type=str, default='',
                    help="Space-separated ρ_H values to sweep, e.g. "
                         "\"0.1 0.2 0.4 0.6\". Each is forced (override the "
                         "trained/collapsed ρ_H) and the full H_GRID is run. "
                         "Empty = use the checkpoint's trained ρ_H.")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading {args.ckpt} ...")
    c = torch.load(args.ckpt, map_location=device, weights_only=False)
    trained_L     = c.get('L_inner', 5)
    trained_H     = c.get('H_outer', 4)
    trained_depth = c.get('depth',   2)
    trained_dim   = c.get('dim',     768)
    trained_vis_dim = c.get('vis_dim', trained_dim)  # resampler width; <dim => vis_proj up
    rho_L = c.get('rho_L', 0.1); rho_H = c.get('rho_H', 0.1)
    state_dim = c.get('state_dim', c.get('action_dim', 6))
    is_vae = c.get('vae_latent', False)
    beta = c.get('beta', 1e-3); free_bits = c.get('free_bits', 0.1)
    print(f"  trained: depth={trained_depth} dim={trained_dim} L={trained_L} "
          f"H={trained_H} ρ_L={rho_L} ρ_H={rho_H}  state_dim={state_dim}  "
          f"vae_latent={is_vae}")

    print(f"Loading frozen VAE from {args.vae_ckpt} ...")
    vck = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    action_dim = vck.get('action_dim', 6)
    vae_kind   = vck.get('kind', 'cqvae')
    if vae_kind == 'vqvae':
        vae = ActionVQVAE1d(action_dim=action_dim, vq_cls=VQ1d_EMA).to(device)
    else:
        vae = ActionRQUNet1d(action_dim=action_dim, vq_cls=VQ1d_EMA).to(device)
    revin = RevIN(action_dim).to(device)
    vae.load_state_dict(vck['vae']); revin.load_state_dict(vck['revin']); vae.eval(); revin.eval()
    seq_lens = tuple(vae.seq_lens)
    K = vae.vqs[0].K
    print(f"  VAE: kind={vae_kind}  levels={seq_lens}  K={K}")

    aggregator = LayerAggregator(hidden_dim=VIS_HIDDEN_DIM, n_layers=25).to(device)
    resampler  = PerceiverResampler(input_dim=VIS_HIDDEN_DIM, dim=trained_vis_dim,
                                    num_latents=NUM_RESAMPLER_LATENTS).to(device)
    # Match train_policy: thin Linear(vis_dim -> dim) when the resampler is
    # narrower than the TRM (the 100M Bridge config); Identity otherwise.
    vis_proj = (torch.nn.Identity() if trained_vis_dim == trained_dim
                else torch.nn.Linear(trained_vis_dim, trained_dim)).to(device)
    PolicyCls = STRMPolicyVAE if is_vae else STRMPolicy
    extra = dict(beta=beta, free_bits=free_bits) if is_vae else {}
    policy = PolicyCls(
        seq_lens=seq_lens, k_codebook=K,
        dim=trained_dim, heads=8, depth=trained_depth,
        L_inner=trained_L, H_outer=trained_H,
        rho_L=rho_L, rho_H=rho_H,
        max_prefix=NUM_RESAMPLER_LATENTS + 16, state_dim=state_dim,
        **extra,
    ).to(device)
    aggregator.load_state_dict(c['aggregator'])
    resampler.load_state_dict(c['resampler'])
    if c.get('vis_proj') is not None and not isinstance(vis_proj, torch.nn.Identity):
        vis_proj.load_state_dict(c['vis_proj'])
    policy.load_state_dict(c['policy'])
    policy.eval(); aggregator.eval(); resampler.eval(); vis_proj.eval()
    trained_rhoH_raw = policy.rho_H_raw.data.clone()  # learned value (for restore after a sweep)
    print(f"  step={c.get('step', '?')}  best_acc={c.get('best_acc', 0)*100:.1f}%")

    # Pick the right episode loader from the ckpt's dataset tag. Cap to the
    # number of episodes actually present in the vision cache (same reason as
    # train_policy: avoid indexing eps whose cache file doesn't exist).
    import json
    ds_tag = c.get('dataset', 'so101')
    oxe_id = c.get('oxe_dataset_id') or 'IPEC-COMMUNITY/bridge_orig_lerobot'
    cache_meta_path = os.path.join(args.cache_dir, 'meta.json')
    cache_n_episodes = None
    if os.path.exists(cache_meta_path):
        with open(cache_meta_path) as f:
            cache_n_episodes = int(json.load(f).get('n_episodes') or 0) or None
    print(f"Loading episodes ({ds_tag}, cap={cache_n_episodes}) ...")
    if ds_tag == 'oxe':
        episodes = load_lerobot_episodes(oxe_id, load_video=False,
                                          n_episodes=cache_n_episodes)
    else:
        episodes = load_so101_episodes(load_video=False)
        if cache_n_episodes:
            episodes = episodes[:cache_n_episodes]
    loader = make_loader(args.cache_dir, episodes,
                         batch_size=1, num_workers=0, shuffle=True,
                         lru_size=2, augment=False)

    def vis_pipeline(hidden):
        layers = [hidden[:, l] for l in range(hidden.shape[1])]
        return vis_proj(resampler(aggregator(layers)))

    def decode_action(idxs):
        """idxs: list of (B, T_l) per level, coarsest-first."""
        B = idxs[0].shape[0]
        embs = []
        for vq, T_l, idx in zip(vae.vqs, vae.seq_lens, idxs):
            embs.append(vq.emb(idx).view(B, T_l, vq.D).permute(0, 2, 1))
        return vae.decode(embs).transpose(1, 2)

    def topk_correct(logits_l, gt_l, k):
        # head emits K+1 logits (last = MASK); rank over the real codes only
        topk = logits_l[..., :K].topk(k, dim=-1).indices
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
        topk_correct_per_lvl = {l: {k: 0 for k in TOPK} for l in range(len(seq_lens))}
        total_per_lvl = [0] * len(seq_lens)
        mse_pol = mse_full = mse_vae = 0.0; n_chunks = 0
        with torch.no_grad():
            for s in cached:
                all_logits = policy(None, s['vis'], s['state'], mask_list=None,
                                    n_outer=H, n_inner=L)
                final = all_logits[-1]
                for l, T_l in enumerate(seq_lens):
                    for k in TOPK:
                        topk_correct_per_lvl[l][k] += topk_correct(final[l], s['gt'][l], k)
                    total_per_lvl[l] += T_l
                pred_idx = [final[l][..., :K].argmax(-1) for l in range(len(seq_lens))]
                pred_recon = decode_action(pred_idx)
                mse_pol  += ((pred_recon - s['gt_recon']) ** 2).mean().item()
                mse_full += ((pred_recon - s['action_norm']) ** 2).mean().item()
                mse_vae  += ((s['gt_recon'] - s['action_norm']) ** 2).mean().item()
                n_chunks += 1
        topk_acc = {l: {k: topk_correct_per_lvl[l][k] / max(total_per_lvl[l], 1) for k in TOPK}
                    for l in range(len(seq_lens))}
        return {'topk': topk_acc,
                'mse_policy': mse_pol / n_chunks,
                'mse_full':   mse_full / n_chunks,
                'mse_vae':    mse_vae / n_chunks}

    n_lvls = len(seq_lens)

    def run_h_grid(tag):
        """Run the H_GRID sweep at the policy's current ρ_H; return list of (H, r)."""
        header_cells = "  ".join(f"{'L%d top1/5/10' % l:>16}" for l in range(n_lvls))
        bar_w = max(92, 36 + n_lvls * 18)
        rho_l_now = float(torch.sigmoid(policy.rho_L_raw))
        rho_h_now = float(torch.sigmoid(policy.rho_H_raw))
        print("=" * bar_w)
        print(f"  [{tag}]  ρ_L={rho_l_now:.4f}  ρ_H={rho_h_now:.4f}")
        print(f"  {'H':>3}  {'time':>5}   {header_cells}  "
              f"{'mse_pol':>8}  {'mse_full':>8}  {'top1mean':>8}")
        print("-" * bar_w)
        res = []
        for H in H_GRID:
            t0 = time.perf_counter()
            r = eval_at(trained_L, H)
            elapsed = time.perf_counter() - t0
            cells = ["/".join(f"{r['topk'][l][k]*100:.0f}" for k in TOPK)
                     for l in range(n_lvls)]
            cell_str = "  ".join(f"{c:>16}" for c in cells)
            t1mean = sum(r['topk'][l][1] for l in range(n_lvls)) / n_lvls * 100
            marker = "  ←train" if H == trained_H else ""
            print(f"  {H:>3}  {elapsed:>5.1f}s   {cell_str}  "
                  f"{r['mse_policy']:>8.4f}  {r['mse_full']:>8.4f}  {t1mean:>7.1f}%{marker}")
            res.append((H, r))
        return res

    print(f"\nH-scaling eval on {N} samples (L fixed at trained={trained_L})")

    # Parse the ρ_H sweep. Empty -> single run at the trained (collapsed) ρ_H.
    sweep = [float(v) for v in args.rho_h_sweep.split()] if args.rho_h_sweep.strip() else []
    if not sweep:
        run_h_grid("trained ρ_H")
    else:
        print("DIAGNOSTIC: forcing ρ_H to test whether the outer loop is "
              "pathologically collapsed (forcing it up should help) or genuinely "
              "redundant (forcing it up should not help / hurt).\n")
        # First the as-trained baseline, then each forced value.
        run_h_grid("trained ρ_H (baseline)")
        for v in sweep:
            vv = min(max(v, 1e-4), 1 - 1e-4)
            policy.rho_H_raw.data.fill_(float(torch.logit(torch.tensor(vv))))
            run_h_grid(f"forced ρ_H={vv:g}")

    if args.diagnostics:
        import torch.nn.functional as F
        # Restore the LEARNED ρ_H (a sweep above may have changed it).
        policy.rho_H_raw.data.copy_(trained_rhoH_raw)
        lnK = math.log(K)
        print("\n" + "=" * 70)
        print(f"  DIAGNOSTICS @ trained L={trained_L} H={trained_H}  "
              f"ρ_L={float(torch.sigmoid(policy.rho_L_raw)):.3f} "
              f"ρ_H={float(torch.sigmoid(policy.rho_H_raw)):.3f}   (N={N} samples)")
        print(f"  refs: random top-1 = {100.0/K:.2f}%   max code-entropy = ln({K}) = {lnK:.3f} nats")
        print("=" * 70)
        gt_codes = {l: [] for l in range(n_lvls)}
        pred_codes = {l: [] for l in range(n_lvls)}
        ent_sum = [0.0] * n_lvls; ent_n = [0] * n_lvls
        a_pred_all, a_gt_all, a_vae_all = [], [], []
        with torch.no_grad():
            for s in cached:
                final = policy(None, s['vis'], s['state'], mask_list=None,
                               n_outer=trained_H, n_inner=trained_L)[-1]
                pred_idx = []
                for l in range(n_lvls):
                    lg = final[l][..., :K]                      # (1,T_l,K)
                    p = lg.argmax(-1); pred_idx.append(p)
                    gt_codes[l].append(s['gt'][l].flatten())
                    pred_codes[l].append(p.flatten())
                    pr = F.softmax(lg.float(), -1)
                    ent_sum[l] += -(pr * pr.clamp_min(1e-9).log()).sum(-1).sum().item()
                    ent_n[l] += pr[..., 0].numel()
                a_pred_all.append(decode_action(pred_idx))
                a_gt_all.append(s['action_norm']); a_vae_all.append(s['gt_recon'])
        for l in range(n_lvls):
            gt = torch.cat(gt_codes[l]); pr = torch.cat(pred_codes[l])
            acc = (gt == pr).float().mean().item()
            gtc = torch.bincount(gt, minlength=K).float()
            maj_acc = (gtc.max() / gtc.sum()).item()
            gtp = gtc / gtc.sum(); gt_ent = -(gtp[gtp > 0] * gtp[gtp > 0].log()).sum().item()
            prc = torch.bincount(pr, minlength=K).float()
            prp = prc / prc.sum(); pr_ent = -(prp[prp > 0] * prp[prp > 0].log()).sum().item()
            conf = ent_sum[l] / max(ent_n[l], 1)
            print(f"  level {l} (T={seq_lens[l]}):")
            print(f"     top-1 {acc*100:5.1f}%   majority-class {maj_acc*100:5.1f}%   "
                  f"lift {acc/max(maj_acc,1e-9):4.2f}x")
            print(f"     codes used  GT {int((gtc>0).sum())}/{K} (H={gt_ent:.2f})   "
                  f"PRED {int((prc>0).sum())}/{K} (H={pr_ent:.2f})  of ln{K}={lnK:.2f}")
            print(f"     pred confidence: mean softmax entropy {conf:.2f} nats "
                  f"({conf/lnK*100:.0f}% of max — lower=more confident)")
        a_pred = torch.cat(a_pred_all); a_gt = torch.cat(a_gt_all); a_vae = torch.cat(a_vae_all)
        A = a_gt.shape[-1]
        pol = ((a_pred - a_gt) ** 2).mean(dim=(0, 1))
        flo = ((a_vae - a_gt) ** 2).mean(dim=(0, 1))
        names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'grip']
        print(f"  per-action-dim MSE (RevIN-normalized space) — policy vs VAE-recon floor:")
        for d in range(A):
            nm = names[d] if d < len(names) else str(d)
            print(f"     dim {d} {nm:>5}: policy {pol[d]:.4f}   floor {flo[d]:.4f}   "
                  f"gap {pol[d]-flo[d]:+.4f}")
        print(f"  overall: policy MSE {pol.mean():.4f}   VAE floor {flo.mean():.4f}")


if __name__ == '__main__':
    main()
