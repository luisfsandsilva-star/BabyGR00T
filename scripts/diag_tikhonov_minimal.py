#!/usr/bin/env python3
"""Minimal Tikhonov test: tiny model, no vision, H=1, longer training.

Setup:
  - dim=128, depth=1, ff_hidden=512 (~0.3M params for g)
  - H_outer=1, L_inner=5  (only inner loop — no outer cycle)
  - vis/state replaced by random fixed tensors (no CNN, no text encoder)
  - 16 random target codes (fully synthetic — pure model behavior)
  - 1500 steps overfit
  - NO trust clamp (trust_max=∞)

Sweep 6 conditions (3 noise × 2 modes):
  - damped:      σ ∈ {0, 0.03, 0.10}
  - accumulator: σ ∈ {0, 0.03, 0.10}

Probes every 50 steps:
  - train_loss
  - ||W_g|| (Frobenius of g's 2D weights)
  - Lipschitz of g (random-direction finite-diff, max of 8 probes)

Expected from user's hypothesis:
  - damped + σ>0: Lipschitz GROWS (shortcut (1-α)·z amplifies g to fight noise)
  - damped + σ=0: Lipschitz stable or grows mildly
  - accumulator + σ>0: Lipschitz stays bounded (no shortcut; iteration just sums)
  - accumulator + σ=0: stable baseline

If even damped + σ>0 stays bounded → Tikhonov works as Bishop predicted.
If damped + σ>0 diverges while accumulator + σ>0 doesn't → user's hypothesis.
"""
import os, sys, random, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import STRMPolicy, MuSGD_LARS

# tiny model
DIM = 128
DEPTH = 1
FF = 512
HEADS = 4
KV = 1
L_INNER = 5
H_OUTER = 1               # <<< KEY: only inner loop
K = 64                    # smaller codebook
SEQ_LEN = 4               # one level of 4 codes
N_PREFIX = 16             # random kv tokens
BS = 16                   # small batch
N_STEPS = 1500
LR = 5e-3
PROBE_EVERY = 50

torch.set_num_threads(4); random.seed(0); torch.manual_seed(0); np.random.seed(0)


def build_policy(g_input_noise, update_mode='damped'):
    """Build a small STRMPolicy with NO vision components."""
    if update_mode == 'damped':
        ap = 'sigmoid'; alpha_per_dim = True
    else:   # accumulator
        ap = 'clamp'; alpha_per_dim = False  # uses geometric weighting
    policy = STRMPolicy(
        seq_lens=(SEQ_LEN,), k_codebook=K, dim=DIM,
        heads=HEADS, kv_heads=KV, ff_hidden=FF,
        depth=DEPTH, L_inner=L_INNER, H_outer=H_OUTER,
        state_dim=DIM, max_prefix=N_PREFIX + SEQ_LEN + 16,
        weighting='geometric' if update_mode == 'accumulator' else 'clamp_direct',
        update_mode=update_mode,
        alpha_parametrization=ap, alpha_per_dim=alpha_per_dim,
        n_embodiments=1, per_emb_head=False, dropout=0.0,
        g_input_noise=g_input_noise,
        grad_checkpoint=False,    # no need with tiny model
    ).cuda()
    policy.train()
    return policy


def build_synthetic_batch():
    """Random fixed vis, state, targets — no vision/text at all."""
    torch.manual_seed(99)
    vis = torch.randn(BS, N_PREFIX, DIM, device='cuda') * 0.5    # random KV tokens
    state = torch.randn(BS, DIM, device='cuda') * 0.5            # random state
    target = torch.randint(0, K, (BS, SEQ_LEN), device='cuda')   # random targets
    gt = [target]
    return vis, state, gt


@torch.no_grad()
def measure_lipschitz(policy, vis, state, gt, n_probes=8, delta_scale=0.01):
    """Probe g's Lipschitz with random perturbations on its input."""
    seq_lens = policy.seq_lens
    Bf = vis.shape[0]
    mask_list = [torch.ones(Bf, T_l, dtype=torch.bool, device=vis.device) for T_l in seq_lens]
    kv = policy._build_kv(vis, state)
    y = policy._y_embed(Bf, vis.device, gt, mask_list)
    g0 = policy.g(y, kv)
    ratios = []
    for _ in range(n_probes):
        delta = torch.randn_like(y) * delta_scale
        g1 = policy.g(y + delta, kv)
        r = (g1 - g0).norm().item() / delta.norm().item()
        ratios.append(r)
    return max(ratios)


def measure_g_norm(policy):
    """Frobenius norm of g's 2D weights only."""
    total_sq = 0.0
    for _, p in policy.g.named_parameters():
        if p.dim() == 2:
            total_sq += (p.data.float().norm() ** 2).item()
    return total_sq ** 0.5


def run_condition(sigma, mode, vis, state, gt, label):
    """Train policy for N_STEPS, return trajectory."""
    print(f"\n[{label}] sigma={sigma}, mode={mode}", flush=True)
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    policy = build_policy(g_input_noise=sigma, update_mode=mode)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"  policy params: {n_params/1e6:.2f}M", flush=True)
    # NO clamp — trust_max set to a huge value
    opt = MuSGD_LARS(policy.parameters(), lr=LR, momentum=0.95,
                     weight_decay=0.0, trust_max=1e6)
    eid = torch.zeros(BS, dtype=torch.long, device='cuda')

    rows = []  # (step, loss, ||W_g||, Lipschitz)
    for step in range(N_STEPS + 1):
        if step % PROBE_EVERY == 0:
            policy.eval()
            Lip = measure_lipschitz(policy, vis, state, gt)
            Wn = measure_g_norm(policy)
            policy.train()
            # also compute eval loss
            with torch.no_grad():
                loss_eval, _, _ = policy.forward_loss(
                    gt, vis, state, n_inner=L_INNER, n_outer=H_OUTER, h_max=H_OUTER,
                    mask_ratio_max=1.0, emb_id=eid, label_smoothing=0.0,
                    mask_sampler='cosine')
            rows.append((step, loss_eval.item(), Wn, Lip))
            if step % (PROBE_EVERY * 2) == 0:
                print(f"  step={step:4d} loss={loss_eval.item():.4f} |W_g|={Wn:8.2f} Lip={Lip:.4f}", flush=True)
            # divergence safety
            if not np.isfinite(loss_eval.item()) or Lip > 1e6:
                print(f"  DIVERGED at step {step}", flush=True); break

        opt.zero_grad(set_to_none=True)
        loss, _, _ = policy.forward_loss(
            gt, vis, state, n_inner=L_INNER, n_outer=H_OUTER, h_max=H_OUTER,
            mask_ratio_max=1.0, emb_id=eid, label_smoothing=0.0,
            mask_sampler='cosine')
        loss.backward()
        opt.step()

    del policy, opt
    import gc; gc.collect(); torch.cuda.empty_cache()
    return rows


def main():
    print(f"=== Minimal Tikhonov test ===")
    print(f"  Architecture: dim={DIM}, depth={DEPTH}, ff={FF}, heads={HEADS}, L={L_INNER}, H={H_OUTER}")
    print(f"  Batch: {BS} synthetic samples (no vision/text)")
    print(f"  Steps: {N_STEPS}, LR={LR}, no trust clamp, no WD")

    vis, state, gt = build_synthetic_batch()

    conditions = []
    for sigma in [0.0, 0.03, 0.10]:
        conditions.append((sigma, 'damped',      f'damped  σ={sigma}'))
    for sigma in [0.0, 0.03, 0.10]:
        conditions.append((sigma, 'accumulator', f'accum   σ={sigma}'))

    results = {}
    for sigma, mode, label in conditions:
        try:
            results[label] = run_condition(sigma, mode, vis, state, gt, label)
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}", flush=True)
            results[label] = []

    # summary table
    print(f"\n{'='*100}\nSUMMARY ({N_STEPS} steps overfitting on {BS} synthetic samples):\n{'='*100}")
    print(f"  {'condition':<22s} {'loss_0':>8s} {'loss_end':>9s}  {'|W_g|_0':>8s} {'|W_g|_end':>10s}  "
          f"{'Lip_0':>7s} {'Lip_mid':>8s} {'Lip_end':>8s}  {'verdict':>20s}")
    for label, rows in results.items():
        if not rows: continue
        l0 = rows[0][1]; lE = rows[-1][1]
        W0 = rows[0][2]; WE = rows[-1][2]
        Lip0 = rows[0][3]; LipE = rows[-1][3]
        LipMid = rows[len(rows)//2][3] if len(rows) > 2 else Lip0
        v = ''
        if LipE > 5: v = '⚠ Lip DIVERGES'
        elif LipE > 2.0: v = 'Lip grew >2×'
        elif LipE > LipMid * 1.2: v = 'still growing'
        elif LipE < 1.0: v = '✓ Lip BOUNDED'
        else: v = 'Lip stable'
        print(f"  {label:<22s} {l0:>8.3f} {lE:>9.3f}  {W0:>8.2f} {WE:>10.2f}  "
              f"{Lip0:>7.3f} {LipMid:>8.3f} {LipE:>8.3f}  {v:>20s}")

    # write CSV
    import csv
    with open('/tmp/tikhonov_minimal.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'step', 'loss', 'W_g_norm', 'Lipschitz'])
        for label, rows in results.items():
            for s, l, W, L in rows:
                w.writerow([label, s, l, W, L])
    print(f"\n  detailed → /tmp/tikhonov_minimal.csv")

    # comparison: damped vs accumulator at each sigma
    print(f"\n{'─'*100}\nDAMPED vs ACCUMULATOR comparison (Lipschitz at end):")
    for sigma in [0.0, 0.03, 0.10]:
        d_label = f'damped  σ={sigma}'
        a_label = f'accum   σ={sigma}'
        if d_label in results and a_label in results and results[d_label] and results[a_label]:
            d_lip = results[d_label][-1][3]
            a_lip = results[a_label][-1][3]
            print(f"  σ={sigma}:  damped Lip={d_lip:.3f},  accumulator Lip={a_lip:.3f}  (ratio: {d_lip/max(a_lip,1e-3):.2f}×)")


if __name__ == '__main__':
    main()
