#!/usr/bin/env python3
"""Test ALTERNATIVE Tikhonov placements to find one that doesn't allow the
damped-Banach (1-α)·z shortcut around the noise.

Compares 5 placements (damped iteration, σ=0.03 each, except baseline):
  A) BASELINE σ=0                           — control
  B) g-input (current): g(z + ctx + η)     — model can route signal through (1-α)·z
  C) iteration-output: z_new = (1-α)z + α·g(z+ctx) + η
     — noise added to STATE after damping; propagates through next g-call;
       can't be bypassed by (1-α)·z because z itself is now noisy
  D) initial-state: z_0 = η (noise only at iteration start)
     — bounds the iteration's sensitivity to initial perturbation
  E) prefix-once: vis_noisy = vis + η (noise added to KV prefix once before iter)
     — closest to classical Bishop; regularizes ∂(loss)/∂(input) across full iter

All with: minimal model (dim=128 H=1), 16 synthetic samples, 1500 steps, no clamp.
Expected: B grows Lipschitz (current placement, what we observed).
          C, D, E should bound Lipschitz tighter than B.
          Best one = where to move the noise.
"""
import os, sys, random
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import STRMPolicy, MuSGD_LARS

DIM = 128; DEPTH = 1; FF = 512; HEADS = 4; KV = 1
L_INNER = 5; H_OUTER = 1
K = 64; SEQ_LEN = 4; N_PREFIX = 16; BS = 16
N_STEPS = 1500; LR = 5e-3; PROBE_EVERY = 50
SIGMA = 0.03      # fixed noise magnitude — focus on PLACEMENT not amount
torch.set_num_threads(4); random.seed(0); torch.manual_seed(0); np.random.seed(0)


def build_policy():
    """Standard damped policy, g_input_noise=0 (we apply noise manually per placement)."""
    return STRMPolicy(
        seq_lens=(SEQ_LEN,), k_codebook=K, dim=DIM,
        heads=HEADS, kv_heads=KV, ff_hidden=FF,
        depth=DEPTH, L_inner=L_INNER, H_outer=H_OUTER,
        state_dim=DIM, max_prefix=N_PREFIX + SEQ_LEN + 16,
        weighting='clamp_direct', update_mode='damped',
        alpha_parametrization='sigmoid', alpha_per_dim=True,
        n_embodiments=1, per_emb_head=False, dropout=0.0,
        g_input_noise=0.0,   # OFF in policy — we patch manually
        grad_checkpoint=False,
    ).cuda()


@torch.no_grad()
def measure_lipschitz(policy, vis, state, gt, n_probes=8, delta_scale=0.01):
    Bf = vis.shape[0]
    mask_list = [torch.ones(Bf, T_l, dtype=torch.bool, device=vis.device) for T_l in policy.seq_lens]
    kv = policy._build_kv(vis, state)
    y = policy._y_embed(Bf, vis.device, gt, mask_list)
    g0 = policy.g(y, kv)
    ratios = []
    for _ in range(n_probes):
        delta = torch.randn_like(y) * delta_scale
        g1 = policy.g(y + delta, kv)
        ratios.append((g1 - g0).norm().item() / delta.norm().item())
    return max(ratios)


def measure_g_norm(policy):
    return sum((p.data.float().norm() ** 2).item() for _, p in policy.g.named_parameters() if p.dim() == 2) ** 0.5


def make_forward(placement, sigma):
    """Patches STRMPolicy._inner and forward() based on placement string."""
    def custom_inner(self, z_H, y, kv, wL, vis=None):
        z_L = torch.zeros_like(y)
        if self.training and sigma > 0 and placement == 'D':
            # D: initial-state noise
            z_L = z_L + torch.randn_like(z_L) * sigma
        alpha = wL
        for t in range(self.L_inner):
            if self.training and sigma > 0 and placement == 'B':
                # B: g-input noise (current)
                g_in = z_L + z_H + y + torch.randn_like(z_L) * sigma
            else:
                g_in = z_L + z_H + y
            g_out = self.g(g_in, kv)
            z_L = (1 - alpha) * z_L + alpha * g_out
            if self.training and sigma > 0 and placement == 'C':
                # C: iteration-output noise (after damping)
                z_L = z_L + torch.randn_like(z_L) * sigma
        return z_L
    return custom_inner


def run_condition(placement, sigma, vis_base, state, gt, label):
    print(f"\n[{label}] placement={placement} σ={sigma}", flush=True)
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    policy = build_policy()

    # monkey-patch _inner
    custom = make_forward(placement, sigma)
    bound = custom.__get__(policy, type(policy))
    policy._inner = bound

    opt = MuSGD_LARS(policy.parameters(), lr=LR, momentum=0.95, weight_decay=0.0, trust_max=1e6)
    eid = torch.zeros(BS, dtype=torch.long, device='cuda')

    rows = []
    for step in range(N_STEPS + 1):
        # For placement E: noise the prefix vis once per step
        if placement == 'E' and sigma > 0 and policy.training:
            vis_step = vis_base + torch.randn_like(vis_base) * sigma
        else:
            vis_step = vis_base

        if step % PROBE_EVERY == 0:
            policy.eval()
            Lip = measure_lipschitz(policy, vis_base, state, gt)   # measure on CLEAN vis
            Wn = measure_g_norm(policy)
            policy.train()
            with torch.no_grad():
                loss_eval, _, _ = policy.forward_loss(
                    gt, vis_base, state, n_inner=L_INNER, n_outer=H_OUTER, h_max=H_OUTER,
                    mask_ratio_max=1.0, emb_id=eid, label_smoothing=0.0, mask_sampler='cosine')
            rows.append((step, loss_eval.item(), Wn, Lip))
            if step % (PROBE_EVERY * 4) == 0:
                print(f"  step={step:4d}  loss={loss_eval.item():.4f}  |W_g|={Wn:7.2f}  Lip={Lip:.4f}", flush=True)
            if not np.isfinite(loss_eval.item()) or Lip > 1e6:
                print(f"  DIVERGED at step {step}"); break

        opt.zero_grad(set_to_none=True)
        loss, _, _ = policy.forward_loss(
            gt, vis_step, state, n_inner=L_INNER, n_outer=H_OUTER, h_max=H_OUTER,
            mask_ratio_max=1.0, emb_id=eid, label_smoothing=0.0, mask_sampler='cosine')
        loss.backward()
        opt.step()

    del policy, opt; import gc; gc.collect(); torch.cuda.empty_cache()
    return rows


def main():
    print(f"=== Tikhonov PLACEMENT comparison (σ={SIGMA} fixed) ===")
    print(f"  damped iteration, dim={DIM} depth={DEPTH} H={H_OUTER} L={L_INNER}, no clamp, no WD\n")
    torch.manual_seed(99)
    vis = torch.randn(BS, N_PREFIX, DIM, device='cuda') * 0.5
    state = torch.randn(BS, DIM, device='cuda') * 0.5
    target = torch.randint(0, K, (BS, SEQ_LEN), device='cuda')
    gt = [target]

    conds = [
        ('A: baseline σ=0',             'B', 0.0),       # placement irrelevant since σ=0
        ('B: g-input    (current)',     'B', SIGMA),
        ('C: iter-output',              'C', SIGMA),
        ('D: initial z',                'D', SIGMA),
        ('E: prefix-once',              'E', SIGMA),
    ]
    results = {}
    for label, placement, sigma in conds:
        results[label] = run_condition(placement, sigma, vis, state, gt, label)

    print(f"\n{'='*100}\nSUMMARY (placement of σ={SIGMA} noise in damped iteration, 1500 steps):\n{'='*100}")
    print(f"  {'condition':<26s} {'loss_end':>9s} {'|W_g|_end':>10s} {'Lip_mid':>8s} {'Lip_end':>8s}  {'vs baseline':>11s}  {'verdict':>20s}")
    baseline_lip = results['A: baseline σ=0'][-1][3] if results['A: baseline σ=0'] else None
    for label, rows in results.items():
        if not rows: continue
        lE = rows[-1][1]; WE = rows[-1][2]
        LipE = rows[-1][3]
        LipMid = rows[len(rows)//2][3] if len(rows) > 2 else rows[0][3]
        delta = (LipE / baseline_lip - 1) * 100 if baseline_lip else 0
        delta_s = f'{delta:+5.0f}%'
        if LipE < 1.0: v = '✓ tight (Lip<1)'
        elif LipE < 1.5: v = '≈ moderate'
        else: v = 'loose Lip'
        print(f"  {label:<26s} {lE:>9.4f} {WE:>10.2f} {LipMid:>8.3f} {LipE:>8.3f}  {delta_s:>11s}  {v:>20s}")

    # CSV
    import csv
    with open('/tmp/tikhonov_placement.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'step', 'loss', 'W_g_norm', 'Lipschitz'])
        for label, rows in results.items():
            for s, l, W, L in rows: w.writerow([label, s, l, W, L])
    print(f"\n  detailed → /tmp/tikhonov_placement.csv")


if __name__ == '__main__':
    main()
