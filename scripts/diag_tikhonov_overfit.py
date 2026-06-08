#!/usr/bin/env python3
"""Overfit-on-64-samples test to verify:
  (a) Trust clamp prevents weight runaway   (trust_max=1.0 vs ∞)
  (b) Current Tikhonov placement bounds Lipschitz, OR backfires
      (per user's claim that damped-Banach shortcut may inflate g's Lipschitz
       to compensate for noise it can route around via (1-α)·z path).

Runs 4 conditions for 400 steps each on the SAME 64 widowx samples:
  - σ=0,  trust_max=∞     (baseline, both off)
  - σ=0,  trust_max=1     (clamp only)
  - σ=0.03, trust_max=∞   (Tikhonov only — user's "wrong spot" hypothesis)
  - σ=0.03, trust_max=1   (both)

Tracks every 50 steps:
  - train_loss
  - ||W||_2 of policy.g.blocks total (Frobenius across the g sub-net)
  - empirical Lipschitz of g (random-direction finite-diff)

Output: CSV + summary table. The verdict on Tikhonov placement:
  if (σ=0.03, no clamp) Lipschitz stays equal-or-lower than (σ=0, no clamp)
    → Tikhonov works as intended
  if (σ=0.03, no clamp) Lipschitz is HIGHER than (σ=0, no clamp)
    → user's hypothesis is right: noise is backfiring
"""
import os, sys, glob, random, time
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn
from babygroot_strm import (STRMPolicy, LayerAggregator, ScaleNorm,
                             ActionVQVAE1d, VQ1d_EMA, MuSGD_LARS)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image

ROBOT = 'widowx'; DIM = 512; HEADS = 8; KV = 2; FF = 2048; DEPTH = 2
N_EMB = len(EMBODIMENTS) + 1; PFX = 16; IMG = 144; MAX_TEXT = 64
L_INNER = 5; H_OUTER = 3
BS = 64
N_STEPS = 400
LR = 8.97e-3
PROBE_EVERY = 50
torch.set_num_threads(4); random.seed(0); torch.manual_seed(0); np.random.seed(0)


def build_pipeline(g_input_noise):
    cnn = EfficientCNN(dims=[24,48,96,192], depths=[1,1,1,1], expand=4,
                       out_dim=192, norm='gn', pos_emb=True, img_size=IMG,
                       dropout=0.0, n_embodiments=0).cuda()
    text_agg = LayerAggregator(hidden_dim=512, n_layers=9).cuda()
    cnn_proj = nn.Linear(192, DIM).cuda()
    text_proj = nn.Linear(512, DIM).cuda()
    kv_norm = ScaleNorm(DIM).cuda()
    se = nn.ModuleDict({ROBOT: nn.Sequential(
        nn.Linear(8, DIM), nn.GELU(), nn.Linear(DIM, DIM))}).cuda()
    emb_id = nn.Embedding(N_EMB, DIM*PFX).cuda()
    n_vis = (IMG//32)**2
    policy = STRMPolicy(seq_lens=(4,), k_codebook=256, dim=DIM, heads=HEADS,
        kv_heads=KV, ff_hidden=FF, depth=DEPTH, L_inner=L_INNER, H_outer=H_OUTER,
        state_dim=DIM, max_prefix=n_vis+MAX_TEXT+16+PFX,
        weighting='clamp_direct', update_mode='damped',
        alpha_parametrization='sigmoid', alpha_per_dim=True,
        n_embodiments=N_EMB, per_emb_head=False, dropout=0.0,
        g_input_noise=g_input_noise).cuda()
    for m in [cnn, text_agg, cnn_proj, text_proj, kv_norm, se, emb_id, policy]:
        m.train()
    return cnn, text_agg, cnn_proj, text_proj, kv_norm, se, emb_id, policy


def build_batch():
    specs = []
    for d in sorted(glob.glob('data/oxe/*')):
        if not os.path.isfile(os.path.join(d, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(d, chunk_len=16, lookback=16, chunk_stride=4)
            if sp.chunk_index and sp.robot == ROBOT: specs.append(sp)
        except Exception: pass
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    rng = random.Random(0); pool = list(range(len(ds))); rng.shuffle(pool)
    vae_c = torch.load(f'data/ckpts/oxe_vqvae_{ROBOT}.pt', map_location='cpu', weights_only=False)
    adim = vae_c['action_dim']
    vae = ActionVQVAE1d(action_dim=adim, vq_cls=VQ1d_EMA, k=vae_c.get('k', 128)).cuda().eval()
    vae.load_state_dict(vae_c['vae'])
    var_global = vae_c['action_var_global'].view(1, 1, -1).cuda()
    samples = []
    for idx in pool:
        if len(samples) >= BS: break
        try:
            fr, st, ac, pv, tk, eid_, di = ds[idx]
            if ac.shape[-1] != adim or st.shape[-1] != 8: continue
            from PIL import Image
            pil = fr.convert('RGB').resize((IMG, IMG))
            ft = torch.from_numpy(np.asarray(pil).copy()).permute(2,0,1).float() / 255.
            samples.append((ft, st, ac, pv, tk))
        except Exception: pass
    Bf = len(samples)
    frames = torch.stack([s[0] for s in samples]).cuda()
    states = torch.stack([s[1] for s in samples]).cuda()
    actions = torch.stack([s[2] for s in samples]).cuda()
    prevs = torch.stack([s[3] for s in samples]).cuda()
    tasks = [s[4] for s in samples]
    t5 = torch.load('data/cache/t5_text_cache_paraphrased.pt', map_location='cpu', weights_only=False)
    t5s = torch.zeros(9, Bf, MAX_TEXT, 512, device='cuda')
    for b, tk in enumerate(tasks):
        e = t5['embeddings'].get(tk)
        if e is None: continue
        h = e['hidden'].float().cuda()
        t = min(h.shape[1], MAX_TEXT); L = min(h.shape[0], 9)
        t5s[:L, b, :t, :] = h[:L, :t, :]
    img_var = torch.load('data/cache/image_var_global.pt', map_location='cpu', weights_only=False)
    img_n = normalize_image(frames, img_var['var_global'].cuda())
    # GT codes
    m_pv = prevs.mean(dim=1, keepdim=True)
    S = ((prevs - m_pv) ** 2).sum(dim=1, keepdim=True)
    lam = actions.shape[1] / (S + actions.shape[1] * var_global)
    xn = ((actions - m_pv) * lam.sqrt()).transpose(1, 2)
    with torch.no_grad():
        cd, _ = vae.encode_with_soft(xn, tau=0.1)
    gt = [cd[0]]
    eid_t = torch.tensor([EMBODIMENT_ID.get(ROBOT, len(EMBODIMENTS))] * Bf, dtype=torch.long).cuda()
    return img_n, t5s, states, gt, eid_t, Bf


@torch.no_grad()
def measure_lipschitz_g(policy, vis, s_enc, gt, eid_t, n_probes=6, delta_scale=0.01):
    """Probe g's Lipschitz with random perturbations of its input."""
    Bf = vis.shape[0]
    seq_lens = policy.seq_lens
    mask_list = [torch.ones(Bf, T_l, dtype=torch.bool, device=vis.device) for T_l in seq_lens]
    kv = policy._build_kv(vis, s_enc)
    y = policy._y_embed(Bf, vis.device, gt, mask_list)
    # use y as the probe input (same as first inner-loop call would see at z_L=z_H=0)
    g0 = policy.g(y, kv)
    ratios = []
    for _ in range(n_probes):
        delta = torch.randn_like(y) * delta_scale
        g1 = policy.g(y + delta, kv)
        r = (g1 - g0).norm().item() / delta.norm().item()
        ratios.append(r)
    return max(ratios)


def measure_g_weight_norm(policy):
    """Total Frobenius norm of all g sub-net 2D weights."""
    total_sq = 0.0
    for n, p in policy.g.named_parameters():
        if p.dim() == 2:
            total_sq += (p.data.float().norm() ** 2).item()
    return total_sq ** 0.5


def run_condition(sigma, trust_max, img_n, t5s, states, gt, eid_t, Bf, label):
    """Train fresh model on the batch for N_STEPS, return trajectory."""
    print(f"\n[{label}] σ={sigma} trust_max={trust_max} ...", flush=True)
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    mods = build_pipeline(g_input_noise=sigma)
    cnn, text_agg, cnn_proj, text_proj, kv_norm, se, emb_id, policy = mods
    params = [p for m in mods for p in m.parameters() if p.requires_grad]
    opt = MuSGD_LARS(params, lr=LR, momentum=0.95, weight_decay=2e-3, trust_max=trust_max)
    eid = eid_t

    rows = []   # (step, loss, |W_g|, Lip)
    for step in range(N_STEPS + 1):
        # measure first (so step 0 is initial state)
        if step % PROBE_EVERY == 0:
            policy.eval()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
                tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
                etok = emb_id(eid).view(Bf, PFX, DIM)
                vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
                s_enc = se[ROBOT](states)
            Lip = measure_lipschitz_g(policy, vis.float(), s_enc.float(), gt, eid)
            Wn = measure_g_weight_norm(policy)
            policy.train()
            with torch.no_grad():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
                    tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
                    etok = emb_id(eid).view(Bf, PFX, DIM)
                    vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
                    s_enc = se[ROBOT](states)
                    loss_only, _, _ = policy.forward_loss(gt, vis, s_enc,
                        n_inner=L_INNER, n_outer=H_OUTER, h_max=H_OUTER,
                        mask_ratio_max=1.0, emb_id=eid, label_smoothing=0.10,
                        mask_sampler='cosine')
            rows.append((step, loss_only.item(), Wn, Lip))
            print(f"  step={step:3d}  loss={loss_only.item():.4f}  |W_g|={Wn:.2f}  Lip={Lip:.3f}", flush=True)

        # train step
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            vtok, _ = cnn(img_n); vtok = cnn_proj(vtok)
            tagg = text_agg([t5s[l] for l in range(9)]); ttok = text_proj(tagg)
            etok = emb_id(eid).view(Bf, PFX, DIM)
            vis = kv_norm(torch.cat([etok, vtok, ttok], dim=1))
            s_enc = se[ROBOT](states)
            loss, _, _ = policy.forward_loss(gt, vis, s_enc,
                n_inner=L_INNER, n_outer=H_OUTER, h_max=H_OUTER,
                mask_ratio_max=1.0, emb_id=eid, label_smoothing=0.10,
                mask_sampler='cosine')
        loss.backward()
        opt.step()

    del mods, opt
    import gc; gc.collect(); torch.cuda.empty_cache()
    return rows


def main():
    print(f"[diag] loading batch...")
    img_n, t5s, states, gt, eid_t, Bf = build_batch()
    print(f"  Bf={Bf} samples loaded")

    # use torch.inf-equivalent for "no clamp"
    NO_CLAMP = 1e6   # effectively unbounded

    results = {}
    for label, sigma, tmax in [
        ('A: no-noise no-clamp',  0.00, NO_CLAMP),
        ('B: no-noise clamp=1',   0.00, 1.0),
        ('C: noise=0.03 no-clamp',0.03, NO_CLAMP),
        ('D: noise=0.03 clamp=1', 0.03, 1.0),
    ]:
        results[label] = run_condition(sigma, tmax, img_n, t5s, states, gt, eid_t, Bf, label)

    # summary
    print(f"\n{'='*100}\nSUMMARY (after {N_STEPS} steps overfitting on {Bf} samples):\n{'='*100}")
    print(f"  {'condition':<26s} {'loss_0':>8s} {'loss_end':>9s} {'|W_g|_0':>9s} {'|W_g|_end':>10s} "
          f"{'Lip_0':>7s} {'Lip_end':>8s}  {'verdict':>15s}")
    base_lip = None
    for label, rows in results.items():
        l0 = rows[0][1]; lE = rows[-1][1]
        W0 = rows[0][2]; WE = rows[-1][2]
        Lip0 = rows[0][3]; LipE = rows[-1][3]
        v = ''
        if 'no-clamp' in label and WE > W0 * 3: v = '||W|| grew >3×'
        elif 'clamp=1' in label and WE < W0 * 1.5: v = '||W|| bounded'
        print(f"  {label:<26s} {l0:>8.3f} {lE:>9.3f} {W0:>9.1f} {WE:>10.1f} "
              f"{Lip0:>7.3f} {LipE:>8.3f}  {v:>15s}")

    # verdict on Tikhonov: compare A (no-noise no-clamp) vs C (noise no-clamp)
    print(f"\n{'─'*100}\nTIKHONOV PLACEMENT VERDICT:")
    A_end = results['A: no-noise no-clamp'][-1][3]
    C_end = results['C: noise=0.03 no-clamp'][-1][3]
    print(f"  Lipschitz with σ=0 unclamped: {A_end:.3f}")
    print(f"  Lipschitz with σ=0.03 unclamped: {C_end:.3f}")
    if C_end > A_end * 1.2:
        print(f"  ⚠ Tikhonov noise INCREASED Lipschitz by {C_end/A_end:.1f}× — user's hypothesis confirmed:")
        print(f"     damped-Banach (1-α)·z shortcut lets model compensate for noise by amplifying g.")
    elif C_end < A_end * 0.8:
        print(f"  ✓ Tikhonov noise DECREASED Lipschitz by {A_end/C_end:.1f}× — works as intended.")
    else:
        print(f"  ≈ Tikhonov had minimal effect on Lipschitz (within 20%).")

    print(f"\n{'─'*100}\nTRUST CLAMP VERDICT:")
    A_W = results['A: no-noise no-clamp'][-1][2]
    B_W = results['B: no-noise clamp=1'][-1][2]
    print(f"  ||W_g|| with no clamp: {A_W:.1f}")
    print(f"  ||W_g|| with clamp=1:  {B_W:.1f}  (ratio: {B_W/A_W:.2f})")

    # save full csv
    import csv
    with open('/tmp/tikhonov_overfit.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'step', 'loss', 'W_g_norm', 'Lipschitz_g'])
        for label, rows in results.items():
            for s, l, W, L in rows:
                w.writerow([label, s, l, W, L])
    print(f"\n  detailed → /tmp/tikhonov_overfit.csv")


if __name__ == '__main__':
    main()
