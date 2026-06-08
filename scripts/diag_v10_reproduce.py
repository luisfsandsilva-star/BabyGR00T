#!/usr/bin/env python3
"""Reproduce v10 runaway conditions on actual model + actual data.

Loads the v10 step=1000 ckpt, builds the same training pipeline (real OXE
dataloader, T5 cache, image var, bf16 autocast), then runs instrumented
forward+backward passes at the curriculum regime where the crash happened
(step 22200 → rmax ≈ 0.61).

Instruments:
  - logprec values (μ, log τ) at each inner-loop fusion step
  - gradient norm of g's output at each step (retain_grad on intermediates)
  - per-layer gradient flow into the policy (to see what receives the spike)
  - tracks weight stats over multiple optimizer steps to see drift

Runs across rmax = {0.3, 0.61, 1.0} to isolate whether mask curriculum is implicated.
"""
import os, sys, glob, math, time, random, json
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn as nn

from babygroot_strm import (RevIN, ActionVQVAE1d, VQ1d_EMA, STRMPolicyVAE,
                            LayerAggregator, ScaleNorm)
from babygroot_strm.cnn_vision import EfficientCNN
from babygroot_strm.optimizer import MuSGD_LARS
from babygroot_strm.multi_oxe import (load_dataset_spec, MultiOXEDataset,
                                       EMBODIMENTS, EMBODIMENT_ID)
from babygroot_strm.perimg_norm import normalize_image
from babygroot_strm import augment


CKPT_PATH = 'data/ckpts/oxe_policy_v10_bayesian.pt'
T5_CACHE  = 'data/cache/t5_text_cache.pt'
IMG_VAR   = 'data/cache/image_var_global.pt'
OXE_ROOT  = 'data/oxe'
VAE_DIR   = 'data/ckpts'
DEV       = 'cuda' if torch.cuda.is_available() else 'cpu'
N_STEPS_PER_REGIME = 30
BATCH = 12                                  # small for shared-GPU friendliness; v10 used 40
TARGET_STEP = 22200                         # the step where v10 crashed
TOTAL_STEPS_SCHED = 100_000                 # v10 was scheduled for 100k


def build_everything(args):
    """Mirror train_oxe.main() exactly. Returns the bag of modules + helpers."""
    print(f"[diag] device={DEV}")
    # ── datasets ──
    print("loading dataset specs...")
    specs = []
    for ds_dir in sorted(glob.glob(os.path.join(OXE_ROOT, '*'))):
        if not os.path.isfile(os.path.join(ds_dir, 'meta', 'info.json')): continue
        try:
            sp = load_dataset_spec(ds_dir, chunk_len=16, lookback=16)
            if sp.chunk_index: specs.append(sp)
        except Exception as e:
            pass
    print(f"  {len(specs)} datasets")
    present_emb = sorted({sp.robot for sp in specs})

    # ── shared cond VAE ──
    from babygroot_strm.cond_vae import CondActionVQVAE1d
    sc = torch.load(os.path.join(VAE_DIR, 'oxe_shared_vae.pt'), map_location=DEV, weights_only=False)
    shared_vae = CondActionVQVAE1d(action_dim=sc['action_dim'], n_embodiments=sc['n_embodiments'],
                                   k=sc['k']).to(DEV)
    shared_vae.load_state_dict(sc['vae']); shared_vae.eval()
    for p in shared_vae.parameters(): p.requires_grad_(False)
    vaes = {emb: shared_vae for emb in present_emb}
    var_globals = {emb: sc['action_var_globals'][EMBODIMENT_ID.get(emb, -1)].to(DEV).view(1, 1, -1)
                   for emb in present_emb if EMBODIMENT_ID.get(emb, -1) in sc['action_var_globals']}
    seq_lens = tuple(shared_vae.seq_lens); K = shared_vae.vqs[0].K
    print(f"  cond VAE: K={K} seq_lens={seq_lens}")

    # ── T5 + image var ──
    t5 = torch.load(T5_CACHE, map_location='cpu', weights_only=False)
    t5_emb, t5_dim, t5_layers = t5['embeddings'], t5['dim'], t5['n_layers']
    img_var = torch.load(IMG_VAR, map_location='cpu', weights_only=False)
    var_global_img = img_var['var_global'].to(DEV)

    # ── modules ──
    cnn = EfficientCNN(dims=[24, 48, 96, 192], depths=[1, 1, 1, 1], expand=args['cnn_expand'],
                       out_dim=args['cnn_out_dim'], norm=args['cnn_norm'], pos_emb=args['cnn_pe'],
                       img_size=args['img_size'], dropout=args['cnn_dropout']).to(DEV)
    text_agg = LayerAggregator(hidden_dim=t5_dim, n_layers=t5_layers).to(DEV)
    cnn_proj = nn.Linear(args['cnn_out_dim'], args['dim']).to(DEV)
    text_proj = nn.Linear(t5_dim, args['dim']).to(DEV)
    kv_norm = ScaleNorm(args['dim']).to(DEV)
    n_vis = (args['img_size'] // 32) ** 2
    state_encoders = nn.ModuleDict({
        emb: nn.Sequential(nn.Linear(8, args['dim']), nn.GELU(), nn.Linear(args['dim'], args['dim']))
        for emb in present_emb}).to(DEV)
    emb_id_emb = nn.Embedding(len(EMBODIMENTS) + 1, args['dim'] * args['n_emb_prefix']).to(DEV)
    emb_id_to_idx = {emb: EMBODIMENT_ID.get(emb, len(EMBODIMENTS)) for emb in present_emb}
    max_prefix = n_vis + args['max_text'] + 16 + args['n_emb_prefix']
    policy = STRMPolicyVAE(seq_lens=seq_lens, k_codebook=K, dim=args['dim'], heads=8,
                           depth=args['depth'], L_inner=args['L_inner'], H_outer=args['H_outer'],
                           state_dim=args['dim'], max_prefix=max_prefix,
                           weighting=args['weighting'],
                           rho_L=0.0, rho_H=0.0,                  # sigmoid → α=0.5
                           update_mode='bayesian',
                           alpha_parametrization=args['alpha_parametrization'],
                           alpha_per_dim=args['alpha_per_dim'],
                           n_embodiments=len(EMBODIMENTS) + 1,
                           per_emb_head=args['per_emb_head'],
                           beta=args['beta'], free_bits=args['free_bits']).to(DEV)
    mods = [cnn, text_agg, cnn_proj, text_proj, kv_norm, state_encoders, emb_id_emb, policy]
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']

    # ── load v10 weights ──
    ck = torch.load(CKPT_PATH, map_location=DEV, weights_only=False)
    print(f"  loading v10 ckpt at step={ck['step']}")
    for k, m in zip(keys, mods):
        m.load_state_dict(ck[k])
    loaded_step = ck['step']

    # ── data ──
    _AUG_KW = dict(brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4),
                   blur_sigma=(0.0, 1.5), crop_keep=(0.70, 1.0))
    def frame_to_tensor(pil):
        pil = pil.convert('RGB')
        pil = augment._apply_visual_params(pil, augment._sample_visual_params(random, **_AUG_KW))
        pil = pil.resize((args['img_size'], args['img_size']))
        return torch.from_numpy(np.asarray(pil)).permute(2, 0, 1).float() / 255.
    def collate(batch):
        frames = torch.stack([frame_to_tensor(b[0]) for b in batch])
        states = torch.stack([b[1] for b in batch])
        actions = torch.stack([b[2] for b in batch])
        prevs = torch.stack([b[3] for b in batch])
        tasks = [b[4] for b in batch]
        emb_ids_int = [b[5] for b in batch]
        emb_robots = [EMBODIMENTS[e] if e < len(EMBODIMENTS) else 'unknown' for e in emb_ids_int]
        return frames, states, actions, prevs, tasks, emb_robots
    ds = MultiOXEDataset(specs, chunk_len=16, lookback=16)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True,
                                         num_workers=0, collate_fn=collate, drop_last=True)
    print(f"  dataset: {len(ds)} chunks")

    # ── helpers ──
    def t5_layers_batch(tasks):
        B, T = len(tasks), args['max_text']
        out = torch.zeros(t5_layers, B, T, t5_dim)
        for b, tk in enumerate(tasks):
            e = t5_emb.get(tk)
            if e is None: continue
            h = e['hidden'].float(); t = min(h.shape[1], T)
            out[:, b, :t, :] = h[:, :t, :]
        return out.to(DEV)
    def encode_modalities(frames, tasks, emb_robots):
        x = normalize_image(frames, var_global_img)
        vtok, _ = cnn(x); vtok = cnn_proj(vtok)
        t5s = t5_layers_batch(tasks)
        tagg = text_agg([t5s[l] for l in range(t5s.shape[0])])
        ttok = text_proj(tagg)
        idx = torch.tensor([emb_id_to_idx.get(r, len(EMBODIMENTS)) for r in emb_robots], device=DEV)
        etok = emb_id_emb(idx).view(idx.shape[0], args['n_emb_prefix'], args['dim'])
        return kv_norm(torch.cat([etok, vtok, ttok], dim=1))
    @torch.no_grad()
    def encode_state(state, emb_robots, noisy=False):
        out = torch.zeros(state.shape[0], args['dim'], device=DEV)
        for emb in set(emb_robots):
            mask = torch.tensor([r == emb for r in emb_robots], device=DEV)
            if mask.sum() == 0: continue
            s = state[mask]
            if noisy: s = s + torch.randn_like(s) * args['state_noise']
            out[mask] = state_encoders[emb](s)
        return out
    def encode_targets(actions, prevs, emb_robots):
        all_codes = [torch.zeros(actions.shape[0], T_l, dtype=torch.long, device=DEV) for T_l in seq_lens]
        for emb in set(emb_robots):
            mask = torch.tensor([r == emb for r in emb_robots], device=DEV)
            if mask.sum() == 0: continue
            ac = actions[mask]; pv = prevs[mask]
            vg = var_globals[emb]
            nT = ac.shape[1]
            m = pv.mean(dim=1, keepdim=True)
            S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
            lam = nT / (S + nT * vg)
            xn = ((ac - m) * lam.sqrt()).transpose(1, 2)
            eid = torch.full((ac.shape[0],), EMBODIMENT_ID.get(emb, len(EMBODIMENTS)),
                             dtype=torch.long, device=DEV)
            gt, _ = shared_vae.encode_with_soft(xn, eid, tau=0.1)
            for l in range(len(seq_lens)):
                all_codes[l][mask] = gt[l]
        return all_codes
    return dict(policy=policy, mods=mods, loader=loader, loaded_step=loaded_step,
                encode_modalities=encode_modalities, encode_state=encode_state,
                encode_targets=encode_targets, emb_id_to_idx=emb_id_to_idx)


def hook_inner_loop(policy, capture):
    """Monkey-patch policy._inner to capture per-fusion-step logprec stats and
    g outputs. Allows .retain_grad() on each g output so we see grad through them."""
    orig_inner = policy._inner
    def patched_inner(z_H, y, kv, wL):
        # Mirror the bayesian branch with instrumentation; fall back otherwise.
        if policy.update_mode != 'bayesian':
            return orig_inner(z_H, y, kv, wL)
        z_L = torch.zeros_like(y)
        D = policy.latent_dim
        for t in range(policy.L_inner):
            g_t = policy.g(z_L + z_H + y, kv)
            g_t.retain_grad()
            mu_g, lp_g = g_t[..., :D], g_t[..., D:]
            capture.setdefault('logprec_per_step', []).append({
                'step': t, 'lp_min': lp_g.detach().min().item(),
                'lp_max': lp_g.detach().max().item(),
                'lp_mean': lp_g.detach().mean().item(),
                'lp_abs_max': lp_g.detach().abs().max().item(),
                'pct_at_clamp': (lp_g.detach().abs() > 4.5).float().mean().item() * 100,
                'mu_norm': mu_g.detach().norm().item(),
                'g_norm': g_t.detach().norm().item(),
            })
            capture.setdefault('g_outputs', []).append(g_t)
            z_L = policy._bayes_fuse(z_L, g_t, D)
        return z_L
    policy._inner = patched_inner
    return orig_inner


def run_one_regime(name, env, rmax, n_steps=N_STEPS_PER_REGIME, do_optimizer_step=True, lr=2e-3):
    """Run n_steps of forward+backward at fixed mask ratio rmax. Returns logs."""
    policy = env['policy']; mods = env['mods']
    trainable = [p for m in mods for p in m.parameters() if p.requires_grad]
    opt = MuSGD_LARS(trainable, lr=lr, momentum=0.95, weight_decay=1e-3) if do_optimizer_step else None
    logs = []
    print(f"\n{'='*72}\nregime: {name}  rmax={rmax}  n_steps={n_steps}  optimizer={'YES' if do_optimizer_step else 'no'}\n{'='*72}")
    print(f"{'step':>4} {'loss':>10} {'||g||':>10} {'L_lp_max':>9} {'H_lp_max':>9} {'%clamp_L':>9} "
          f"{'g0_norm':>10} {'g_final_norm':>14} {'amp':>7}")
    it = iter(env['loader'])
    for step in range(n_steps):
        try:
            frames, states, actions, prevs, tasks, emb_robots = next(it)
        except StopIteration:
            it = iter(env['loader'])
            frames, states, actions, prevs, tasks, emb_robots = next(it)
        frames = frames.to(DEV, non_blocking=True)
        states = states.to(DEV, non_blocking=True)
        actions = actions.to(DEV, non_blocking=True)
        prevs = prevs.to(DEV, non_blocking=True)
        s_enc = env['encode_state'](states, emb_robots, noisy=True)
        with torch.no_grad():
            gt = env['encode_targets'](actions, prevs, emb_robots)

        # Install instrumentation hook
        capture = {}
        orig_inner = hook_inner_loop(policy, capture)

        if opt is not None: opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            vis = env['encode_modalities'](frames, tasks, emb_robots)
            emb_id_t = torch.tensor([env['emb_id_to_idx'].get(r, len(EMBODIMENTS)) for r in emb_robots],
                                    dtype=torch.long, device=DEV)
            loss, per, _ = policy.forward_loss(gt, vis, s_enc, n_inner=5, h_max=3,
                                               mask_ratio_max=rmax, emb_id=emb_id_t,
                                               label_smoothing=0.05)
        if not torch.isfinite(loss):
            print(f"  step {step:>3d}: NON-FINITE loss, skipping")
            policy._inner = orig_inner
            continue
        loss.backward()

        # Capture per-step g-output gradient norms (before zero_grad)
        per_step_g_grads = []
        for i, g_t in enumerate(capture.get('g_outputs', [])):
            if g_t.grad is not None:
                per_step_g_grads.append(g_t.grad.detach().norm().item())
            else:
                per_step_g_grads.append(None)

        # total grad norm
        total_g = math.sqrt(sum(p.grad.norm().item() ** 2 for p in trainable if p.grad is not None))

        # Restore policy._inner
        policy._inner = orig_inner

        # Pick representative inner-loop stats: L (first inner block) and H (last)
        lp_steps = capture.get('logprec_per_step', [])
        L_inner = 5
        L_stats = lp_steps[:L_inner] if lp_steps else []
        H_stats = lp_steps[L_inner:] if len(lp_steps) > L_inner else []
        L_lp_max = max((s['lp_abs_max'] for s in L_stats), default=0)
        H_lp_max = max((s['lp_abs_max'] for s in H_stats), default=0)
        L_pct_clamp = max((s['pct_at_clamp'] for s in L_stats), default=0)
        g0_norm = lp_steps[0]['g_norm'] if lp_steps else 0
        g_final_norm = lp_steps[-1]['g_norm'] if lp_steps else 0
        amplification = (g_final_norm / max(g0_norm, 1e-8)) if g0_norm else 0

        # apply optimizer step
        if opt is not None:
            opt.step()

        logs.append(dict(step=step, loss=loss.item(), total_g=total_g, L_lp_max=L_lp_max,
                         H_lp_max=H_lp_max, L_pct_clamp=L_pct_clamp,
                         g0_norm=g0_norm, g_final_norm=g_final_norm, amp=amplification,
                         per_step_g_grads=per_step_g_grads,
                         lp_steps=lp_steps))
        if step % 5 == 0 or step < 5:
            print(f"{step:>4d} {loss.item():>10.3f} {total_g:>10.2e} {L_lp_max:>9.2f} {H_lp_max:>9.2f} "
                  f"{L_pct_clamp:>8.1f}% {g0_norm:>10.2f} {g_final_norm:>14.2f} {amplification:>7.2f}x")
    return logs


def analyze_per_step_grad(logs, name):
    """Average per-fusion-step gradient amplification across all logged steps."""
    print(f"\n--- per-fusion-iteration gradient flow ({name}) ---")
    print(f"{'iter':>4} {'avg_||∂L/∂g_t||':>20} {'avg_||g_t||':>15} {'avg_|lp_max|':>14} {'%clamp':>8}")
    all_grads_per_iter = {}
    all_norms_per_iter = {}
    all_lpmax_per_iter = {}
    all_clamp_per_iter = {}
    for l in logs:
        for i, gn in enumerate(l['per_step_g_grads']):
            if gn is not None:
                all_grads_per_iter.setdefault(i, []).append(gn)
        for i, s in enumerate(l['lp_steps']):
            all_norms_per_iter.setdefault(i, []).append(s['g_norm'])
            all_lpmax_per_iter.setdefault(i, []).append(s['lp_abs_max'])
            all_clamp_per_iter.setdefault(i, []).append(s['pct_at_clamp'])
    for i in sorted(all_grads_per_iter.keys()):
        ag = sum(all_grads_per_iter[i]) / len(all_grads_per_iter[i])
        an = sum(all_norms_per_iter.get(i, [0])) / max(len(all_norms_per_iter.get(i, [])), 1)
        al = sum(all_lpmax_per_iter.get(i, [0])) / max(len(all_lpmax_per_iter.get(i, [])), 1)
        ac = sum(all_clamp_per_iter.get(i, [0])) / max(len(all_clamp_per_iter.get(i, [])), 1)
        marker = '←L start' if i == 0 else '←H start' if i == 5 else ''
        print(f"{i:>4d} {ag:>20.4e} {an:>15.2f} {al:>14.3f} {ac:>7.1f}% {marker}")


def main():
    args = dict(cnn_expand=2, cnn_out_dim=192, cnn_norm='scalenorm', cnn_pe=True,
                cnn_dropout=0.1, img_size=224, max_text=24, dim=768, depth=3,
                L_inner=5, H_outer=2, n_emb_prefix=16,
                weighting='geometric', alpha_parametrization='sigmoid',
                alpha_per_dim=False, per_emb_head=True, beta=1e-3, free_bits=0.1,
                state_noise=0.02)
    env = build_everything(args)
    print(f"\n→ ckpt was at step {env['loaded_step']}; v10 crashed at step {TARGET_STEP}")
    print(f"→ at step {TARGET_STEP}, rmax = min(1, 0.3 + 0.7·{TARGET_STEP}/{TOTAL_STEPS_SCHED//2}) = "
          f"{min(1.0, 0.3 + 0.7*TARGET_STEP/(TOTAL_STEPS_SCHED*0.5)):.3f}")

    # --- Regime A: low mask (early curriculum) ---
    logs_a = run_one_regime('rmax=0.30 (early curriculum)', env, rmax=0.30, do_optimizer_step=True)
    analyze_per_step_grad(logs_a, 'rmax=0.30')

    # Reload model so each regime gets clean step-1000 weights
    print("\n[reload] restoring v10 step=1000 weights for next regime")
    ck = torch.load(CKPT_PATH, map_location=DEV, weights_only=False)
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']
    for k, m in zip(keys, env['mods']):
        m.load_state_dict(ck[k])

    # --- Regime B: rmax at crash point ---
    logs_b = run_one_regime(f'rmax=0.61 (v10 crash regime)', env, rmax=0.61, do_optimizer_step=True)
    analyze_per_step_grad(logs_b, 'rmax=0.61')

    print("\n[reload] restoring v10 step=1000 weights for next regime")
    for k, m in zip(keys, env['mods']):
        m.load_state_dict(ck[k])

    # --- Regime C: rmax=1.0 (full mask, hardest task) ---
    logs_c = run_one_regime('rmax=1.0 (full mask)', env, rmax=1.0, do_optimizer_step=True)
    analyze_per_step_grad(logs_c, 'rmax=1.0')

    # --- Summary comparison ---
    print(f"\n{'='*72}\nSUMMARY: how each regime evolves over {N_STEPS_PER_REGIME} steps\n{'='*72}")
    def fmt(name, ls):
        first = ls[0]; last = ls[-1]
        avg_g = sum(l['total_g'] for l in ls) / len(ls)
        max_g = max(l['total_g'] for l in ls)
        max_lp = max(max(l['L_lp_max'], l['H_lp_max']) for l in ls)
        avg_amp = sum(l['amp'] for l in ls) / len(ls)
        print(f"  {name:<32s} loss {first['loss']:.2f}→{last['loss']:.2f}  "
              f"||g|| avg={avg_g:.2e} max={max_g:.2e}  max|lp|={max_lp:.2f}  "
              f"g-amp(final/first) avg={avg_amp:.2f}x")
    fmt('rmax=0.30', logs_a)
    fmt('rmax=0.61 (crash regime)', logs_b)
    fmt('rmax=1.00', logs_c)

    # save raw logs for plotting / further analysis
    import pickle
    with open('/tmp/v10_repro_logs.pkl', 'wb') as f:
        pickle.dump({'rmax=0.30': logs_a, 'rmax=0.61': logs_b, 'rmax=1.00': logs_c}, f)
    print(f"\nraw logs → /tmp/v10_repro_logs.pkl")


if __name__ == '__main__':
    main()
