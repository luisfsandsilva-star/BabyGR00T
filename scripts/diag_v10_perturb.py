#!/usr/bin/env python3
"""Two-part v10 diagnostic continuation.

(A) Perturbation test: manually push logprec output of g toward the clamp.
    Find the threshold beyond which training cannot recover.
(B) Forward-march from step=1000 for up to 5000 steps with v10's exact config.
    Watch for spontaneous spike. Includes safety guards disabled so we see raw.

These continue scripts/diag_v10_reproduce.py with the same env builder.
"""
import os, sys, math, time, importlib.util, pickle
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import torch, torch.nn as nn

# import the env builder from the reproduce script
spec = importlib.util.spec_from_file_location('repro', os.path.join(THIS, 'diag_v10_reproduce.py'))
repro = importlib.util.module_from_spec(spec); spec.loader.exec_module(repro)

from babygroot_strm.optimizer import MuSGD_LARS
from babygroot_strm.multi_oxe import EMBODIMENTS
DEV = repro.DEV
CKPT_PATH = repro.CKPT_PATH


def fresh_env():
    args = dict(cnn_expand=2, cnn_out_dim=192, cnn_norm='scalenorm', cnn_pe=True,
                cnn_dropout=0.1, img_size=224, max_text=24, dim=768, depth=3,
                L_inner=5, H_outer=2, n_emb_prefix=16,
                weighting='geometric', alpha_parametrization='sigmoid',
                alpha_per_dim=False, per_emb_head=True, beta=1e-3, free_bits=0.1,
                state_noise=0.02)
    return repro.build_everything(args)


def find_logprec_layer(policy):
    """The g network output's last D dims ARE the logprec channels. To perturb,
    we add a positive bias to the output of the last g block's FF (the 2nd Linear).
    Actually g's output is the residual stream of TRMNet, last layer's output."""
    # g.blocks[last].ff outputs the residual contribution. Find ff.w3 (gate-down).
    last_block = policy.g.blocks[-1]
    # The g output is the residual stream, modified additively by ff.w3 inside ff.
    # We add a per-channel bias via a hook on policy.g forward.
    return last_block


def perturb_test(env, lp_push):
    """Inject a forward-hook on g that adds `lp_push` to the logprec channels of
    its output. Run a few steps and report whether training stays stable."""
    policy = env['policy']
    D = policy.latent_dim
    hooks = []
    def hook_fn(module, input, output):
        # output: (B, T, 2D). Add lp_push to the last D dims (logprec channels).
        push = torch.zeros_like(output)
        push[..., D:] = lp_push
        return output + push
    h = policy.g.register_forward_hook(hook_fn)
    hooks.append(h)
    print(f"\n=== PERTURB test: pushing logprec output by +{lp_push} ===")
    try:
        logs = repro.run_one_regime(f'lp_push={lp_push}', env, rmax=0.61,
                                    n_steps=15, do_optimizer_step=True)
    finally:
        for h in hooks: h.remove()
    losses = [l['loss'] for l in logs]
    grads = [l['total_g'] for l in logs]
    lps = [max(l['L_lp_max'], l['H_lp_max']) for l in logs]
    print(f"  loss: {min(losses):.2f} .. {max(losses):.2f}  (final {losses[-1]:.2f})")
    print(f"  ||g||: {min(grads):.2f} .. {max(grads):.2f}  (final {grads[-1]:.2f})")
    print(f"  max|lp|: {min(lps):.2f} .. {max(lps):.2f}")
    n_nan = sum(1 for l in losses if not math.isfinite(l))
    return dict(losses=losses, grads=grads, lps=lps, n_nan=n_nan, lp_push=lp_push)


def long_march(env, n_steps=2000, log_every=50, ckpt_every=500):
    """Run forward training for n_steps with v10's exact config from step=1000.
    Watch for any spontaneous spike. Save ckpts so we can restart-and-analyze
    if NaN occurs.
    """
    policy = env['policy']; mods = env['mods']
    trainable = [p for m in mods for p in m.parameters() if p.requires_grad]
    opt = MuSGD_LARS(trainable, lr=2e-3, momentum=0.95, weight_decay=1e-3)
    warmup = 150
    print(f"\n=== LONG MARCH from step=1000 for {n_steps} steps ===")
    print(f"{'step':>5} {'loss':>8} {'||g||':>10} {'max|lp|':>9} {'%clamp':>7} "
          f"{'g_amp':>7} {'wall_s':>7}")
    it = iter(env['loader'])
    history = []
    t0 = time.perf_counter()
    spike_step = None
    for step in range(n_steps):
        try:
            frames, states, actions, prevs, tasks, emb_robots = next(it)
        except StopIteration:
            it = iter(env['loader'])
            frames, states, actions, prevs, tasks, emb_robots = next(it)
        frames = frames.to(DEV); states = states.to(DEV)
        actions = actions.to(DEV); prevs = prevs.to(DEV)
        s_enc = env['encode_state'](states, emb_robots, noisy=True)
        with torch.no_grad():
            gt = env['encode_targets'](actions, prevs, emb_robots)

        capture = {}
        orig_inner = repro.hook_inner_loop(policy, capture)
        rmax = min(1.0, 0.3 + 0.7 * (step + 1000) / 50000)            # continue v10's curriculum from step 1000
        opt.zero_grad(set_to_none=True)
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                vis = env['encode_modalities'](frames, tasks, emb_robots)
                emb_id_t = torch.tensor([env['emb_id_to_idx'].get(r, len(EMBODIMENTS)) for r in emb_robots],
                                        dtype=torch.long, device=DEV)
                loss, per, _ = policy.forward_loss(gt, vis, s_enc, n_inner=5, h_max=3,
                                                   mask_ratio_max=rmax, emb_id=emb_id_t,
                                                   label_smoothing=0.05)
            if not torch.isfinite(loss):
                print(f"  step {step}: NON-FINITE loss → SPIKE DETECTED")
                spike_step = step; break
            loss.backward()
            total_g = math.sqrt(sum(p.grad.norm().item() ** 2 for p in trainable if p.grad is not None))
            # apply hard ceiling at 10 — same as v10's config
            if total_g > 10:
                scale = 10 / total_g
                for p in trainable:
                    if p.grad is not None: p.grad.mul_(scale)
            for g in opt.param_groups:
                g['lr'] = 2e-3 * min(1.0, (step + 1000 + warmup) / warmup)
            opt.step()
        finally:
            policy._inner = orig_inner

        lp_steps = capture.get('logprec_per_step', [])
        max_lp = max((s['lp_abs_max'] for s in lp_steps), default=0)
        pct_clamp = max((s['pct_at_clamp'] for s in lp_steps), default=0)
        g_amp = lp_steps[-1]['g_norm'] / max(lp_steps[0]['g_norm'], 1e-8) if lp_steps else 0
        history.append(dict(step=step, loss=loss.item(), total_g=total_g, max_lp=max_lp,
                            pct_clamp=pct_clamp, g_amp=g_amp))
        if step % log_every == 0 or total_g > 8:
            elapsed = time.perf_counter() - t0
            print(f"{step:>5d} {loss.item():>8.3f} {total_g:>10.2e} {max_lp:>9.2f} "
                  f"{pct_clamp:>6.1f}% {g_amp:>7.2f} {elapsed:>7.1f}")
        # Spike detection: ||g|| > 100 is clearly anomalous (v10 normal ~7)
        if total_g > 100:
            print(f"  step {step}: SPIKE — ||g||={total_g:.2e}")
            spike_step = step; break
    elapsed = time.perf_counter() - t0
    print(f"\n  finished {len(history)} steps in {elapsed:.1f}s  ({len(history)/elapsed:.1f} steps/s)")
    if spike_step is not None:
        print(f"  SPIKE @ step {spike_step}")
    else:
        print(f"  no spike in {len(history)} steps; max||g||={max(h['total_g'] for h in history):.2f}, "
              f"max|lp|={max(h['max_lp'] for h in history):.2f}")
    return history, spike_step


def main():
    env = fresh_env()
    ck = torch.load(CKPT_PATH, map_location=DEV, weights_only=False)
    keys = ['cnn', 'text_agg', 'cnn_proj', 'text_proj', 'kv_norm', 'state_encoders', 'emb_id_emb', 'policy']

    # (A) — perturbation sweep
    print("\n" + "=" * 72)
    print("PART (A): logprec perturbation sweep")
    print("=" * 72)
    perturb_results = []
    for push in (0.0, 2.0, 5.0, 10.0):
        # reload weights fresh
        for k, m in zip(keys, env['mods']): m.load_state_dict(ck[k])
        r = perturb_test(env, lp_push=push)
        perturb_results.append(r)

    print(f"\n--- perturbation summary ---")
    print(f"{'lp_push':>8} {'final_loss':>10} {'max||g||':>10} {'max|lp|':>9} {'n_nan':>6}")
    for r in perturb_results:
        final_loss = r['losses'][-1] if r['losses'] else float('nan')
        max_g = max(r['grads']) if r['grads'] else float('nan')
        max_lp = max(r['lps']) if r['lps'] else float('nan')
        print(f"{r['lp_push']:>8.1f} {final_loss:>10.2f} {max_g:>10.2f} {max_lp:>9.2f} {r['n_nan']:>6d}")

    # (B) — long march
    print("\n" + "=" * 72)
    print("PART (B): forward-march 2000 steps from step=1000")
    print("=" * 72)
    for k, m in zip(keys, env['mods']): m.load_state_dict(ck[k])
    history, spike_step = long_march(env, n_steps=2000, log_every=50)
    with open('/tmp/v10_long_march.pkl', 'wb') as f:
        pickle.dump({'history': history, 'spike_step': spike_step,
                     'perturb': perturb_results}, f)
    print(f"\nlogs → /tmp/v10_long_march.pkl")


if __name__ == '__main__':
    main()
