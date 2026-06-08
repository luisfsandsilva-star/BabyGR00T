#!/usr/bin/env python3
"""Train one VQ-VAE per embodiment from all the OXE datasets on disk.

For each embodiment (widowx, franka, ur5, ...) we:
  1. Gather all per-frame actions from the matching datasets' parquets.
  2. Form (current_chunk, prev_chunk) pairs of 16 actions each.
  3. Compute action_var_global per-dim (Gamma-prior for the precision norm).
  4. Train an ActionVQVAE1d (K=128, seq_lens=(4,)) on those pairs.
  5. Save data/ckpts/oxe_vqvae_<embodiment>.pt with the same schema as the single-embodiment ckpt.

Usage: python -m scripts.train_oxe_vaes [--oxe-root data/oxe] [--steps 4000] [--out-dir data/ckpts]
"""
import os, sys, time, glob, json, argparse
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pyarrow.parquet as pq
from babygroot_strm import RevIN, ActionVQVAE1d, VQ1d_EMA
from babygroot_strm.fsq import FSQ1d


def _action_convention(info: dict) -> str:
    """Classify a dataset's action space from its info.json 'names' field.

    Returns 'cartesian' (xyz+rpy+gripper), 'jointspace' (motor_0..N+gripper),
    or 'unknown'. Used to filter per-emb VAE training to a single convention,
    since you can't mix Cartesian deltas and joint angles in one codebook.
    """
    feats = info.get('features', {}).get('action', {})
    names_dict = feats.get('names', {})
    if isinstance(names_dict, dict):
        names = names_dict.get('motors') or names_dict.get('axes') or []
    else:
        names = list(names_dict) if names_dict else []
    names_lower = [str(n).lower() for n in names]
    has_xyz = any(n in ('x', 'y', 'z') for n in names_lower)
    has_euler = any(n in ('roll', 'pitch', 'yaw', 'rx', 'ry', 'rz') for n in names_lower)
    has_axisangle = any(n.startswith('axis_angle') for n in names_lower)
    if has_xyz and (has_euler or has_axisangle):
        # All end-effector control with xyz + rotation. Different rotation
        # parameterizations (Euler vs axis-angle) are approximately equal for the
        # small per-step rotations typical of manipulation, so we lump them as
        # 'cartesian' and let one VAE handle both. Mostly fine; sharp-rotation
        # tasks may see slight code quality degradation.
        return 'cartesian'
    if any(n.startswith('motor_') or n.startswith('joint_') for n in names_lower):
        return 'jointspace'
    return 'unknown'


def gather_action_pairs_for_embodiment(oxe_root: str, embodiment: str, chunk_len: int = 16,
                                       convention: str = 'cartesian',
                                       exclude_datasets: set = None):
    """Walk datasets matching this embodiment AND this action convention,
    collect (current, prev) chunk pairs.

    convention: 'cartesian' = [x, y, z, roll, pitch, yaw, gripper] (7-dim end-effector
    deltas), 'jointspace' = [motor_0..N, gripper] (joint angles). These describe
    different action spaces and can't be merged in one codebook — train separate
    VAEs per (embodiment, convention) if you need both.

    Within the selected convention, action_dim must already be consistent (true
    in practice for current OXE; if a future case breaks this, error out).
    """
    cur_list, prev_list = [], []
    matched = []; skipped_conv = []
    locked_A = None
    for ds_dir in sorted(glob.glob(os.path.join(oxe_root, '*'))):
        info_p = os.path.join(ds_dir, 'meta', 'info.json')
        if not os.path.isfile(info_p): continue
        info = json.load(open(info_p))
        if info.get('robot_type') != embodiment: continue
        ds_name = os.path.basename(ds_dir)
        if exclude_datasets and ds_name in exclude_datasets:
            print(f"  [{embodiment}/{convention}] EXCLUDED by request: {ds_name}")
            continue
        conv = _action_convention(info)
        if conv != convention:
            skipped_conv.append((ds_name, conv))
            continue
        ad = info.get('features', {}).get('action', {}).get('shape', [None])[0]
        if locked_A is None and ad is not None: locked_A = ad
        if locked_A is not None and ad is not None and ad != locked_A:
            print(f"  [{embodiment}/{convention}] UNEXPECTED action_dim {ad} (expected {locked_A}) "
                  f"in {os.path.basename(ds_dir)} — investigate"); continue
        n_eps = 0
        for pq_path in sorted(glob.glob(os.path.join(ds_dir, 'data', 'chunk-*', '*.parquet'))):
            try:
                t = pq.read_table(pq_path, columns=['action'])
                ac_flat = np.stack(t.column('action').to_pylist())          # (T, A)
                T, A = ac_flat.shape
                if T < 2 * chunk_len: continue
                if A != locked_A: continue
                n_full = T // chunk_len
                ac_chunks = torch.from_numpy(ac_flat[:n_full * chunk_len].reshape(n_full, chunk_len, A)).float()
                cur_list.append(ac_chunks[1:])
                prev_list.append(ac_chunks[:-1])
                n_eps += 1
            except Exception:
                pass
        if n_eps > 0:
            matched.append((os.path.basename(ds_dir), n_eps))
    if skipped_conv:
        print(f"  [{embodiment}/{convention}] SKIPPED (different action convention): {skipped_conv}")
    if not cur_list:
        return None, None, []
    return torch.cat(cur_list, dim=0), torch.cat(prev_list, dim=0), matched


def train_one(embodiment: str, cur: torch.Tensor, prev: torch.Tensor, steps: int,
              lr: float, batch_size: int, dev: str, action_dim: int, quantizer: str = 'vq',
              k: int = 128, val_frac: float = 0.1, patience_evals: int = 5,
              eval_every: int = None,
              action_noise: float = 0.0, time_drop: float = 0.0,
              weight_decay: float = 1e-4, dropout: float = 0.0,
              binary_gripper: bool = False, dead_threshold: int = 5, grip_weight: float = 1.0):
    """Train with held-out validation + early stopping + best-ckpt tracking.

    - Splits 10% of pairs as validation (shuffled then split, fixed seed).
    - Eval every eval_every steps (default = max(steps//20, 100)).
    - Early-stops when val recon hasn't improved by 0.001 over `patience_evals` evals.
    - Returns best-by-val-recon VAE state, not last.
    - binary_gripper: last action dim → Bernoulli (sigmoid+BCE), excluded from MSE.
    """
    var_global = cur.var(dim=(0, 1)).clamp(min=1e-8).to(dev).view(1, 1, -1)
    vq_cls = FSQ1d if quantizer == 'fsq' else VQ1d_EMA
    vae = ActionVQVAE1d(action_dim=action_dim, vq_cls=vq_cls, k=k, dropout=dropout,
                        binary_last=binary_gripper, dead_threshold=dead_threshold).to(dev)
    # Gripper = last dim. Derive its [lo,hi] range + binarization threshold from data.
    g = cur[..., -1]
    g_lo, g_hi = float(g.min()), float(g.max())
    grip_thr = 0.5 * (g_lo + g_hi)
    if binary_gripper:
        vae.gripper_range.copy_(torch.tensor([g_lo, g_hi], device=dev))
        print(f"  [{embodiment}] binary gripper: range=[{g_lo:.3f},{g_hi:.3f}] thr={grip_thr:.3f} "
              f"frac_open={float((g > grip_thr).float().mean()):.3f}  grip_weight={grip_weight}")
    revin = RevIN(action_dim).to(dev)
    opt = torch.optim.AdamW(list(vae.parameters()) + list(revin.parameters()), lr=lr, weight_decay=weight_decay)

    # train/val split (fixed seed for reproducibility). Cap val to 2048 — large
    # val sets cause GPU OOM during the all-in-one _eval_val forward pass on
    # data-rich embodiments (google_robot val would be 15k+ samples).
    n_total = cur.shape[0]
    g = torch.Generator().manual_seed(42 + hash(embodiment) % 1000)
    perm = torch.randperm(n_total, generator=g)
    n_val_raw = max(int(n_total * val_frac), min(64, n_total // 4))
    n_val = min(n_val_raw, 2048)                                    # cap at 2048
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    cur_tr, prev_tr = cur[tr_idx], prev[tr_idx]
    cur_val, prev_val = cur[val_idx].to(dev), prev[val_idx].to(dev)

    loader = DataLoader(TensorDataset(cur_tr, prev_tr), batch_size=batch_size,
                        shuffle=True, drop_last=True)
    if eval_every is None: eval_every = max(steps // 20, 100)
    print(f"  [{embodiment}] var_global per-dim: {[round(x,4) for x in var_global.sqrt().flatten().tolist()]}")
    print(f"  [{embodiment}] train_pairs={cur_tr.shape[0]} val_pairs={cur_val.shape[0]}  "
          f"steps_budget={steps} eval_every={eval_every} early_stop_patience={patience_evals}")

    ncont = action_dim - 1 if binary_gripper else action_dim
    val_var_cont = cur_val[..., :ncont].var(dim=(0, 1)).clamp(min=1e-8)     # (ncont,) global per-dim var

    def _eval_val():
        """Mini-batched val eval. Returns dict: continuous raw/NMSE/NRMSE (gripper
        excluded when binary), gripper accuracy, codebook usage. raw drives early-stop."""
        vae.eval()
        VAL_BS = 256
        raw_sum = norm_sum = mean_sum = 0.0
        sq_err = torch.zeros(ncont, device=dev)
        grip_correct = grip_total = 0.0
        used_codes = set(); nb = 0
        with torch.no_grad():
            for s in range(0, cur_val.shape[0], VAL_BS):
                ac = cur_val[s:s+VAL_BS]; pv = prev_val[s:s+VAL_BS]
                n = ac.shape[1]
                m = pv.mean(dim=1, keepdim=True)
                S = ((pv - m) ** 2).sum(dim=1, keepdim=True)
                lam = n / (S + n * var_global)
                x = ((ac - m) * lam.sqrt()).transpose(1, 2)
                embs, _, idx = vae.encode(x)
                recon = vae.decode(embs)
                used_codes.update(idx[0].flatten().tolist())
                norm_sum += F.mse_loss(recon[:, :ncont], x[:, :ncont]).item() * ac.shape[0]
                ac_recon = vae.recon_to_actions(recon, m, lam)            # gripper-aware → (B,T,A)
                cont_err = (ac_recon[..., :ncont] - ac[..., :ncont]) ** 2  # (B,T,ncont)
                raw_sum += cont_err.mean().item() * ac.shape[0]
                sq_err += cont_err.mean(dim=(0, 1)) * ac.shape[0]
                mean_sum += ((m[..., :ncont] - ac[..., :ncont]) ** 2).mean().item() * ac.shape[0]
                if binary_gripper:
                    g_pred = torch.sigmoid(recon[:, -1, :]) > 0.5
                    g_tgt = ac[..., -1] > grip_thr
                    grip_correct += (g_pred == g_tgt).float().sum().item()
                    grip_total += g_tgt.numel()
                nb += ac.shape[0]
        vae.train()
        nmse = float((sq_err / nb / val_var_cont).mean())
        return dict(raw=raw_sum / nb, norm=norm_sum / nb, mean=mean_sum / nb,
                    nmse=nmse, nrmse=nmse ** 0.5, usage=len(used_codes) / k, n_codes=len(used_codes),
                    grip_acc=(grip_correct / grip_total if grip_total else float('nan')))

    vae.train(); revin.train()
    step = 0; t0 = time.perf_counter(); wr = wq = wn = 0
    best_val = float('inf'); best_step = 0; best_state = None; no_improve = 0
    history = []; last_eval = None
    CACHE_CLEAR_EVERY = 2000          # periodic empty_cache to fight slow allocator fragmentation
    OOM_FRACTION_LIMIT = 0.92         # if cuda usage exceeds 92% of total, force empty_cache early
    oom_break = False                  # flag set if an OOM occurs; loop exits cleanly
    while step < steps and not oom_break:
        for action, prv in loader:
            if step >= steps or oom_break: break
            try:
                action = action.to(dev, non_blocking=True); prv = prv.to(dev, non_blocking=True)
                n = action.shape[1]
                m = prv.mean(dim=1, keepdim=True)
                S = ((prv - m) ** 2).sum(dim=1, keepdim=True)
                lam = n / (S + n * var_global)
                x = ((action - m) * lam.sqrt()).transpose(1, 2)
                # === regularization (training-only): denoise + time-drop ===
                if action_noise > 0.0 or time_drop > 0.0:
                    x_in = x.clone()
                    if action_noise > 0.0:
                        x_in = x_in + torch.randn_like(x_in) * action_noise
                    if time_drop > 0.0:
                        drop_mask = (torch.rand(x_in.shape[0], 1, x_in.shape[2], device=dev) >= time_drop).float()
                        x_in = x_in * drop_mask
                else:
                    x_in = x
                embs, vql, _ = vae.encode(x_in)
                recon = vae.decode(embs)                 # (B, A, T)
                if binary_gripper:
                    # continuous dims: MSE in normalized space; gripper: BCE vs binarized raw.
                    cont_mse = F.mse_loss(recon[:, :-1], x[:, :-1])
                    g_tgt = (action[:, :, -1] > grip_thr).float()       # (B, T)
                    g_bce = F.binary_cross_entropy_with_logits(recon[:, -1, :], g_tgt)
                    recon_loss = cont_mse + grip_weight * g_bce
                else:
                    recon_loss = F.mse_loss(recon, x)    # reconstruct CLEAN target
                loss = recon_loss + vql
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                wr += recon_loss.item(); wq += float(vql); wn += 1
                step += 1
            except torch.cuda.OutOfMemoryError as _oom:
                # Final guardrail: don't crash. Save best, break, exit cleanly.
                print(f"  [{embodiment}] CUDA OOM at step {step}: best_val={best_val:.5f} @ step {best_step}. "
                      f"Saving best ckpt and exiting train loop.", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                oom_break = True
                break
            # Periodic empty_cache to mitigate slow fragmentation of the CUDA allocator
            # (we saw OOMs at step 56k despite expandable_segments=True; small periodic clears
            # have negligible cost and prevent the slow leak from accumulating).
            if torch.cuda.is_available() and step % CACHE_CLEAR_EVERY == 0:
                # Watchdog: if we're close to full, force-clear
                free, total = torch.cuda.mem_get_info()
                used_frac = 1.0 - free / total
                if used_frac >= OOM_FRACTION_LIMIT:
                    print(f"  [{embodiment}] (watchdog) GPU at {used_frac*100:.0f}% — empty_cache()", flush=True)
                    torch.cuda.empty_cache()
                else:
                    torch.cuda.empty_cache()           # cheap; do it periodically anyway
            if step % eval_every == 0:
                last_eval = _eval_val()
                raw_mse, norm_mse, mean_mse = last_eval['raw'], last_eval['norm'], last_eval['mean']
                ratio = raw_mse / max(mean_mse, 1e-9)
                history.append((step, raw_mse, norm_mse, mean_mse))
                gstr = f"grip_acc={last_eval['grip_acc']*100:.1f}% " if binary_gripper else ""
                print(f"    step {step:>6d}/{steps}  train_recon={wr/wn:.4f}  vq={wq/wn:.4f}  "
                      f"VAL cont_mse={raw_mse:.5f} ({ratio*100:.1f}% of mean) NRMSE={last_eval['nrmse']:.4f} "
                      f"NMSE={last_eval['nmse']:.4f} {gstr}usage={last_eval['usage']*100:.0f}% "
                      f"({last_eval['n_codes']}/{k}) norm={norm_mse:.4f}  [{time.perf_counter()-t0:.0f}s]", flush=True)
                wr = wq = wn = 0
                # early stopping by val raw_mse
                if raw_mse < best_val - 1e-5:
                    best_val = raw_mse; best_step = step
                    best_state = {kk: vv.detach().cpu().clone() for kk, vv in vae.state_dict().items()}
                    no_improve = 0
                    # Persist best to disk immediately so a crash mid-training doesn't lose it.
                    try:
                        import os as _os, torch as _t
                        _p = _os.path.join('/tmp', f'_vae_best_{embodiment}.pt')
                        _t.save({'state': best_state, 'best_val': best_val, 'best_step': best_step,
                                 'embodiment': embodiment}, _p)
                    except Exception as _e:
                        print(f"  [{embodiment}] (warn) couldn't checkpoint best to /tmp: {_e}", flush=True)
                else:
                    no_improve += 1
                    if no_improve >= patience_evals:
                        print(f"  [{embodiment}] EARLY STOP at step {step}: best val_raw_mse={best_val:.5f} "
                              f"@ step {best_step} ({no_improve} evals no improvement)", flush=True)
                        if best_state is not None: vae.load_state_dict(best_state)
                        return vae, revin, var_global, dict(best_val=best_val, best_step=best_step,
                                                              actual_steps=step, history=history,
                                                              final_eval=last_eval)
    # Restore best ALWAYS (cap reached, OOM, or normal exit)
    if best_state is not None:
        vae.load_state_dict(best_state)
        tag = 'OOM-BREAK' if oom_break else 'CAP'
        print(f"  [{embodiment}] finished {step} steps ({tag}); best val_raw_mse={best_val:.5f} @ step {best_step}",
              flush=True)
    else:
        # No best_state yet — emit a warning; ckpt will be the random-init weights
        print(f"  [{embodiment}] WARNING: training ended (step={step}) before first val eval; "
              f"using random-init weights. This is a bug — check eval_every < total steps.",
              flush=True)
    return vae, revin, var_global, dict(best_val=best_val, best_step=best_step,
                                          actual_steps=step, history=history, oom_break=oom_break,
                                          final_eval=last_eval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oxe-root', default='data/oxe')
    ap.add_argument('--out-dir', default='data/ckpts')
    ap.add_argument('--steps', type=int, default=None,
                    help="Hard cap on steps. If None, auto-compute as max(--min-steps, --target-epochs × n_train / batch_size), "
                         "capped at --max-steps. Early stopping usually fires before the cap.")
    ap.add_argument('--target-epochs', type=int, default=50,
                    help="Aim for this many epochs through the per-emb training data (subject to min/max-steps).")
    ap.add_argument('--min-steps', type=int, default=2000,
                    help="Don't train fewer than this even for tiny datasets — gives early-stopping room to detect plateau.")
    ap.add_argument('--max-steps', type=int, default=120000,
                    help="Cap on per-emb steps so huge embs (franka, google_robot) don't run forever.")
    ap.add_argument('--patience-evals', type=int, default=6,
                    help="Early stop after this many evaluations with no validation improvement >1e-5.")
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--action-noise', type=float, default=0.1,
                    help="Std of Gaussian noise added to the encoder's input (after precision norm). "
                         "Targets stay clean → denoising-AE regularization. Default 0.1 helps small datasets.")
    ap.add_argument('--time-drop', type=float, default=0.1,
                    help="Probability of zeroing out each timestep in the encoder input. "
                         "Sequence-aware Cutout — forces the codebook to encode local structure.")
    ap.add_argument('--weight-decay', type=float, default=5e-4,
                    help="AdamW weight decay (was 1e-4; stronger default to combat small-data overfit).")
    ap.add_argument('--dropout', type=float, default=0.0,
                    help="Dropout1d after each major encoder/decoder stage. Channel-wise zeroing — "
                         "more impactful for convs than per-element. Identity (no params) when 0. "
                         "Use 0.2-0.4 for embs that overfit despite input regularization (e.g. franka).")
    ap.add_argument('--binary-gripper', action='store_true',
                    help="Treat the last action dim (gripper) as Bernoulli: decoder's last channel "
                         "is a logit trained with BCE (not MSE), read out via sigmoid. Excluded from "
                         "the continuous MSE/NRMSE/NMSE. Encoder input + codes unchanged.")
    ap.add_argument('--grip-weight', type=float, default=1.0,
                    help="Weight on the gripper BCE term (relative to continuous MSE).")
    ap.add_argument('--dead-threshold', type=int, default=5,
                    help="Revive a code after this many consecutive batches unused (VQ1d_EMA). "
                         "Lower => more aggressive revival => higher codebook usage.")
    ap.add_argument('--exclude-datasets', nargs='+', default=None,
                    help="List of dataset names to skip in VAE training. Useful when one robot has "
                         "datasets with incompatible action conventions / scales (e.g., iamlab uses "
                         "absolute-pose-like actions while other franka datasets are deltas).")
    ap.add_argument('--min-pairs', type=int, default=200,
                    help="skip embodiments with fewer than this many (cur, prev) pairs")
    ap.add_argument('--quantizer', choices=['vq', 'fsq'], default='vq',
                    help="vq = learned codebook (VQ1d_EMA); fsq = Finite Scalar Quantization (no codebook collapse)")
    ap.add_argument('--k', type=int, default=128,
                    help="codebook size. With FSQ, valid options: 27, 125, 343, 729, 875, 1125, 1715.")
    ap.add_argument('--only-embodiments', nargs='*', default=None,
                    help="if set, only train VAEs for these embodiments (e.g. widowx google_robot)")
    ap.add_argument('--ckpt-suffix', default='',
                    help="suffix to add to the saved ckpt filename (e.g. '_fsq')")
    ap.add_argument('--convention', choices=['cartesian', 'jointspace'], default='cartesian',
                    help="Action convention to filter to. 'cartesian' = [x,y,z,r,p,y,gripper]; "
                         "'jointspace' = [motor_0..N, gripper]. These describe different action "
                         "spaces and can't be merged in one codebook.")
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    # discover all embodiments present in oxe_root
    embodiments = set()
    for ds_dir in sorted(glob.glob(os.path.join(args.oxe_root, '*'))):
        info_p = os.path.join(ds_dir, 'meta', 'info.json')
        if os.path.isfile(info_p):
            embodiments.add(json.load(open(info_p)).get('robot_type', 'unknown'))
    print(f"discovered embodiments on disk: {sorted(embodiments)}")
    print()

    for emb in sorted(embodiments):
        if args.only_embodiments and emb not in args.only_embodiments:
            continue
        out = os.path.join(args.out_dir, f'oxe_vqvae_{emb}{args.ckpt_suffix}.pt')
        if os.path.exists(out):
            print(f"  [{emb}] SKIP (ckpt exists at {out})"); continue
        print(f"  [{emb}] gathering action chunks...")
        cur, prev, matched = gather_action_pairs_for_embodiment(
            args.oxe_root, emb, convention=args.convention,
            exclude_datasets=set(args.exclude_datasets) if args.exclude_datasets else None)
        if cur is None or cur.shape[0] < args.min_pairs:
            n = 0 if cur is None else cur.shape[0]
            print(f"  [{emb}] only {n} pairs — skipping")
            continue
        print(f"  [{emb}] {cur.shape[0]} pairs from {len(matched)} datasets: {matched}")
        print(f"  [{emb}] action_dim={cur.shape[-1]}, chunk_len={cur.shape[1]}")
        # auto-budget steps if not explicitly given
        if args.steps is None:
            steps = max(args.min_steps, args.target_epochs * cur.shape[0] // args.batch_size)
            steps = min(steps, args.max_steps)
        else:
            steps = args.steps
        print(f"  [{emb}] step budget = {steps}  (target_epochs={args.target_epochs}, "
              f"actual_epochs_at_cap = {steps * args.batch_size / cur.shape[0]:.1f})")
        vae, revin, var_global, train_metrics = train_one(
            emb, cur, prev, steps, args.lr, args.batch_size, dev,
            action_dim=cur.shape[-1], quantizer=args.quantizer, k=args.k,
            patience_evals=args.patience_evals,
            action_noise=args.action_noise, time_drop=args.time_drop,
            weight_decay=args.weight_decay, dropout=args.dropout,
            binary_gripper=args.binary_gripper, dead_threshold=args.dead_threshold,
            grip_weight=args.grip_weight)
        torch.save({'kind': 'vqvae', 'embodiment': emb, 'quantizer': args.quantizer,
                    'convention': args.convention,                          # cartesian / jointspace
                    'vae': vae.state_dict(), 'revin': revin.state_dict(),
                    'action_dim': cur.shape[-1],
                    'action_var_global': var_global.squeeze().cpu(),
                    'norm_reg': 'global',
                    'binary_gripper': bool(args.binary_gripper),
                    'gripper_range': (vae.gripper_range.cpu() if args.binary_gripper
                                      else torch.tensor([0.0, 1.0])),
                    'seq_lens': vae.seq_lens, 'k': vae.vq.K,
                    'n_train_pairs': int(cur.shape[0]), 'source_datasets': matched,
                    'train_metrics': train_metrics}, out)
        fe = train_metrics.get('final_eval') or {}
        print(f"  [{emb}] saved {out}  best_cont_mse={train_metrics['best_val']:.5f} @ step {train_metrics['best_step']}"
              f"  | final NRMSE={fe.get('nrmse', float('nan')):.4f} usage={fe.get('usage', 0)*100:.0f}% "
              f"grip_acc={fe.get('grip_acc', float('nan'))*100:.1f}%\n", flush=True)
        # Explicitly free GPU memory between embodiments (avoid OOM on large embs).
        del vae, revin, var_global, cur, prev
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
