#!/usr/bin/env python3
"""Train the AgiBot arms-only action VQ-VAE from proprioception (no video needed).

AgiBot World stores proprio in per-episode HDF5 (proprio_stats.h5). We take the
ARMS-ONLY action = hstack[ action/joint/position (14, rad), action/effector/position
(2, grippers) ] = 16-D, joint-space, and tokenize it with the same ActionVQVAE1d
machinery as the OXE per-emb VAEs (EMA codebook + dead-code revival + optional
binary gripper). Reuses train_one() so the arch/regularization match.

Run AFTER the proprio tar is extracted:
  tar xf data/agibot_raw/proprio_stats/648533-923022.tar -C data/agibot_proprio
  python -m scripts.train_agibot_vae --proprio-dir data/agibot_proprio \
      --task-json data/agibot_raw/task_info/task_354.json --k 256

Gripper note: AgiBot has TWO grippers (effector dims 14,15). --binary-gripper here
binarizes only the LAST dim (right gripper) via train_one; pass --grip-stats first to
inspect whether the effectors are bimodal before deciding. Default: continuous.
"""
import os, sys, glob, json, argparse, tarfile
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch, h5py
from scripts.train_oxe_vaes import train_one

ARM_JOINT_KEY = 'action/joint/position'        # (T,14)  7+7 dual-arm joint angles (rad)
ARM_EFFECTOR_KEY = 'action/effector/position'  # (T,2)   left/right gripper


def episode_ids_from_task(task_json):
    info = json.load(open(task_json))
    # task_info JSON is a list of per-episode dicts
    ids = []
    for e in info:
        for k in ('episode_id', 'episode_index', 'eid'):
            if k in e:
                ids.append(int(e[k])); break
    return set(ids)


def gather_arms(proprio_dir, ep_filter=None, chunk_len=16, max_eps=None):
    """Walk proprio h5s → (cur, prev) 16-D arms-action chunk pairs."""
    h5s = sorted(glob.glob(os.path.join(proprio_dir, '**', 'proprio_stats.h5'), recursive=True))
    cur_list, prev_list = [], []
    n_used = 0
    for p in h5s:
        # episode id is the parent-dir name (…/<episode_id>/proprio_stats.h5)
        try:
            eid = int(os.path.basename(os.path.dirname(p)))
        except ValueError:
            eid = None
        if ep_filter is not None and eid is not None and eid not in ep_filter:
            continue
        try:
            with h5py.File(p, 'r') as f:
                aj = np.asarray(f[ARM_JOINT_KEY], dtype=np.float32)        # (T,14)
                ae = np.asarray(f[ARM_EFFECTOR_KEY], dtype=np.float32)     # (T,2)
        except Exception:
            continue
        if aj.ndim != 2 or aj.shape[1] != 14 or ae.shape[1] != 2:
            continue
        arms = np.hstack([aj, ae])                                        # (T,16)
        T = arms.shape[0]
        if T < 2 * chunk_len:
            continue
        n_full = T // chunk_len
        ch = torch.from_numpy(arms[:n_full * chunk_len].reshape(n_full, chunk_len, 16))
        cur_list.append(ch[1:]); prev_list.append(ch[:-1])
        n_used += 1
        if max_eps and n_used >= max_eps:
            break
    if not cur_list:
        return None, None, 0
    return torch.cat(cur_list, 0), torch.cat(prev_list, 0), n_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proprio-dir', required=True, help='dir with extracted proprio_stats.h5 (recursive)')
    ap.add_argument('--task-json', default=None, help='restrict to a task_info json\'s episodes (else all)')
    ap.add_argument('--out', default='data/ckpts/oxe_vqvae_agibot.pt')
    ap.add_argument('--k', type=int, default=256)
    ap.add_argument('--steps', type=int, default=12000)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dead-threshold', type=int, default=3)
    ap.add_argument('--binary-gripper', action='store_true')
    ap.add_argument('--grip-weight', type=float, default=1.0)
    ap.add_argument('--action-noise', type=float, default=0.1)
    ap.add_argument('--time-drop', type=float, default=0.1)
    ap.add_argument('--max-eps', type=int, default=None)
    ap.add_argument('--grip-stats', action='store_true', help='just print effector distribution and exit')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ep_filter = episode_ids_from_task(args.task_json) if args.task_json else None
    print(f"gathering arms actions (filter={'task '+str(len(ep_filter))+' eps' if ep_filter else 'ALL'}) ...", flush=True)
    cur, prev, n_eps = gather_arms(args.proprio_dir, ep_filter, max_eps=args.max_eps)
    if cur is None:
        print("No episodes gathered — check --proprio-dir / extraction."); return
    print(f"  {cur.shape[0]} chunk-pairs from {n_eps} episodes; arms action_dim=16 "
          f"(joint14+effector2)", flush=True)

    # effector (gripper) distribution — decide binary vs continuous
    eff = cur[..., 14:16].reshape(-1, 2)
    print(f"  effector L: [{eff[:,0].min():.1f},{eff[:,0].max():.1f}] mean={eff[:,0].mean():.1f} "
          f"| R: [{eff[:,1].min():.1f},{eff[:,1].max():.1f}] mean={eff[:,1].mean():.1f}", flush=True)
    # bimodality check: fraction near the two extremes
    for i, nm in [(0,'L'),(1,'R')]:
        v = eff[:, i]; lo, hi = v.min(), v.max(); mid = (lo+hi)/2
        frac_extreme = ((v < lo+0.1*(hi-lo)) | (v > hi-0.1*(hi-lo))).float().mean()
        print(f"    gripper {nm}: {frac_extreme*100:.0f}% of samples in outer 10% (>~80% => binary-like)", flush=True)
    if args.grip_stats:
        return

    vae, revin, var_global, mtr = train_one(
        'agibot', cur, prev, args.steps, args.lr, args.batch_size, dev,
        action_dim=16, quantizer='vq', k=args.k, dead_threshold=args.dead_threshold,
        binary_gripper=args.binary_gripper, grip_weight=args.grip_weight,
        action_noise=args.action_noise, time_drop=args.time_drop)
    torch.save({'kind': 'vqvae', 'embodiment': 'agibot', 'quantizer': 'vq',
                'convention': 'jointspace', 'vae': vae.state_dict(), 'revin': revin.state_dict(),
                'action_dim': 16, 'action_var_global': var_global.squeeze().cpu(),
                'norm_reg': 'global', 'binary_gripper': bool(args.binary_gripper),
                'gripper_range': (vae.gripper_range.cpu() if args.binary_gripper else torch.tensor([0., 1.])),
                'seq_lens': vae.seq_lens, 'k': vae.vq.K, 'n_train_pairs': int(cur.shape[0]),
                'train_metrics': mtr}, args.out)
    fe = mtr.get('final_eval') or {}
    print(f"saved {args.out}  best_cont_mse={mtr['best_val']:.5f}  NRMSE={fe.get('nrmse',float('nan')):.4f} "
          f"usage={fe.get('usage',0)*100:.0f}% grip_acc={fe.get('grip_acc',float('nan'))*100:.1f}%", flush=True)


if __name__ == '__main__':
    main()
