#!/usr/bin/env python3
"""Diagnose franka VAE overfit. Per-dataset action stats + cross-dataset distribution overlap."""
import os, glob, json
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, torch
import pyarrow.parquet as pq

ROOT = 'data/oxe'


def gather_per_dataset(robot='franka', chunk_len=16):
    by_ds = {}
    for ds_dir in sorted(glob.glob(os.path.join(ROOT, '*'))):
        info_p = os.path.join(ds_dir, 'meta', 'info.json')
        if not os.path.isfile(info_p): continue
        info = json.load(open(info_p))
        if info.get('robot_type') != robot: continue
        names = info.get('features', {}).get('action', {}).get('names', {}).get('motors', [])
        ad = info.get('features', {}).get('action', {}).get('shape', [None])[0]
        actions = []
        for pq_path in sorted(glob.glob(os.path.join(ds_dir, 'data', 'chunk-*', '*.parquet'))):
            try:
                t = pq.read_table(pq_path, columns=['action'])
                a = np.stack(t.column('action').to_pylist())
                actions.append(a)
            except: pass
        if not actions: continue
        a = np.concatenate(actions, axis=0)
        by_ds[os.path.basename(ds_dir)] = dict(
            n=a.shape[0], ad=ad,
            mean=a.mean(0), std=a.std(0),
            p1=np.percentile(a, 1, 0), p99=np.percentile(a, 99, 0),
            names=names,
            actions=a if a.shape[0] < 50000 else a[np.random.choice(a.shape[0], 50000, replace=False)],
        )
    return by_ds


def main():
    print("Gathering per-dataset franka action stats...")
    by_ds = gather_per_dataset()
    print(f"  {len(by_ds)} franka datasets")
    print()

    # Print per-dim mean ± std for each
    print("PER-DIM MEAN ± STD  (rows = dataset, cols = action dim)")
    print(f"  {'dataset':<45s} {'n':>7s}  d0±std        d1±std        d2±std        d3±std        d4±std        d5±std        d6±std (gripper)")
    for name in sorted(by_ds):
        d = by_ds[name]
        mu = d['mean']; sd = d['std']
        stats = ' '.join(f"{m:+.2f}±{s:.2f}" for m, s in zip(mu, sd))
        print(f"  {name:<45s} {d['n']:>7d}  {stats}")
    print()

    # Identify outlier datasets: which dim's mean/std is most different?
    print("ACTION CONVENTION ('names' field):")
    for name in sorted(by_ds):
        n = by_ds[name].get('names', [])
        print(f"  {name:<45s} {n}")
    print()

    # Action magnitude per dataset (||a|| typical scale)
    print("ACTION MAGNITUDE STATS:")
    print(f"  {'dataset':<45s} {'|a|_mean':>10s} {'|a|_99p':>10s} {'|d0..2|_typ':>13s} {'|d3..5|_typ':>13s} {'|d6|':>8s}")
    for name in sorted(by_ds):
        a = by_ds[name]['actions']
        norms = np.linalg.norm(a, axis=1)
        trans = np.linalg.norm(a[:, :3], axis=1)
        rot = np.linalg.norm(a[:, 3:6], axis=1)
        grip = np.abs(a[:, 6])
        print(f"  {name:<45s} {norms.mean():>10.3f} {np.percentile(norms,99):>10.3f} "
              f"{trans.mean():>13.4f} {rot.mean():>13.4f} {grip.mean():>8.3f}")


if __name__ == '__main__':
    main()
