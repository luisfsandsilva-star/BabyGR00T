#!/usr/bin/env python3
"""Build GLOBAL train/val episode-id splits across ALL converted AgiBot tasks.

Episode ids are globally unique across AgiBot tasks, so train_oxe's `ep in eps_set`
filter works against one combined list. We split PER TASK (deterministic shuffle,
val_frac held out) so every task contributes held-out val episodes — true within-task
generalization, not a whole-task holdout.

  python -m scripts.make_agibot_splits --val-frac 0.15 --seed 0
"""
import os, glob, json, argparse, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oxe-root', default='data/oxe')
    ap.add_argument('--glob', default='agibot_task*')
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-out', default='data/splits/agibot_train_eps.json')
    ap.add_argument('--val-out', default='data/splits/agibot_val_eps.json')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train, val = [], []
    rows = []
    for d in sorted(glob.glob(os.path.join(args.oxe_root, args.glob))):
        ep_file = os.path.join(d, 'meta', 'episodes.jsonl')
        if not os.path.exists(ep_file):
            continue
        eps = [json.loads(l)['episode_index'] for l in open(ep_file)]
        eps = sorted(set(eps))
        rng.shuffle(eps)
        nv = max(1, int(round(len(eps) * args.val_frac)))
        v, t = eps[:nv], eps[nv:]
        val += v; train += t
        rows.append((os.path.basename(d), len(eps), len(t), len(v)))

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    json.dump(sorted(train), open(args.train_out, 'w'))
    json.dump(sorted(val), open(args.val_out, 'w'))

    print(f"{'task':>22} {'eps':>5} {'train':>6} {'val':>5}")
    for name, n, t, v in rows:
        print(f"{name:>22} {n:5d} {t:6d} {v:5d}")
    print(f"{'TOTAL':>22} {sum(r[1] for r in rows):5d} {len(train):6d} {len(val):5d}")
    print(f"wrote {args.train_out} ({len(train)}) + {args.val_out} ({len(val)})")


if __name__ == '__main__':
    main()
