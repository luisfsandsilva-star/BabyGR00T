#!/usr/bin/env python3
"""Watch the sentinel log; when val_acc ≥ TARGET for 2 consecutive checks, SIGTERM training.

Polls sentinel log every 30s. Tracks the last 2 val_acc readings observed. When BOTH
are ≥ TARGET, sends SIGTERM to TRAIN_PID. atexit handler in training will save a
final ckpt, then exit cleanly. Best ckpt remains preserved by sentinel.
"""
import os, sys, time, re, signal, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--sentinel-log', default='/tmp/sentinel_widowx_v3_resumed.log')
ap.add_argument('--train-pid', type=int, required=True)
ap.add_argument('--target', type=float, default=90.0)
ap.add_argument('--consecutive', type=int, default=2)
ap.add_argument('--poll', type=int, default=30)
args = ap.parse_args()

PAT = re.compile(r'step (\d+): val_acc=([\d.]+)%')
print(f"[stop-90] watching {args.sentinel_log}", flush=True)
print(f"[stop-90]   train PID: {args.train_pid}", flush=True)
print(f"[stop-90]   target: {args.target}% for {args.consecutive} consecutive checks", flush=True)

seen_steps = set()   # avoid acting on a single check twice
val_history = []      # list of (step, val_acc) in chronological order

while True:
    # is training alive?
    try: os.kill(args.train_pid, 0)
    except ProcessLookupError:
        print(f"[stop-90] training PID {args.train_pid} exited — watcher done.", flush=True)
        sys.exit(0)
    # parse sentinel log fresh each poll
    if not os.path.exists(args.sentinel_log):
        time.sleep(args.poll); continue
    new_in_round = []
    with open(args.sentinel_log) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                step = int(m.group(1)); va = float(m.group(2))
                if step not in seen_steps:
                    seen_steps.add(step)
                    val_history.append((step, va))
                    new_in_round.append((step, va))
    # check most recent N
    if len(val_history) >= args.consecutive:
        recent = val_history[-args.consecutive:]
        all_above = all(va >= args.target for _, va in recent)
        if all_above:
            print(f"[stop-90] CONDITION MET — {args.consecutive} consecutive val_acc ≥ {args.target}%:", flush=True)
            for st, va in recent:
                print(f"  step {st}: val_acc={va:.2f}%", flush=True)
            print(f"[stop-90] sending SIGTERM to PID {args.train_pid}...", flush=True)
            try:
                os.kill(args.train_pid, signal.SIGTERM)
                print(f"[stop-90] SIGTERM sent. atexit will save final ckpt.", flush=True)
            except ProcessLookupError:
                print(f"[stop-90] training already gone.", flush=True)
            sys.exit(0)
    if new_in_round:
        for st, va in new_in_round:
            print(f"[stop-90] saw step {st}: val_acc={va:.2f}%  (need {args.consecutive} consecutive ≥ {args.target}%)", flush=True)
    time.sleep(args.poll)
