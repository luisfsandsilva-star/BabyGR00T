#!/usr/bin/env python3
"""Durable OXE dataset downloader — runs forever, retries indefinitely, never gives up.

Strategy:
  * Walks all 36 IPEC-COMMUNITY OXE LeRobot datasets
  * For each, tries `snapshot_download` with single worker
  * On 429 → backoff (5 → 10 → 30 → 60 min, capped at 60)
  * On any other error → log and continue to next dataset
  * After one pass through all datasets, sleeps 5 min and starts over
  * Skips datasets that have meta/info.json + at least 1 mp4 (assumed complete enough)

Run with: `HF_TOKEN=... python -u scripts/durable_oxe_downloader.py 2>&1 | tee /tmp/oxe_dl.log &`
The process is idempotent — kill and restart any time.
"""
import os, sys, time, json
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download

TOKEN = os.environ.get('HF_TOKEN')
DEST = os.environ.get('OXE_DEST', '/home/research/Projects/BBGr/BabyGR00T/data/oxe')

DSETS = [
    'IPEC-COMMUNITY/bridge_orig_lerobot',
    'IPEC-COMMUNITY/fractal20220817_data_lerobot',
    'IPEC-COMMUNITY/bc_z_lerobot',
    'IPEC-COMMUNITY/fmb_dataset_lerobot',
    'IPEC-COMMUNITY/droid_lerobot',
    'IPEC-COMMUNITY/furniture_bench_dataset_lerobot',
    'IPEC-COMMUNITY/toto_lerobot',
    'IPEC-COMMUNITY/dobbe_lerobot',
    'IPEC-COMMUNITY/stanford_hydra_dataset_lerobot',
    'IPEC-COMMUNITY/berkeley_autolab_ur5_lerobot',
    'IPEC-COMMUNITY/roboturk_lerobot',
    'IPEC-COMMUNITY/austin_sailor_dataset_lerobot',
    'IPEC-COMMUNITY/berkeley_rpt_lerobot',
    'IPEC-COMMUNITY/iamlab_cmu_pickup_insert_lerobot',
    'IPEC-COMMUNITY/utaustin_mutex_lerobot',
    'IPEC-COMMUNITY/taco_play_lerobot',
    'IPEC-COMMUNITY/cmu_play_fusion_lerobot',
    'IPEC-COMMUNITY/viola_lerobot',
    'IPEC-COMMUNITY/austin_sirius_dataset_lerobot',
    'IPEC-COMMUNITY/kuka_lerobot',
    'IPEC-COMMUNITY/language_table_lerobot',
    'IPEC-COMMUNITY/jaco_play_lerobot',
    'IPEC-COMMUNITY/libero_90_no_noops_lerobot',
    'IPEC-COMMUNITY/berkeley_cable_routing_lerobot',
    'IPEC-COMMUNITY/austin_buds_dataset_lerobot',
    'IPEC-COMMUNITY/berkeley_mvp_lerobot',
    'IPEC-COMMUNITY/berkeley_fanuc_manipulation_lerobot',
    'IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot',
    'IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot',
    'IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot',
    'IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot',
    'IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot',
    'IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot',
    'IPEC-COMMUNITY/cmu_stretch_lerobot',
    'IPEC-COMMUNITY/dlr_edan_shared_control_lerobot',
    'IPEC-COMMUNITY/ucsd_kitchen_dataset_lerobot',
]


def is_complete(local: str) -> tuple:
    """Returns (is_complete_bool, size_gb, n_mp4s, expected_total)."""
    meta_p = os.path.join(local, 'meta', 'info.json')
    if not os.path.isfile(meta_p): return (False, 0.0, 0, None)
    try:
        ij = json.load(open(meta_p))
        expected = ij.get('total_videos', None) or ij.get('total_episodes', None)
    except Exception:
        expected = None
    n_mp4 = sum(1 for r, _, fs in os.walk(os.path.join(local, 'videos'))
                for f in fs if f.endswith('.mp4'))
    sz = sum(os.path.getsize(os.path.join(r, f))
             for r, _, fs in os.walk(local) for f in fs) / 1e9
    # treat "complete" as having at least 80% of expected videos
    complete = expected is None or (n_mp4 >= 0.8 * expected)
    return (complete and n_mp4 > 0, sz, n_mp4, expected)


def pull_one(repo: str, retry_state: dict) -> str:
    name = repo.split('/')[1]
    local = os.path.join(DEST, name)
    if os.path.islink(local):
        # if it's a symlink (e.g., bridge to Desktop), skip — symlinks are user-managed
        return f"SKIP-LINK {name}"
    complete, sz, nv, exp = is_complete(local)
    if complete:
        return f"DONE {name}: {sz:.2f}GB ({nv}/{exp} mp4s)"
    try:
        t0 = time.time()
        snapshot_download(repo_id=repo, repo_type='dataset', local_dir=local,
                          max_workers=8, etag_timeout=180, token=TOKEN)
        retry_state[name] = 0     # reset backoff on success
        c, sz, nv, exp = is_complete(local)
        return f"PROGRESS {name}: {sz:.2f}GB ({nv}/{exp or '?'} mp4s) [{time.time()-t0:.0f}s]"
    except Exception as e:
        # exponential backoff per-dataset on repeated failure
        retry_state[name] = retry_state.get(name, 0) + 1
        wait = min(60 * 2 ** min(retry_state[name], 6), 60 * 60)
        msg = str(e)[:100]
        return f"FAIL {name} (retry #{retry_state[name]}, sleeping {wait}s next pass): {msg}"


def main():
    print(f"[durable-dl] starting; dest={DEST}; token={'set' if TOKEN else 'missing'}", flush=True)
    if not TOKEN:
        print("[durable-dl] HF_TOKEN env var not set — falling back to anonymous (will hit 429 fast)", flush=True)
    retry_state = {}
    pass_n = 0
    while True:
        pass_n += 1
        t_pass = time.time()
        n_done = n_progress = n_fail = n_skip = 0
        print(f"\n[durable-dl] === pass {pass_n} starting at {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
        for repo in DSETS:
            msg = pull_one(repo, retry_state)
            print(f"  {msg}", flush=True)
            if msg.startswith('DONE'): n_done += 1
            elif msg.startswith('PROGRESS'): n_progress += 1
            elif msg.startswith('FAIL'): n_fail += 1
            else: n_skip += 1
            time.sleep(10)         # be gentle between datasets
        # total disk used
        sz = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(DEST) for f in fs) / 1e9
        print(f"[durable-dl] pass {pass_n} complete in {time.time()-t_pass:.0f}s: "
              f"done={n_done}, progress={n_progress}, fail={n_fail}, skip={n_skip}, total {sz:.1f}GB", flush=True)
        if n_done == len(DSETS) - n_skip:
            print(f"[durable-dl] ALL DATASETS COMPLETE — sleeping 1h before next sanity-check pass", flush=True)
            time.sleep(3600)
        else:
            print(f"[durable-dl] {len(DSETS) - n_done - n_skip} still incomplete; sleeping 5min before next pass", flush=True)
            time.sleep(300)


if __name__ == '__main__':
    main()
