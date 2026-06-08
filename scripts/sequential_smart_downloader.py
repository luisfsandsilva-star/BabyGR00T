#!/usr/bin/env python3
"""Calm sequential OXE downloader.

Lessons from the fast parallel attempt:
  - HF rate-limits us hard at N=2-4 parallel + max_workers=8 internal.
  - Bridge_orig and fractal20220817 are HUGE (9-17 GB), already partially
    downloaded, and trip XET timeouts — skip them entirely.
  - Small datasets (<1 GB) complete in <60s when not competing.

This downloader: single-threaded (1 dataset at a time), max_workers=4 internal,
XET disabled, with explicit skiplist for the time-sink datasets. Runs CPU-only
(no GPU contention with v13 training).
"""
import os, time, json
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''        # no GPU
from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

DEST = '/home/research/Projects/BBGr/BabyGR00T/data/oxe'
TOKEN = os.environ.get('HF_TOKEN')

# Skip these — already large & partial; bandwidth-inefficient to "complete"
# them when we already have 38% (bridge) or 79% (fractal). Plenty of data already.
SKIP_REPOS = {
    'IPEC-COMMUNITY/bridge_orig_lerobot',
    'IPEC-COMMUNITY/fractal20220817_data_lerobot',
}

USABLE_ROBOTS = {
    'widowx', 'google_robot', 'franka', 'ur5', 'jaco_2',
    'kuka_iiwa', 'xarm', 'sawyer', 'hello_stretch', 'fanuc_mate', 'dlr_edan',
}

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
    return (expected and n_mp4 >= 0.8 * expected, sz, n_mp4, expected)


def probe_robot(repo: str) -> str:
    try:
        info_p = hf_hub_download(repo_id=repo, repo_type='dataset',
                                 filename='meta/info.json', token=TOKEN)
        return json.load(open(info_p)).get('robot_type', 'unknown')
    except Exception as e:
        return f'<err:{type(e).__name__}>'


def estimate_size(repo: str) -> float:
    """Estimate dataset size in GB by reading info.json + a guess based on episode count."""
    try:
        info_p = hf_hub_download(repo_id=repo, repo_type='dataset',
                                 filename='meta/info.json', token=TOKEN)
        info = json.load(open(info_p))
        n_eps = info.get('total_episodes', 100)
        # very rough: 1 MB per episode for most LeRobot datasets, more for image-heavy
        return n_eps * 0.001  # GB estimate
    except Exception:
        return 1.0


def main():
    print(f"[seq-dl] starting, dest={DEST}, token={'set' if TOKEN else 'missing'}", flush=True)
    print(f"[seq-dl] XET disabled, single-thread, max_workers=4 internal\n", flush=True)

    # Phase 1: classify
    queue = []   # (size_gb_estimate, repo, robot)
    skip_huge = []; skip_done = []; skip_unusable = []
    for repo in DSETS:
        name = repo.split('/')[-1]
        local = os.path.join(DEST, name)
        if repo in SKIP_REPOS:
            skip_huge.append(repo); continue
        complete, sz, nv, exp = is_complete(local)
        if complete:
            skip_done.append((repo, sz, nv)); continue
        robot = probe_robot(repo)
        if robot not in USABLE_ROBOTS:
            skip_unusable.append((repo, robot)); continue
        size_g = estimate_size(repo)
        queue.append((size_g, repo, robot))

    queue.sort()   # small first
    print(f"  SKIP-HUGE      ({len(skip_huge)}): {[r.split('/')[-1] for r in skip_huge]}")
    print(f"  SKIP-DONE      ({len(skip_done)}): {[r.split('/')[-1] for r,_,_ in skip_done]}")
    print(f"  SKIP-UNUSABLE  ({len(skip_unusable)}): {[r.split('/')[-1]+':'+rb for r,rb in skip_unusable]}")
    print(f"\n  QUEUED ({len(queue)}, smallest first):")
    for sz, repo, robot in queue:
        print(f"    ~{sz:.2f} GB  {robot:<14s} {repo.split('/')[-1]}")

    # Phase 2: download sequentially
    print(f"\n[seq-dl] starting downloads...\n", flush=True)
    for i, (sz, repo, robot) in enumerate(queue):
        name = repo.split('/')[-1]
        local = os.path.join(DEST, name)
        t0 = time.time()
        print(f"  [{i+1}/{len(queue)}] {name} ({robot}, ~{sz:.2f}GB est) ... ", end='', flush=True)
        try:
            snapshot_download(repo_id=repo, repo_type='dataset', local_dir=local,
                              max_workers=4, etag_timeout=120, token=TOKEN)
            c, real_sz, nv, exp = is_complete(local)
            print(f"OK {real_sz:.2f}GB ({nv}/{exp or '?'} mp4s) [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:
            print(f"FAIL {type(e).__name__}: {str(e)[:80]} [{time.time()-t0:.0f}s]", flush=True)
            # if rate-limited, sleep before next attempt
            if '429' in str(e):
                print(f"  → rate-limited, sleeping 120s", flush=True)
                time.sleep(120)
    print(f"\n[seq-dl] all done")


if __name__ == '__main__':
    main()
