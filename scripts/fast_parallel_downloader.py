#!/usr/bin/env python3
"""Parallel OXE downloader. Improvements over durable_oxe_downloader.py:

  - Skips datasets whose robot is NOT one of the shared VAE's known embodiments
    (5: widowx, google_robot, franka, ur5, jaco_2) — anything else would be
    filtered out at train load anyway, so downloading wastes bandwidth.
  - Skips datasets already >= 80% complete (same is_complete check as before).
  - 4 parallel workers across datasets (each calls snapshot_download with
    max_workers=8 internally → 32 concurrent file transfers max).
  - Per-dataset hard timeout (30 min) — if a download hangs, kill it and move on.
  - HF_HUB_DISABLE_XET=1 to bypass the 1000-req/5-min XET rate limit.
  - One pass through the list; no infinite loop (use cron for repeats).
"""
import os, sys, json, time
os.environ['HF_HUB_DISABLE_XET'] = '1'
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout
from huggingface_hub import snapshot_download

DEST = '/home/research/Projects/BBGr/BabyGR00T/data/oxe'
TOKEN = os.environ.get('HF_TOKEN')
PER_DATASET_TIMEOUT_S = 1800           # 30 min
N_PARALLEL = 2                         # was 4, but hit 429 — keep concurrency modest
COMPLETENESS_THRESHOLD = 0.80

# robot types we want to download. By default = all 11 known embodiments so
# we can RETRAIN the shared VAE to cover them. (Previously filtered to only the
# 5 our current VAE supports.)
USABLE_ROBOTS = {
    'widowx', 'google_robot', 'franka', 'ur5', 'jaco_2',                # current VAE-supported (5)
    'kuka_iiwa', 'xarm', 'sawyer', 'hello_stretch', 'fanuc_mate', 'dlr_edan'  # new (6)
}

# Reuse the same 36-repo list as durable_oxe_downloader.py
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
    """(is_complete_bool, size_gb, n_mp4s, expected_total)."""
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
    complete = expected is None or (n_mp4 >= COMPLETENESS_THRESHOLD * expected)
    return (complete and n_mp4 > 0, sz, n_mp4, expected)


def get_robot_type(repo: str) -> str:
    """Fetch just info.json to read robot_type. ~1 request, fast."""
    from huggingface_hub import hf_hub_download
    try:
        info_p = hf_hub_download(repo_id=repo, repo_type='dataset',
                                 filename='meta/info.json', token=TOKEN)
        info = json.load(open(info_p))
        return info.get('robot_type', 'unknown')
    except Exception as e:
        return f'<err:{type(e).__name__}>'


def download_one(repo: str) -> str:
    """Snapshot-download one dataset. Called via ProcessPoolExecutor."""
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    name = repo.split('/')[-1]
    local = os.path.join(DEST, name)
    if os.path.islink(local):
        return f"SKIP-LINK {name}"
    complete, sz, nv, exp = is_complete(local)
    if complete:
        return f"DONE {name}: {sz:.2f}GB ({nv}/{exp} mp4s, ≥80%)"
    try:
        t0 = time.time()
        snapshot_download(repo_id=repo, repo_type='dataset', local_dir=local,
                          max_workers=8, etag_timeout=120, token=TOKEN)
        c, sz, nv, exp = is_complete(local)
        return f"COMPLETED {name}: {sz:.2f}GB ({nv}/{exp or '?'} mp4s) [{time.time()-t0:.0f}s]"
    except Exception as e:
        return f"FAIL {name}: {type(e).__name__}: {str(e)[:120]}"


def main():
    print(f"[fast-dl] starting; dest={DEST}; token={'set' if TOKEN else 'MISSING'}", flush=True)
    print(f"[fast-dl] XET disabled, {N_PARALLEL} workers, {PER_DATASET_TIMEOUT_S}s/dataset timeout")
    print(f"[fast-dl] USABLE robots: {USABLE_ROBOTS}\n", flush=True)

    # Step 1: probe robot types for all datasets (single-threaded, fast)
    print("=" * 60)
    print("STEP 1: probe robot_types (single-threaded; meta/info.json only)")
    print("=" * 60)
    to_fetch = []                                              # (repo, robot)
    for repo in DSETS:
        name = repo.split('/')[-1]
        local = os.path.join(DEST, name)
        complete, sz, nv, exp = is_complete(local)
        if complete:
            print(f"  SKIP-DONE  {name:<50s} ({sz:.1f}GB, {nv}/{exp} mp4s)")
            continue
        # need to know robot type to decide if usable
        robot = get_robot_type(repo)
        if robot not in USABLE_ROBOTS:
            print(f"  SKIP-ROBOT {name:<50s} (robot='{robot}', not in usable set)")
            continue
        print(f"  QUEUE      {name:<50s} (robot={robot})  start={sz:.1f}GB existing")
        to_fetch.append((repo, robot))

    print(f"\n→ {len(to_fetch)} datasets queued for download")
    if not to_fetch:
        print("Nothing to do — all usable datasets are complete."); return

    # Step 2: parallel download
    print("\n" + "=" * 60)
    print(f"STEP 2: parallel download ({N_PARALLEL} workers)")
    print("=" * 60)
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
        futures = {ex.submit(download_one, repo): repo for repo, _ in to_fetch}
        for fut in list(futures.keys()):                       # poll in submitted order
            repo = futures[fut]
            try:
                result = fut.result(timeout=PER_DATASET_TIMEOUT_S)
            except FTimeout:
                fut.cancel()
                result = f"TIMEOUT {repo.split('/')[-1]} after {PER_DATASET_TIMEOUT_S}s"
            except Exception as e:
                result = f"WORKER-ERR {repo.split('/')[-1]}: {type(e).__name__}: {e}"
            elapsed = time.time() - t_start
            print(f"  [{elapsed/60:.1f}min] {result}", flush=True)

    print(f"\n[fast-dl] done in {(time.time()-t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
