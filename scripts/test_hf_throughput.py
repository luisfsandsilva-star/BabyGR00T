#!/usr/bin/env python3
"""Proper streaming throughput test: how many sustained episode fetches/sec can we get
from HF before/after rate limit, with and without XET protocol.

Picks a non-locally-cached dataset, fetches N=50 episodes (parquet + mp4), and
reports timing + when rate limit kicks in.
"""
import os, sys, time, json
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

# Pick a dataset NOT in local data/oxe (verify before running)
REPO = 'IPEC-COMMUNITY/nyu_franka_play_dataset_lerobot'
N_EPISODES = 50


def fetch_meta(repo, disable_xet=False):
    if disable_xet:
        os.environ['HF_HUB_DISABLE_XET'] = '1'
    else:
        os.environ.pop('HF_HUB_DISABLE_XET', None)
    info_p = hf_hub_download(repo, 'meta/info.json', repo_type='dataset')
    eps_p = hf_hub_download(repo, 'meta/episodes.jsonl', repo_type='dataset')
    snapshot = os.path.dirname(os.path.dirname(info_p))
    info = json.load(open(info_p))
    eps = []
    for line in open(eps_p):
        d = json.loads(line)
        ep_id = d.get('episode_index')
        if ep_id is not None: eps.append(ep_id)
    return info, eps, snapshot


def run_test(label, disable_xet=False):
    print(f"\n=== {label} ===")
    info, eps, snap = fetch_meta(REPO, disable_xet=disable_xet)
    cams = sorted([k for k in info.get('features', {}) if 'image' in k.lower() or 'video' in k.lower()])
    cam = cams[0]
    chunks_size = info.get('chunks_size', 1000)
    data_t = info.get('data_path', 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet')
    vid_t = info.get('video_path', 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4')
    print(f"  repo: {REPO}; total eps: {len(eps)}; cam: {cam}")

    t0 = time.perf_counter()
    timings = []
    rate_limit_hit = False
    n_success = 0
    for i, ep_id in enumerate(eps[:N_EPISODES]):
        cc = ep_id // chunks_size
        pq_rel = data_t.format(episode_chunk=cc, episode_index=ep_id)
        vid_rel = vid_t.format(episode_chunk=cc, video_key=cam, episode_index=ep_id)
        t_ep = time.perf_counter()
        try:
            hf_hub_download(REPO, pq_rel, repo_type='dataset')
            hf_hub_download(REPO, vid_rel, repo_type='dataset')
            n_success += 1
        except HfHubHTTPError as e:
            if '429' in str(e):
                print(f"  RATE LIMIT after {n_success} episodes ({time.perf_counter()-t0:.1f}s, "
                      f"~{n_success/(time.perf_counter()-t0)*60:.1f} eps/min)")
                rate_limit_hit = True
                break
            else:
                print(f"  ERROR on ep {ep_id}: {e}")
                continue
        timings.append(time.perf_counter() - t_ep)
        if (i + 1) % 10 == 0:
            cur = time.perf_counter() - t0
            print(f"  ...{i+1} eps in {cur:.1f}s  ({(i+1)/cur:.2f} eps/sec)")
    total = time.perf_counter() - t0
    print(f"  total: {n_success} episodes in {total:.1f}s = {n_success/total:.2f} eps/sec "
          f"= {n_success/total*60:.0f} eps/min")
    print(f"  median per-ep fetch: {sorted(timings)[len(timings)//2]:.3f}s" if timings else "  no timings")
    return dict(n=n_success, sec=total, eps_per_sec=n_success/total if total else 0,
                rate_limit=rate_limit_hit)


if __name__ == '__main__':
    # check it's not locally cached
    local = f"/home/research/Projects/BBGr/BabyGR00T/data/oxe/{REPO.split('/')[-1]}"
    if os.path.isdir(local):
        print(f"WARN: {local} already exists locally — test won't be cold-cache")
    a = run_test("WITH XET (default)", disable_xet=False)
    print("\n  (sleeping 60s for partial rate-limit recovery before next test)")
    time.sleep(60)
    b = run_test("WITHOUT XET (HF_HUB_DISABLE_XET=1)", disable_xet=True)

    print("\n=== SUMMARY ===")
    print(f"  with XET:    {a['eps_per_sec']:.2f} eps/sec  ({'rate limited' if a['rate_limit'] else 'completed'})")
    print(f"  without XET: {b['eps_per_sec']:.2f} eps/sec  ({'rate limited' if b['rate_limit'] else 'completed'})")
    print(f"\n  v13 training needs: 0.7 step/s × batch 256 = 179 eps/sec (cold) to keep up.")
    print(f"  Once warm (cached), local SSD reads, so streaming throughput only matters for first-pass.")
