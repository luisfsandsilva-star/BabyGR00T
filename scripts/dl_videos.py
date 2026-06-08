#!/usr/bin/env python3
"""Throttled, resumable downloader for bridge image_0 videos (respects HF's
5000-resolver-requests / 5-min limit). Skips files already present, rate-limits
to ~13 req/s, retries 429 with exponential backoff. Resumable: just rerun.

Run: source ~/.hf_env && .venv/bin/python scripts/dl_videos.py
"""
import os, sys, json, time, threading
sys.path.insert(0, '/home/research/Projects/BBGr/BabyGR00T'); os.chdir('/home/research/Projects/BBGr/BabyGR00T')
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download
from babygroot_strm.multi_oxe import load_dataset_spec, _episode_paths

TOK = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
BR = 'data/oxe/bridge_orig_lerobot'
sp = load_dataset_spec(BR, chunk_len=16, lookback=16, chunk_stride=4)
missing = [ep for ep in range(53192) if not os.path.isfile(_episode_paths(sp, ep)[1])]
print(f"missing image_0 videos: {len(missing)}", flush=True)

_lock = threading.Lock(); _next = [time.time()]; RATE = 13.0      # req/sec (limit is ~16.7)
def throttle():
    with _lock:
        now = time.time(); w = _next[0] - now
        if w > 0: time.sleep(w)
        _next[0] = max(now, _next[0]) + 1.0 / RATE

def get(ep):
    rel = f'videos/chunk-{ep//1000:03d}/observation.images.image_0/episode_{ep:06d}.mp4'
    for attempt in range(8):
        throttle()
        try:
            hf_hub_download(repo_id='IPEC-COMMUNITY/bridge_orig_lerobot', repo_type='dataset',
                            filename=rel, local_dir=BR, token=TOK)
            return 'ok'
        except Exception as e:
            s = str(e)
            if '429' in s or 'Too Many Requests' in s or 'rate limit' in s.lower():
                time.sleep(min(240, 30 * (2 ** attempt)))      # backoff on rate-limit
                continue
            if '404' in s or 'EntryNotFound' in s:
                return 'missing404'                            # episode genuinely lacks image_0
            time.sleep(5)
    return 'fail'

ok = fail = miss404 = 0; t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(get, ep): ep for ep in missing}
    for i, f in enumerate(as_completed(futs)):
        r = f.result()
        if r == 'ok': ok += 1
        elif r == 'missing404': miss404 += 1
        else: fail += 1
        if (i + 1) % 500 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(missing)}  ok={ok} 404={miss404} fail={fail}  "
                  f"{rate:.1f}/s  eta={(len(missing)-i-1)/max(rate,0.1)/60:.0f}m", flush=True)
print(f"DONE ok={ok} 404={miss404} fail={fail} [{(time.time()-t0)/60:.1f}m]", flush=True)
