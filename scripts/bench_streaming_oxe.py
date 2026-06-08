#!/usr/bin/env python3
"""Benchmark local OXE pipeline vs HuggingFace streaming.

Measures wall-time per batch for:
  (A) existing MultiOXEDataset + DataLoader (local SSD)
  (B) HF datasets streaming for the parquet metadata (state/action) PLUS
      on-demand hf_hub_download of the corresponding mp4 per frame (no cache reset)
  (C) Same as (B) but USING the hf_hub local cache (warm — second pass)

Uses a small OXE dataset (austin_buds, 2055 chunks) for fast results.
Batch=32 (smaller than training to make first-touch cost visible per batch),
N_BATCHES=20 batches measured after a small warmup.
"""
import os, sys, time, glob, random as _random
THIS = os.path.abspath(os.path.dirname(__file__)); sys.path.insert(0, os.path.dirname(THIS))
import numpy as np, torch

LOCAL_ROOT = 'data/oxe'
HF_REPO = 'IPEC-COMMUNITY/austin_buds_dataset_lerobot'
HF_LOCAL_NAME = 'austin_buds_dataset_lerobot'
BATCH = 32
N_BATCHES = 20
WARMUP_BATCHES = 3


def bench_local():
    """Existing MultiOXEDataset over local SSD."""
    from babygroot_strm.multi_oxe import load_dataset_spec, MultiOXEDataset
    ds_dir = os.path.join(LOCAL_ROOT, HF_LOCAL_NAME)
    if not os.path.isfile(os.path.join(ds_dir, 'meta', 'info.json')):
        print(f"  ✗ local dataset not found at {ds_dir}; skip")
        return None
    sp = load_dataset_spec(ds_dir, chunk_len=16, lookback=16)
    ds = MultiOXEDataset([sp], chunk_len=16, lookback=16)
    print(f"  local dataset: {len(ds)} chunks")
    def collate(batch):
        return batch         # raw return, no aug — we're measuring raw IO
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True,
                                         num_workers=8, collate_fn=collate, drop_last=True,
                                         persistent_workers=True)
    it = iter(loader)
    # warmup
    for _ in range(WARMUP_BATCHES): _ = next(it)
    t0 = time.perf_counter()
    n_chunks = 0
    for _ in range(N_BATCHES):
        batch = next(it)
        n_chunks += len(batch)
    dt = time.perf_counter() - t0
    return dict(label='LOCAL (SSD, num_workers=8)',
                wall=dt, chunks=n_chunks, batches=N_BATCHES,
                chunks_per_sec=n_chunks/dt, sec_per_batch=dt/N_BATCHES)


def bench_streaming_hf():
    """HF datasets streaming for the parquet rows + on-demand mp4 fetch per frame.

    The OXE lerobot format has:
      - data/chunk-XXX/episode_YYY.parquet  (state/action rows; can stream)
      - videos/chunk-XXX/<cam>/episode_YYY.mp4 (video frames; need full file to decode)

    Native HF streaming does NOT handle the mp4 (only the parquet). So pure
    'streaming=True' on the dataset gives metadata only. To actually train you'd
    need to fetch each episode's mp4 — that's what we measure here.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    try:
        ds = load_dataset(HF_REPO, streaming=True, split='train')
    except Exception as e:
        return dict(label='STREAMING HF (parquet only)', error=str(e))

    # iterate the parquet stream + on-demand mp4 download per episode
    it = iter(ds)
    seen_eps = set()
    # warmup
    for _ in range(WARMUP_BATCHES * BATCH):
        try: _ = next(it)
        except StopIteration: it = iter(ds); _ = next(it)

    t0 = time.perf_counter()
    n_chunks = 0
    mp4_fetches = 0
    # build a per-episode mp4 cache map (so we only fetch each mp4 once per run)
    cached_mp4 = {}
    for _ in range(N_BATCHES * BATCH):
        try: row = next(it)
        except StopIteration: it = iter(ds); row = next(it)
        n_chunks += 1
        # row has 'episode_index' field; lookup video path
        ep = row.get('episode_index', None)
        if ep is None: continue
        if ep not in cached_mp4:
            # find mp4 file path inside the repo for this episode
            cc = ep // 1000
            video_key = 'observation.images.image'                                          # common key, may vary
            mp4_path_in_repo = f"videos/chunk-{cc:03d}/{video_key}/episode_{ep:06d}.mp4"
            try:
                local = hf_hub_download(repo_id=HF_REPO, repo_type='dataset',
                                        filename=mp4_path_in_repo)
                cached_mp4[ep] = local
                mp4_fetches += 1
            except Exception:
                cached_mp4[ep] = None
        # we don't decode the frame in this bench — just measure FETCH cost
    dt = time.perf_counter() - t0
    return dict(label='STREAMING HF (parquet + on-demand mp4)',
                wall=dt, chunks=n_chunks, batches=N_BATCHES,
                chunks_per_sec=n_chunks/dt, sec_per_batch=dt/N_BATCHES,
                mp4_fetches=mp4_fetches)


def bench_streaming_warm():
    """Second pass — same code as bench_streaming_hf but the hub-cache is warm
    from the first pass. Measures steady-state cost after first-touch."""
    return bench_streaming_hf()


def report(r):
    if r is None: return
    if 'error' in r: print(f"  {r['label']}: ERROR — {r['error']}"); return
    print(f"  {r['label']}:")
    print(f"    wall:           {r['wall']:.1f}s  ({r['batches']} batches × {BATCH} chunks)")
    print(f"    sec/batch:      {r['sec_per_batch']:.3f}")
    print(f"    chunks/sec:     {r['chunks_per_sec']:.1f}")
    if 'mp4_fetches' in r:
        print(f"    fresh mp4 hits: {r['mp4_fetches']} (rest cached)")


if __name__ == '__main__':
    print("== A. LOCAL ==")
    a = bench_local(); report(a)
    print()
    print("== B. STREAMING (cold cache) ==")
    b = bench_streaming_hf(); report(b)
    print()
    print("== C. STREAMING (warm cache, second pass) ==")
    c = bench_streaming_warm(); report(c)
    print()
    print("== ratios ==")
    if a and b and 'sec_per_batch' in b:
        print(f"  streaming cold / local:  {b['sec_per_batch']/a['sec_per_batch']:.1f}× slower")
    if a and c and 'sec_per_batch' in c:
        print(f"  streaming warm / local:  {c['sec_per_batch']/a['sec_per_batch']:.1f}× slower")
