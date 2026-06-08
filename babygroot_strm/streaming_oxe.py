"""Streaming OXE dataset — lazy-fetches episode files via HuggingFace Hub.

Why this works without rewriting everything:
  `hf_hub_download(repo_id, filename)` returns a local path to the file. On
  first access it downloads to ~/.cache/huggingface/hub/...; on subsequent
  accesses the local cached file is reused. So "streaming" here means lazy
  download with persistent local cache — close to true streaming for any
  single-pass workload; near-free for multi-epoch.

  We pre-fetch ONLY meta files for each repo to build the chunk index (small
  KB-sized files). Episode parquet + mp4 are fetched on demand in __getitem__.
"""
import os, json, threading
from typing import List, Optional
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from .multi_oxe import (DatasetSpec, MultiOXEDataset, _load_episodes_meta,
                         _load_tasks_meta, _load_episode_parquet, _decode_frame,
                         EMBODIMENTS, EMBODIMENT_ID)


# Full 36 OXE LeRobot ports under the IPEC-COMMUNITY namespace.
OXE_REPOS = [
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

_dl_lock = threading.Lock()


def _hf_download(repo_id: str, filename: str, repo_type: str = 'dataset',
                 max_retries: int = 3) -> Optional[str]:
    """hf_hub_download with retries; returns local cached path or None on hard failure."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
        except EntryNotFoundError:
            return None              # file genuinely doesn't exist in repo
        except Exception as e:
            last_err = e
    if last_err: raise last_err
    return None


def load_streaming_dataset_spec(repo_id: str,
                                chunk_len: int = 16,
                                lookback: int = 16,
                                max_episodes: Optional[int] = None,
                                chunk_stride: Optional[int] = None,
                                verbose: bool = True) -> Optional[DatasetSpec]:
    """Like multi_oxe.load_dataset_spec but for an HF repo. Pre-fetches only
    meta/{info.json, episodes.jsonl, tasks.jsonl}. Episode files are NOT
    fetched here — they're lazy-downloaded inside __getitem__.

    Returns None if the repo's meta is unavailable (skip this repo).
    """
    # 1. fetch meta files (small, KB)
    info_p = _hf_download(repo_id, 'meta/info.json')
    if info_p is None:
        if verbose: print(f"  [streaming] skip {repo_id} — no meta/info.json")
        return None
    snapshot_dir = os.path.dirname(os.path.dirname(info_p))
    _hf_download(repo_id, 'meta/episodes.jsonl')                            # required
    _hf_download(repo_id, 'meta/tasks.jsonl')                               # may be absent in some repos

    info = json.load(open(info_p))
    feats = info.get('features', {})
    cams = sorted([k for k in feats if 'image' in k.lower() or 'video' in k.lower()])

    name = repo_id.split('/')[-1]
    robot = info.get('robot_type', 'unknown')
    eid = EMBODIMENT_ID.get(robot, len(EMBODIMENTS))
    instructions = _load_tasks_meta(snapshot_dir)

    spec = DatasetSpec(
        name=name, root=snapshot_dir, robot=robot, embodiment_id=eid,
        n_episodes=info.get('total_episodes', 0),
        n_frames=info.get('total_frames', 0),
        fps=info.get('fps', 5),
        chunks_size=info.get('chunks_size', 1000),
        data_path_template=info.get('data_path',
                                     'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'),
        video_path_template=info.get('video_path',
                                      'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'),
        camera_keys=[cams[0]] if cams else [],
        instructions_by_task_idx=instructions,
    )
    # stamp the spec with its HF repo so __getitem__ can lazy-fetch
    object.__setattr__(spec, '_hf_repo_id', repo_id)

    # 2. build chunk index WITHOUT filtering by file existence (lazy fetch handles it)
    eps_meta = list(_load_episodes_meta(snapshot_dir))
    if max_episodes:
        eps_meta = eps_meta[:max_episodes]
    n_chunks = chunk_len
    _stride = chunk_stride if chunk_stride is not None else n_chunks
    for r in eps_meta:
        ep_id = r.get('episode_index')
        if ep_id is None: continue
        length = r.get('length') or r.get('num_frames') or 0
        if length - (n_chunks + lookback) <= 0: continue
        for start in range(lookback, length - n_chunks + 1, _stride):
            spec.chunk_index.append((ep_id, start))

    if verbose:
        print(f"  [streaming] {name:<55s} robot={robot:<14s} chunks={len(spec.chunk_index)}")
    return spec


class StreamingMultiOXEDataset(MultiOXEDataset):
    """MultiOXEDataset variant that lazy-downloads each chunk's parquet + mp4
    on first access via hf_hub_download. After the first access, the file is
    cached locally (~/.cache/huggingface/) and subsequent reads are local-fast.

    Streaming overhead applies once per (episode, file_kind); typical worker
    sees ~ a few hundred unique episodes before the cache is fully warm.
    """
    def __getitem__(self, i):
        di, ci = self.flat[i]
        spec = self.specs[di]
        ep_id, start = spec.chunk_index[ci]
        repo = getattr(spec, '_hf_repo_id', None)
        # lazy-fetch parquet + mp4 if this spec is a streaming spec
        if repo is not None:
            cc = ep_id // spec.chunks_size
            pq_rel = spec.data_path_template.format(episode_chunk=cc, episode_index=ep_id)
            vid_rel = spec.video_path_template.format(
                episode_chunk=cc, video_key=spec.camera_keys[0], episode_index=ep_id)
            # serialize per-process downloads to avoid duplicate concurrent fetches
            with _dl_lock:
                pq_local = _hf_download(repo, pq_rel)
                vid_local = _hf_download(repo, vid_rel)
            if pq_local is None or vid_local is None:
                # missing file — fall back to sampling next index
                return self.__getitem__((i + 1) % len(self.flat))
        # delegate the rest to the parent's __getitem__ (now reads from cache)
        return super().__getitem__(i)
