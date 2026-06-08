"""Multi-embodiment OXE streaming dataset for GR00T-style training.

Each dataset on disk follows the LeRobot layout:
  <root>/<name>/meta/info.json
  <root>/<name>/data/chunk-{cc:03d}/episode_{ee:06d}.parquet     # per-frame: action, state, task_index, ...
  <root>/<name>/videos/chunk-{cc:03d}/<camera_key>/episode_{ee:06d}.mp4

Per-chunk samples are (action_chunk, state_at_chunk_start, prev_chunk, last_frame, task_string, embodiment_id).

We pre-build a flat chunk index (RAM-resident, ~few hundred MB max for ~5M chunks)
and decode video frames LAZILY per __getitem__ call. PyTorch DataLoader with
num_workers spawns one decoder per worker process, so video IO doesn't block
the trainer.
"""
import os, json, glob, random, io
from dataclasses import dataclass, field
from typing import Optional
import numpy as np, torch
import pyarrow.parquet as pq
from PIL import Image

# Robot string -> embodiment id (stable across datasets sharing a robot type).
EMBODIMENTS = ['widowx', 'google_robot', 'kuka_iiwa', 'franka', 'ur5',
               'xarm', 'sawyer', 'jaco_2', 'hello_stretch', 'fanuc_mate', 'dlr_edan',
               'agibot']                                  # AgiBot humanoid dual-arm (arms-only 16D jointspace)
EMBODIMENT_ID = {name: i for i, name in enumerate(EMBODIMENTS)}


@dataclass
class DatasetSpec:
    """Parsed metadata + chunk index for one OXE dataset on disk."""
    name: str                                   # e.g. 'bridge_orig_lerobot'
    root: str                                   # local dir holding meta/, data/, videos/
    robot: str                                  # 'widowx', 'franka', ...
    embodiment_id: int                          # index into EMBODIMENTS
    n_episodes: int
    n_frames: int
    fps: int
    chunks_size: int                            # how many eps per chunk-XXX dir
    data_path_template: str                     # e.g. 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
    video_path_template: str                    # similar, with {video_key}
    camera_keys: list                           # list of all available camera key names
    instructions_by_task_idx: dict = field(default_factory=dict)   # task_index -> instruction text
    # Built lazily; one entry per usable chunk across all episodes in this dataset.
    chunk_index: list = field(default_factory=list)
    # episode_id -> (actions, states, task_indices) per-frame tensors, lazily loaded.
    _episode_cache: dict = field(default_factory=dict)


def _load_episodes_meta(root: str):
    """Yield {ep_id, task_idx, length} from the meta/episodes/ jsonl files."""
    for p in sorted(glob.glob(os.path.join(root, 'meta', 'episodes', 'chunk-*', '*.jsonl'))):
        for line in open(p):
            r = json.loads(line)
            yield r
    # Fallback: some layouts have a single meta/episodes.jsonl
    p = os.path.join(root, 'meta', 'episodes.jsonl')
    if os.path.isfile(p):
        for line in open(p):
            yield json.loads(line)


def _load_tasks_meta(root: str):
    """task_index -> task string."""
    tasks = {}
    for p in glob.glob(os.path.join(root, 'meta', 'tasks*.jsonl')):
        for line in open(p):
            r = json.loads(line)
            tasks[r['task_index']] = r.get('task') or r.get('task_str') or ''
    return tasks


def load_dataset_spec(root: str, camera_key: Optional[str] = None,
                      chunk_len: int = 16, lookback: int = 16,
                      max_episodes: Optional[int] = None,
                      chunk_stride: Optional[int] = None) -> DatasetSpec:
    """Open one LeRobot dataset dir, build its flat chunk index."""
    info = json.load(open(os.path.join(root, 'meta', 'info.json')))
    name = os.path.basename(root.rstrip('/'))
    robot = info.get('robot_type', 'unknown')
    eid = EMBODIMENT_ID.get(robot, len(EMBODIMENTS))   # unknown → spare id
    feats = info.get('features', {})
    cams = sorted([k for k in feats if 'image' in k.lower() or 'video' in k.lower()])
    if camera_key is None:
        camera_key = cams[0] if cams else None
    instructions = _load_tasks_meta(root)
    spec = DatasetSpec(
        name=name, root=root, robot=robot, embodiment_id=eid,
        n_episodes=info.get('total_episodes', 0),
        n_frames=info.get('total_frames', 0),
        fps=info.get('fps', 5),
        chunks_size=info.get('chunks_size', 1000),
        data_path_template=info.get('data_path', 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'),
        video_path_template=info.get('video_path', 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'),
        camera_keys=[camera_key] if camera_key else cams,
        instructions_by_task_idx=instructions,
    )

    # Build the chunk index by walking episodes metadata.
    # IMPORTANT: many OXE downloads are partial (chunk-000 only / N first eps), but the
    # meta lists ALL episodes. Filter to those whose video file actually exists on disk
    # — otherwise __getitem__ would FileNotFoundError on the missing mp4s.
    eps_meta = list(_load_episodes_meta(root))
    if max_episodes:
        eps_meta = eps_meta[:max_episodes]
    n_chunks = chunk_len
    n_skipped_missing = 0
    cache_ep_exists = {}
    def _ep_video_exists(ep_id):
        if ep_id in cache_ep_exists: return cache_ep_exists[ep_id]
        cc = ep_id // spec.chunks_size
        vid = os.path.join(root, spec.video_path_template.format(
            episode_chunk=cc, video_key=spec.camera_keys[0], episode_index=ep_id))
        pqp = os.path.join(root, spec.data_path_template.format(
            episode_chunk=cc, episode_index=ep_id))
        ok = os.path.isfile(vid) and os.path.isfile(pqp)
        cache_ep_exists[ep_id] = ok
        return ok
    for r in eps_meta:
        ep_id = r.get('episode_index')
        if ep_id is None: continue
        length = r.get('length') or r.get('num_frames') or 0
        # need (lookback + chunk_len) consecutive frames per sample
        max_start = length - (n_chunks + lookback)
        if max_start <= 0: continue
        if not _ep_video_exists(ep_id):
            n_skipped_missing += 1
            continue
        # chunk_stride: distance between chunk starts. Default = chunk_len (non-overlapping).
        # Smaller stride = more overlap = more samples per episode (same frames, different windows).
        # E.g. stride=4 with chunk_len=16 → 4× more chunks per episode, each shifted by 4 frames.
        _stride = chunk_stride if chunk_stride is not None else n_chunks
        for start in range(lookback, length - n_chunks + 1, _stride):
            spec.chunk_index.append((ep_id, start))
    if n_skipped_missing > 0:
        spec.n_episodes = len(eps_meta) - n_skipped_missing
        # quiet — caller logs the chunk count
    return spec


def _episode_paths(spec: DatasetSpec, ep_id: int) -> tuple:
    cc = ep_id // spec.chunks_size
    pq_path = os.path.join(spec.root, spec.data_path_template.format(episode_chunk=cc, episode_index=ep_id))
    vid_path = os.path.join(spec.root, spec.video_path_template.format(
        episode_chunk=cc, video_key=spec.camera_keys[0], episode_index=ep_id))
    return pq_path, vid_path


def _load_episode_parquet(spec: DatasetSpec, ep_id: int):
    """Return (actions, states, task_idxs) per-frame as torch tensors."""
    if ep_id in spec._episode_cache:
        return spec._episode_cache[ep_id]
    pq_path, _ = _episode_paths(spec, ep_id)
    t = pq.read_table(pq_path, columns=['action', 'observation.state', 'task_index'])
    actions = torch.from_numpy(np.stack(t.column('action').to_pylist())).float()       # (T, A)
    states  = torch.from_numpy(np.stack(t.column('observation.state').to_pylist())).float()  # (T, S)
    task_ix = torch.tensor(t.column('task_index').to_pylist(), dtype=torch.long)        # (T,)
    out = (actions, states, task_ix)
    # Cheap LRU: cap to ~1024 episodes per worker
    if len(spec._episode_cache) >= 1024:
        spec._episode_cache.pop(next(iter(spec._episode_cache)))
    spec._episode_cache[ep_id] = out
    return out


def _decode_frame(video_path: str, frame_idx: int) -> Image.Image:
    """Decode one frame from an mp4. Uses pyav (fast random access) if available, else torchvision.io."""
    try:
        import av
        with av.open(video_path) as ctr:
            stream = ctr.streams.video[0]
            ctr.seek(int(frame_idx * stream.time_base.denominator // max(stream.average_rate, 1)),
                     stream=stream, any_frame=False, backward=True)
            for i, frame in enumerate(ctr.decode(stream)):
                if frame.pts is None: continue
                if i >= 0:
                    return frame.to_image()
    except Exception:
        pass
    import torchvision.io as tvio
    vframes, _, _ = tvio.read_video(video_path, output_format='THWC', pts_unit='sec')
    f = min(frame_idx, vframes.shape[0] - 1)
    return Image.fromarray(vframes[f].numpy())


class MultiOXEDataset(torch.utils.data.Dataset):
    """Multi-dataset OXE chunk sampler.

    __getitem__ returns: (frame_pil, state, action, prev_action, task_str, embodiment_id, dataset_idx)
    """
    def __init__(self, specs: list, chunk_len: int = 16, lookback: int = 16, two_frame: bool = False):
        self.specs = specs
        self.chunk_len = chunk_len
        self.lookback = lookback
        self.two_frame = two_frame          # if True, frame slot returns (prev_frame, cur_frame)
        # Global flat index: (dataset_idx, chunk_local_idx) so we sample uniformly across all chunks.
        self.flat = []
        for di, sp in enumerate(specs):
            self.flat.extend((di, i) for i in range(len(sp.chunk_index)))

    def __len__(self): return len(self.flat)

    def __getitem__(self, i):
        di, ci = self.flat[i]
        spec = self.specs[di]
        ep_id, start = spec.chunk_index[ci]
        actions, states, task_ix = _load_episode_parquet(spec, ep_id)
        ac = actions[start:start + self.chunk_len]                          # (16, A)
        pv = actions[start - self.lookback:start]                           # (16, A)
        st = states[start]                                                  # (S,)
        ti = int(task_ix[start])
        task = spec.instructions_by_task_idx.get(ti, '')
        # Last frame of the predicted chunk == the frame the policy SEES at chunk start;
        # match training convention: use the frame at `start` (current state observation).
        _, vid = _episode_paths(spec, ep_id)
        cur = _decode_frame(vid, start)                                     # PIL.Image
        if self.two_frame:
            prev = _decode_frame(vid, start - 1) if start - 1 >= 0 else cur
            frame = (prev, cur)                                             # (previous, current)
        else:
            frame = cur
        return frame, st, ac, pv, task, spec.embodiment_id, di


def build_balanced_index(specs: list, samples_per_dataset: Optional[int] = None) -> list:
    """Build a sampling index that gives every dataset equal weight per epoch
    (vs the raw size-weighted sampling that would let bridge dominate).
    Returns a list of (dataset_idx, chunk_local_idx) tuples.
    """
    n_per = samples_per_dataset or min(len(sp.chunk_index) for sp in specs)
    idx = []
    rng = random.Random(0)
    for di, sp in enumerate(specs):
        pool = list(range(len(sp.chunk_index)))
        rng.shuffle(pool)
        idx.extend((di, i) for i in pool[:n_per])
    rng.shuffle(idx)
    return idx
