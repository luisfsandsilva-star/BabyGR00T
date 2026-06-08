"""Data loading.

Two stages:
  1. SO101Streamer / load_so101_episodes  — stream LeRobot-format SO-101
     episodes from local disk or HuggingFace, returning (action_chunks,
     chunk_states, per_chunk_frames, prompt) tuples.
  2. ChunkDataset / make_loader            — random-access PyTorch Dataset
     over the per-episode vision cache (int8-quantized InternVL3 hidden states)
     plus the loaded action/state chunks.

Augmentation (online, in the dataset):
  - Standard dropout on cached visual features
  - Ornstein-Uhlenbeck noise on action joint trajectories (temporally
    correlated, σ ~1% of joint range)
"""
from __future__ import annotations

import os
import json
import glob
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image


# ── Defaults (override via load_so101_episodes args) ──
ACTION_DIM = 6
STATE_DIM  = 6
CHUNK_LEN  = 16
NUM_FRAMES = 4
IMG_SIZE   = 128

DEFAULT_DATA_DIR       = os.path.expanduser("~/Desktop/BabyGR00TNew/data/so101-pick-and-place")
DEFAULT_SUPPLEMENT_DIR = os.path.expanduser("~/Desktop/BabyGR00TNew/data/so101-supplement")

TASK_PROMPTS = [
    "Pick and Place SO101 Arm.",
    "Pick up the object and place it at the target location.",
    "Grasp the item and move it to the designated spot.",
    "Use the robot arm to pick and place the object.",
    "Reach for the object, grasp it, and set it down at the goal position.",
    "Grab the object and put it in the target area.",
    "Pick up the item, then set it down at the goal.",
    "Lift the object and transfer it to the placement zone.",
    "The robot arm picks up an object and places it at the target.",
    "Pick and place the object.",
]

SUPPLEMENT_PROMPTS = [
    "Grasp the red cube and place it in the gray bowl.",
    "Pick up the red block and put it into the gray bowl.",
    "Grab the red cube and drop it in the bowl.",
    "Take the red cube and move it to the gray bowl.",
    "The robot picks up the red cube and places it in the bowl.",
]


# ════════════════════════════════════════════════════════════
#  Episode streaming
# ════════════════════════════════════════════════════════════

class SO101Streamer:
    """LeRobot-style SO-101 episode streamer.
    Yields (action_chunks (n_ch,T,A), chunk_states (n_ch,A), per_chunk_frames, prompt).
    `load_video=False` returns placeholder frames (used for VAE training).
    """
    def __init__(self, dataset_id="pavelsimo/SO-101-pick-and-place",
                 n_episodes=200, chunk_len=CHUNK_LEN,
                 local_data_dir=DEFAULT_DATA_DIR, load_video=False,
                 camera_key="observation.images.front",
                 episode_indices=None):
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        self.chunk_len = chunk_len
        self.dataset_id = dataset_id
        self.local_data_dir = local_data_dir
        self.load_video = load_video
        self.camera_key = camera_key
        if os.path.isdir(local_data_dir):
            print(f"Loading from local: {local_data_dir} ...")
            # parquet-only load — avoids auto-detecting the Video feature (which would
            # require torchcodec just to iterate); we decode videos ourselves via PyAV.
            import glob as _glob
            data_files = _glob.glob(os.path.join(local_data_dir, 'data/**/*.parquet'), recursive=True)
            self.ds = load_dataset('parquet', data_files=sorted(data_files), split='train')
            info_path = os.path.join(local_data_dir, 'meta/info.json')
            tasks_path = os.path.join(local_data_dir, 'meta/tasks.jsonl')
            tasks_path = tasks_path if os.path.exists(tasks_path) else None
        else:
            print(f"Loading {dataset_id} (streaming) ...")
            self.ds = load_dataset(dataset_id, split='train', streaming=True)
            info_path = hf_hub_download(dataset_id, 'meta/info.json', repo_type='dataset')
            try:
                tasks_path = hf_hub_download(dataset_id, 'meta/tasks.jsonl', repo_type='dataset')
            except Exception:
                tasks_path = None
        with open(info_path) as f:
            self.info = json.load(f)
        self.video_path_template = self.info['video_path']
        self.chunks_size = self.info['chunks_size']
        self.n_episodes = n_episodes
        self.episode_indices = set(episode_indices) if episode_indices is not None else None
        # Task lookup (LeRobot v2.0+ datasets ship meta/tasks.jsonl mapping
        # task_index -> human-readable task). BridgeData V2 has 20k unique
        # tasks; without this map we'd lose the language signal entirely.
        self.tasks = {}
        if tasks_path:
            try:
                with open(tasks_path) as f:
                    for line in f:
                        t = json.loads(line)
                        self.tasks[t['task_index']] = t['task']
                print(f"  {len(self.tasks)} unique task descriptions")
            except Exception:
                pass
        print(f"  {self.info['total_episodes']} episodes, "
              f"{self.info['total_frames']} frames, {self.info['fps']} FPS")

    def _download_video(self, episode_idx, camera=None):
        camera = camera or self.camera_key
        chunk_idx = episode_idx // self.chunks_size
        # LeRobot v2.0 templates use {episode_chunk}/{episode_index}
        # (e.g. BridgeData V2 IPEC-COMMUNITY/bridge_orig_lerobot); some
        # v2.x variants use {chunk_index}/{file_index}. Passing both pairs
        # covers both layouts since str.format ignores unused keys.
        rel = self.video_path_template.format(
            video_key=camera,
            chunk_index=chunk_idx, episode_chunk=chunk_idx,
            file_index=episode_idx, episode_index=episode_idx,
        )
        local = os.path.join(self.local_data_dir, rel)
        if os.path.exists(local):
            return local
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.dataset_id, rel, repo_type='dataset')

    def _decode_per_chunk_frames(self, video_path, n_chunks, chunk_len, n_frames):
        """For chunk i, return n_frames consecutive frames ending at timestep
        i * chunk_len (stride 1, last frame = state-action at chunk start).
        Early chunks padded with the earliest available frame.
        """
        import av
        placeholder = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        all_targets = set(); per_chunk_idxs = []
        for chunk_i in range(n_chunks):
            current = chunk_i * chunk_len
            idxs = [max(0, current - (n_frames - 1 - k)) for k in range(n_frames)]
            per_chunk_idxs.append(idxs); all_targets.update(idxs)
        if not all_targets:
            return [[placeholder] * n_frames for _ in range(n_chunks)]
        max_t = max(all_targets); cache = {}
        try:
            container = av.open(video_path)
            for i, frame in enumerate(container.decode(video=0)):
                if i in all_targets:
                    cache[i] = frame.to_image().resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                if i >= max_t:
                    break
            container.close()
        except Exception as e:
            print(f"    [warn] decode failed ({video_path}): {e}")
            return [[placeholder] * n_frames for _ in range(n_chunks)]
        return [[cache.get(i, placeholder) for i in idxs] for idxs in per_chunk_idxs]

    def stream_episodes(self):
        it = iter(self.ds)
        current_ep = -1; ep_actions, ep_states = [], []
        ep_task_idx = 0
        yielded = 0
        for sample in it:
            ep = sample['episode_index']
            if self.episode_indices is not None and ep not in self.episode_indices:
                continue                                          # skip unwanted episodes
            if ep != current_ep:
                if current_ep >= 0 and len(ep_actions) >= self.chunk_len:
                    yield self._package(current_ep, ep_actions, ep_states, ep_task_idx)
                    yielded += 1
                    if yielded >= self.n_episodes:
                        return
                current_ep = ep; ep_actions, ep_states = [], []
                ep_task_idx = sample.get('task_index', 0)
            ep_actions.append(sample['action'])
            ep_states.append(sample['observation.state'])
        if len(ep_actions) >= self.chunk_len and yielded < self.n_episodes:
            yield self._package(current_ep, ep_actions, ep_states, ep_task_idx)

    def _package(self, ep_idx, actions, states, task_idx=0):
        actions = torch.tensor(actions, dtype=torch.float32)
        states  = torch.tensor(states,  dtype=torch.float32)
        # Respect the actual action dim from the dataset (Bridge=7, SO-101=6),
        # don't reshape against a hardcoded constant.
        A = actions.shape[-1]
        n_ch = max(1, len(actions) // self.chunk_len)
        action_chunks = actions[:n_ch * self.chunk_len].view(n_ch, self.chunk_len, A)
        chunk_states  = states[::self.chunk_len][:n_ch]
        per_chunk_frames = None
        if self.load_video:
            try:
                vp = self._download_video(ep_idx)
                per_chunk_frames = self._decode_per_chunk_frames(
                    vp, n_ch, self.chunk_len, NUM_FRAMES)
            except Exception:
                pass
        if per_chunk_frames is None:
            empty = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
            per_chunk_frames = [[empty] * NUM_FRAMES for _ in range(n_ch)]
        task_str = self.tasks.get(task_idx, "manipulate the object")
        return action_chunks, chunk_states, per_chunk_frames, task_str


def _has_real_video(ep_idx, source, data_dir, supplement_dir):
    if source == 'pavelsimo':
        path = os.path.join(data_dir,
            f"videos/observation.images.front/chunk-000/file-{ep_idx:03d}.mp4")
    elif source == 'supplement_above':
        path = os.path.join(supplement_dir,
            f"view_above/videos/chunk-000/observation.images.front/episode_{ep_idx:06d}.mp4")
    elif source == 'supplement_side':
        path = os.path.join(supplement_dir,
            f"view_side/videos/chunk-000/observation.images.front/episode_{ep_idx:06d}.mp4")
    else:
        return False
    return os.path.exists(path)


def load_so101_episodes(load_video=False, only_real_video=True,
                        data_dir=DEFAULT_DATA_DIR,
                        supplement_dir=DEFAULT_SUPPLEMENT_DIR):
    """Load the merged SO-101 dataset (pavelsimo + view_above + view_side
    supplement). If only_real_video, episodes without on-disk video are
    skipped — those are typically filler entries.
    Returns: list[ (action_chunks, chunk_states, per_chunk_frames, prompt) ].
    """
    eps = []
    print("  Loading pavelsimo dataset ...")
    s1 = SO101Streamer(dataset_id="pavelsimo/SO-101-pick-and-place",
                       n_episodes=101, local_data_dir=data_dir,
                       load_video=load_video)
    skipped = 0
    for i, ep in enumerate(s1.stream_episodes()):
        if only_real_video and not _has_real_video(i, 'pavelsimo', data_dir, supplement_dir):
            skipped += 1; continue
        eps.append(ep)
    if skipped:
        print(f"    Skipped {skipped} episodes without video")

    # Optional supplement (xinjiehu76 / view_above + view_side)
    for view_name, fallback_prompt in [
        ("view_above", "Grasp the red cube and place it in the gray bowl."),
        ("view_side",  "Grasp the red cube and place it in the gray bowl."),
    ]:
        view_dir = os.path.join(supplement_dir, view_name)
        if not os.path.isdir(view_dir):
            continue
        before = len(eps)
        from datasets import load_dataset
        parquets = sorted(glob.glob(os.path.join(view_dir, 'data/**/*.parquet'),
                                    recursive=True))
        if not parquets:
            continue
        ds = load_dataset('parquet', data_files=parquets, split='train')
        cur = -1; ep_a, ep_s = [], []
        for sample in ds:
            ep = sample['episode_index']
            if ep != cur:
                if cur >= 0 and len(ep_a) >= CHUNK_LEN:
                    eps.append(_package_supplement(cur, ep_a, ep_s, view_dir,
                                                   load_video, fallback_prompt))
                cur = ep; ep_a, ep_s = [], []
            ep_a.append(sample['action'])
            ep_s.append(sample['observation.state'])
        if len(ep_a) >= CHUNK_LEN:
            eps.append(_package_supplement(cur, ep_a, ep_s, view_dir,
                                           load_video, fallback_prompt))
        print(f"    {len(eps) - before} episodes from {view_name}")
    print(f"  TOTAL: {len(eps)} episodes, "
          f"{sum(e[0].shape[0] for e in eps)} chunks")
    return eps


def load_lerobot_episodes(dataset_id="IPEC-COMMUNITY/bridge_orig_lerobot",
                          load_video=False,
                          camera_key="observation.images.image_0",
                          local_data_dir=None,
                          n_episodes=None,
                          prompt=None,
                          episode_indices=None):
    """Load a single LeRobot dataset (e.g. an OXE subset).

    Returns the same tuple format as `load_so101_episodes`:
        list[ (action_chunks, chunk_states, per_chunk_frames, prompt) ].

    Default is **BridgeData V2** (`IPEC-COMMUNITY/bridge_orig_lerobot`) — the
    chosen OXE migration target for v5. WidowX 6-DoF + 1 gripper (action_dim=7,
    state_dim=8), 53k episodes / 1.9M frames at 5 fps, four cameras
    (`observation.images.image_0/1/2/3`). Pass any LeRobot-format
    `dataset_id` to swap (e.g. `IPEC-COMMUNITY/fractal20220817_data_lerobot`
    for RT-1 Google Robot).

    Args:
        dataset_id     : HuggingFace dataset id (LeRobot v2.0 format).
        load_video     : if True, decode per-chunk frames; else placeholders.
        camera_key     : video key inside the dataset
                         (BridgeV2: `observation.images.image_0` is primary).
        local_data_dir : optional local cache path. If unset, defaults under
                         `~/Desktop/BabyGR00TNew/data/<repo-name>`.
        n_episodes     : cap (default = all episodes from `meta/info.json`).
        prompt         : task prompt. If None, derives one from each episode's
                         own `task` field (LeRobot v2.0+ datasets carry these).
    """
    if local_data_dir is None:
        local_data_dir = os.path.expanduser(
            f"~/Desktop/BabyGR00TNew/data/{dataset_id.replace('/', '__')}")
    if prompt is None:
        prompt = ("Pick up the object and place it at the target location."
                  if 'so101' in dataset_id.lower()
                  else "Perform the demonstrated manipulation task.")

    streamer = SO101Streamer(
        dataset_id=dataset_id,
        n_episodes=n_episodes if n_episodes is not None else 10**9,
        local_data_dir=local_data_dir,
        load_video=load_video,
        camera_key=camera_key,
        episode_indices=episode_indices,
    )
    eps = []
    import time as _t
    _t0 = _t.perf_counter()
    target = streamer.n_episodes if streamer.n_episodes < 10**8 else None
    for ep in streamer.stream_episodes():
        ac, sc, frames, ep_task_str = ep
        # OXE LeRobot v2.0 datasets carry per-episode task descriptions in
        # meta/tasks.jsonl (e.g. Bridge V2 has ~20k unique strings). Use them
        # directly; fall back to the supplied default prompt only if missing.
        ep_prompt = ep_task_str.strip() if (ep_task_str and ep_task_str.strip()) else prompt
        eps.append((ac, sc, frames, ep_prompt))
        n = len(eps)
        if n % 25 == 0:
            el = _t.perf_counter() - _t0
            rate = n / max(el, 1e-6)
            eta = (target - n) / rate if (target and rate > 0) else 0
            print(f"  preload {n}{'/'+str(target) if target else ''} eps  "
                  f"[{el:.0f}s, {rate:.1f} ep/s, ETA {eta:.0f}s]",
                  flush=True)
    print(f"  TOTAL: {len(eps)} episodes, "
          f"{sum(e[0].shape[0] for e in eps)} chunks  ({dataset_id})")
    return eps


def _package_supplement(ep_idx, actions, states, view_dir, load_video, fallback_prompt):
    a = torch.tensor(actions, dtype=torch.float32)
    s = torch.tensor(states,  dtype=torch.float32)
    n_ch = len(a) // CHUNK_LEN
    ac = a[:n_ch * CHUNK_LEN].view(n_ch, CHUNK_LEN, ACTION_DIM)
    sc = s[::CHUNK_LEN][:n_ch]
    per_chunk_frames = None
    if load_video:
        try:
            vp = os.path.join(view_dir,
                f"videos/chunk-000/observation.images.front/episode_{ep_idx:06d}.mp4")
            if os.path.exists(vp):
                streamer = SO101Streamer(load_video=False)
                per_chunk_frames = streamer._decode_per_chunk_frames(
                    vp, n_ch, CHUNK_LEN, NUM_FRAMES)
        except Exception:
            pass
    if per_chunk_frames is None:
        empty = Image.new('RGB', (IMG_SIZE, IMG_SIZE))
        per_chunk_frames = [[empty] * NUM_FRAMES for _ in range(n_ch)]
    prompt = (random.choice(SUPPLEMENT_PROMPTS) if 'red cube' in fallback_prompt.lower()
              else fallback_prompt)
    return ac, sc, per_chunk_frames, prompt


# ════════════════════════════════════════════════════════════
#  Per-chunk dataset over the int8 vision cache
# ════════════════════════════════════════════════════════════

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated action noise."""
    def __init__(self, dim, theta=0.15, sigma=0.02):
        self.dim, self.theta, self.sigma = dim, theta, sigma

    def sample(self, length):
        noise = torch.zeros(length, self.dim)
        x = torch.zeros(self.dim)
        for t in range(length):
            x = x + self.theta * (-x) + self.sigma * torch.randn(self.dim)
            noise[t] = x
        return noise


class ChunkDataset(Dataset):
    """One sample = one (episode, chunk).

    Returns:
      hidden       : (25, N_tok, 896) float — InternVL3 hidden states
      action       : (CHUNK_LEN, action_dim)
      state        : (state_dim,)
      next_action  : (CHUNK_LEN, action_dim)  — successor chunk if present
      next_state   : (state_dim,)
      has_next     : bool
      ep_i, ch_i   : ints
    """
    def __init__(self, cache_dir, episodes, lru_size=2,
                 augment=False,
                 dropout=0.0,
                 action_noise_sigma=0.02,
                 action_noise_theta=0.15):
        self.cache_dir = cache_dir
        self.episodes = episodes
        self.lru_size = lru_size
        self.augment = augment
        self.dropout = dropout
        self.ou = (OUNoise(dim=episodes[0][0].shape[-1],
                           theta=action_noise_theta,
                           sigma=action_noise_sigma)
                   if augment else None)
        # If the cache was written with --n-vis-aug>0, meta.json carries
        # `n_vis_aug` and each ep_NNN.pt holds n_chunks*(1+n_vis_aug) entries.
        # Old caches (no meta.json) simply have n_chunks entries — n_vis_aug=0.
        self.n_vis_aug = 0
        meta_path = os.path.join(cache_dir, 'meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                self.n_vis_aug = int(meta.get('n_vis_aug', 0))
            except Exception:
                pass
        self.index = []
        for ep_i, ep in enumerate(episodes):
            for ch_i in range(ep[0].shape[0]):
                self.index.append((ep_i, ch_i))
        self._cache = None  # per-worker LRU initialised lazily

    def __len__(self):
        return len(self.index)

    def _get_ep_data(self, ep_i):
        if self._cache is None:
            from collections import OrderedDict
            self._cache = OrderedDict()
        if ep_i in self._cache:
            self._cache.move_to_end(ep_i)
            return self._cache[ep_i]
        path = os.path.join(self.cache_dir, f'ep_{ep_i:03d}.pt')
        data = torch.load(path, map_location='cpu', weights_only=True)
        self._cache[ep_i] = data
        while len(self._cache) > self.lru_size:
            self._cache.popitem(last=False)
        return data

    def __getitem__(self, idx):
        ep_i, ch_i = self.index[idx]
        ep_data = self._get_ep_data(ep_i)

        # ep_data layout when n_vis_aug > 0:
        #   [chunk0_v0, chunk1_v0, ..., chunkN_v0,
        #    chunk0_v1, ..., chunkN_v1,
        #    ..., chunkN_v(n_vis_aug)]
        # so the offset for variant v on chunk ch_i is v*n_chunks + ch_i.
        # When augment=True we sample v uniformly; otherwise v=0 (original).
        ep = self.episodes[ep_i]
        n_chunks = ep[0].shape[0]
        if self.n_vis_aug > 0 and self.augment:
            v = random.randint(0, self.n_vis_aug)        # 0..n_vis_aug inclusive
        else:
            v = 0
        entry_idx = v * n_chunks + ch_i
        # Old cache (no augs, no meta) just stores n_chunks entries.
        if entry_idx >= len(ep_data):
            entry_idx = ch_i
        q, scales = ep_data[entry_idx]
        hidden = q.float() * scales.float().unsqueeze(0).unsqueeze(0)

        ep = self.episodes[ep_i]
        action = ep[0][ch_i]
        state  = ep[1][ch_i]
        n_ch   = ep[0].shape[0]
        has_next   = ch_i + 1 < n_ch
        next_action = ep[0][min(ch_i + 1, n_ch - 1)]
        next_state  = ep[1][min(ch_i + 1, n_ch - 1)]

        if self.augment:
            if self.dropout > 0:
                mask = torch.rand_like(hidden) > self.dropout
                hidden = hidden * mask / max(1.0 - self.dropout, 0.1)
            if self.ou is not None:
                action = action + self.ou.sample(action.shape[0])

        return {
            'hidden': hidden, 'action': action, 'state': state,
            'next_action': next_action, 'next_state': next_state,
            'has_next': has_next, 'ep_i': ep_i, 'ch_i': ch_i,
        }


def chunk_collate(batch):
    """Pads variable N_tok across the batch."""
    B = len(batch)
    n_tok_max = max(s['hidden'].shape[1] for s in batch)
    n_layers, D = batch[0]['hidden'].shape[0], batch[0]['hidden'].shape[2]
    hidden = torch.zeros(B, n_layers, n_tok_max, D)
    for b, s in enumerate(batch):
        h = s['hidden']
        hidden[b, :, :h.shape[1], :] = h
    return {
        'hidden': hidden,
        'action': torch.stack([s['action'] for s in batch]),
        'state':  torch.stack([s['state']  for s in batch]),
        'next_action': torch.stack([s['next_action'] for s in batch]),
        'next_state':  torch.stack([s['next_state']  for s in batch]),
        'has_next': torch.tensor([s['has_next'] for s in batch]),
        'ep_i': torch.tensor([s['ep_i'] for s in batch]),
        'ch_i': torch.tensor([s['ch_i'] for s in batch]),
    }


def make_loader(cache_dir, episodes, batch_size=2, num_workers=2, shuffle=True,
                lru_size=2, augment=False, dropout=0.0):
    ds = ChunkDataset(cache_dir, episodes, lru_size=lru_size,
                      augment=augment, dropout=dropout)
    return DataLoader(
        ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=shuffle, pin_memory=True, drop_last=True,
        collate_fn=chunk_collate, persistent_workers=(num_workers > 0),
    )


# ════════════════════════════════════════════════════════════
#  Vision cache loader (lazy LRU)
# ════════════════════════════════════════════════════════════

class _LazyVisionCache:
    """Episode-level lazy LRU over the int8-quantized vision cache."""
    def __init__(self, cache_dir, meta, max_episodes_in_ram=4):
        from collections import OrderedDict
        self.cache_dir = cache_dir
        self.meta = meta
        self.max_in_ram = max_episodes_in_ram
        self._lru = OrderedDict()

    def get_episode(self, ep_i):
        if ep_i in self._lru:
            self._lru.move_to_end(ep_i)
            return self._lru[ep_i]
        path = os.path.join(self.cache_dir, f'ep_{ep_i:03d}.pt')
        data = torch.load(path, map_location='cpu', weights_only=True)
        self._lru[ep_i] = data
        while len(self._lru) > self.max_in_ram:
            self._lru.popitem(last=False)
        return data

    def __getitem__(self, ep_i):
        return self.get_episode(ep_i)


def load_vision_cache(cache_dir='vision_cache', max_episodes_in_ram=4):
    with open(os.path.join(cache_dir, 'meta.json')) as f:
        meta = json.load(f)
    return _LazyVisionCache(cache_dir, meta, max_episodes_in_ram)
