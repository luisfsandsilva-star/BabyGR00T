import os
from typing import List, Optional, Tuple, Dict, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataset.common import PuzzleDatasetMetadata


class LatentNPZDatasetConfig:
    def __init__(
        self,
        *,
        seed: int,
        dataset_paths: List[str],
        global_batch_size: int,
        test_set_mode: bool,
        epochs_per_iter: int,
        rank: int,
        num_replicas: int,
        time_offset: int = 1,
        glob_pattern: str = "**/latents/*.npz",
        max_files: Optional[int] = None,
        vlm_context_dirs: Optional[List[str]] = None,
        context_frame_period: int = 12,
        vlm_verbose: bool = False,
        auto_split: bool = True,
        train_fraction: float = 0.9,
    ) -> None:
        self.seed = seed
        self.dataset_paths = dataset_paths
        self.global_batch_size = global_batch_size
        self.test_set_mode = test_set_mode
        self.epochs_per_iter = epochs_per_iter
        self.rank = rank
        self.num_replicas = num_replicas
        self.time_offset = time_offset
        self.glob_pattern = glob_pattern
        self.max_files = max_files
        self.vlm_context_dirs = vlm_context_dirs
        self.context_frame_period = int(max(1, context_frame_period))
        self.vlm_verbose = vlm_verbose
        self.auto_split = auto_split
        # Clamp train_fraction to a sane range
        self.train_fraction = float(min(max(train_fraction, 0.0), 1.0))


class LatentNPZDataset(IterableDataset):
    """
    Iterable dataset for latent .npz episodes.

    Each .npz is expected to contain 'latents' with shape (T, H, F):
      - T: episode length (index for samples)
      - H: sequence length (becomes L)
      - F: feature dimension (Fin)

    A training sample at index t yields:
      inputs  = latents[t,   :, :] -> [H, F]
      targets = latents[t+k, :, :] -> [H, F] (k = time_offset)
    """

    def __init__(self, config: LatentNPZDatasetConfig, split: str = "train") -> None:
        super().__init__()
        self.config = config
        self.split = split

        assert self.config.global_batch_size % self.config.num_replicas == 0, (
            f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        )
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        self._iters = 0
        self._episode_files: List[str] = []
        self.metadata: Optional[PuzzleDatasetMetadata] = None
        self._vlm_index: Dict[int, List[str]] = {}
        self._printed_vlm_paths: set = set()

    def _discover_files(self) -> None:
        if self._episode_files:
            return
        files: List[str] = []
        for root in self.config.dataset_paths:
            for dirpath, _dirnames, filenames in os.walk(root):
                if "latents" not in dirpath:
                    continue
                for fname in filenames:
                    if fname.endswith(".npz"):
                        files.append(os.path.join(dirpath, fname))
        files.sort()
        if self.config.max_files is not None:
            files = files[: self.config.max_files]

        # Optional automatic train/test split by episodes
        if self.config.auto_split:
            n_files = len(files)
            split_idx = int(n_files * self.config.train_fraction)
            if self.split == "train":
                split_files = files[:split_idx]
            else:  # "test" or any other split uses the remainder
                split_files = files[split_idx:]
            self._episode_files = split_files
            if not self._episode_files:
                # Fallback: if a split would be empty, use all files
                self._episode_files = files
        else:
            self._episode_files = files

        # Build metadata from first file
        if len(self._episode_files):
            arr = np.load(self._episode_files[0], allow_pickle=True)["latents"]
            # arr: (T, H, F)
            _, H, _F = arr.shape
            self.metadata = PuzzleDatasetMetadata(
                pad_id=0,
                ignore_label_id=None,
                blank_identifier_id=0,
                vocab_size=1,
                seq_len=H,
                num_puzzle_identifiers=0,
                total_groups=len(self._episode_files),
                mean_puzzle_examples=1.0,
                total_puzzles=len(self._episode_files),
                sets=["latents"],
            )

        # Optionally discover VLM context files
        if self.config.vlm_context_dirs:
            self._discover_vlm_files()

    def _extract_episode_id(self, path: str) -> Optional[int]:
        base = os.path.basename(path)
        if base.startswith("episode_") and base.endswith(".npz"):
            try:
                return int(base[len("episode_"): -len(".npz")])
            except Exception:
                return None
        return None

    def _discover_vlm_files(self) -> None:
        index: Dict[int, List[str]] = {}
        for root in (self.config.vlm_context_dirs or []):
            for dirpath, _dirnames, filenames in os.walk(root):
                for fname in filenames:
                    if not fname.endswith('.npz'):
                        continue
                    eid = self._extract_episode_id(fname)
                    if eid is None:
                        continue
                    index.setdefault(eid, []).append(os.path.join(dirpath, fname))
        self._vlm_index = index

    def __iter__(self) -> Iterator[Tuple[str, Dict[str, torch.Tensor], int]]:
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not supported."

        self._discover_files()
        rng = np.random.default_rng(self.config.seed + self._iters)
        self._iters += 1

        episode_files = list(self._episode_files)
        if not self.config.test_set_mode:
            rng.shuffle(episode_files)

        # Accumulators for one global batch
        batch_inputs: List[np.ndarray] = []
        batch_targets: List[np.ndarray] = []
        batch_contexts: List[np.ndarray] = []

        # Helper to emit one global batch with per-rank slicing
        def emit_batch():
            nonlocal batch_inputs, batch_targets, batch_contexts
            if not batch_inputs:
                return None
            X = np.stack(batch_inputs, axis=0)  # [B, H, F]
            Y = np.stack(batch_targets, axis=0)
            B = X.shape[0]
            # Pad contexts per batch if present
            ctx_tensor = None
            if len(batch_contexts):
                # Determine max L_ctx and F_ctx in this batch
                max_L = 0
                max_F = 0
                for c in batch_contexts:
                    max_L = max(max_L, c.shape[0])
                    max_F = max(max_F, c.shape[1])
                if max_L > 0 and max_F > 0:
                    padded_ctx = []
                    for c in batch_contexts:
                        pad_L = max_L - c.shape[0]
                        pad_F = max_F - c.shape[1]
                        c_pad = np.pad(c, ((0, pad_L), (0, pad_F)), mode='constant') if (pad_L or pad_F) else c
                        padded_ctx.append(c_pad)
                    C = np.stack(padded_ctx, axis=0)
                    ctx_tensor = torch.from_numpy(C).to(torch.float32)
                else:
                    # All contexts are empty; skip cross_context_raw for this batch
                    ctx_tensor = None

            # Rank slicing
            lstart = self.config.rank * self.local_batch_size
            lend = min(lstart + self.local_batch_size, B)
            if lstart >= lend:
                # This rank has no data for this incomplete batch
                batch_inputs, batch_targets, batch_contexts = [], [], []
                return None
            batch = {
                "inputs": torch.from_numpy(X[lstart:lend]).to(torch.float32),
                "targets": torch.from_numpy(Y[lstart:lend]).to(torch.float32),
            }
            if ctx_tensor is not None:
                batch["cross_context_raw"] = ctx_tensor[lstart:lend]
            gbs_this = B  # global chunk size for proper loss scaling upstream
            batch_inputs, batch_targets, batch_contexts = [], [], []
            return ("latents", batch, gbs_this)

        k = max(1, int(self.config.time_offset))
        for ep in episode_files:
            arr = np.load(ep, allow_pickle=True)["latents"]  # (T,H,F)
            T, H, F = arr.shape
            max_t = T - k
            if max_t <= 0:
                continue

            # Optional VLM contexts for this episode
            ctx_arrays: List[np.ndarray] = []
            if self._vlm_index:
                eid = self._extract_episode_id(ep)
                paths = self._vlm_index.get(eid or -1, [])
                for p in paths:
                    try:
                        with np.load(p, allow_pickle=True) as data:
                            # Prefer 'embeddings' key if present, else first ndarray
                            key = 'embeddings' if 'embeddings' in data.files else None
                            arr_candidate = data[key] if key else None
                            if arr_candidate is None:
                                for kname in data.files:
                                    v = data[kname]
                                    if isinstance(v, np.ndarray):
                                        arr_candidate = v
                                        break
                            if isinstance(arr_candidate, np.ndarray):
                                if self.config.vlm_verbose and p not in self._printed_vlm_paths:
                                    print(f"[VLM] Loaded {p} shape={arr_candidate.shape}, dtype={arr_candidate.dtype}")
                                    if arr_candidate.ndim == 3:
                                        N, V, Fctx = arr_candidate.shape
                                        print(f"[VLM] Interpreting as (N,V,H) -> N={N}, V={V}, H={Fctx}; frame_period={self.config.context_frame_period}")
                                    elif arr_candidate.ndim == 2:
                                        print(f"[VLM] Interpreting as 2D context (either (T,F) or (V,F)) with shape={arr_candidate.shape}")
                                    elif arr_candidate.ndim == 1:
                                        print(f"[VLM] Interpreting as 1D feature vector shape={arr_candidate.shape}")
                                    else:
                                        print(f"[VLM] Warning: Unexpected ndim={arr_candidate.ndim} for {p}")
                                    self._printed_vlm_paths.add(p)
                                ctx_arrays.append(arr_candidate)
                    except Exception:
                        continue

            # iterate all t; for test mode do sequential; for train we allow full sweep
            t_indices = np.arange(max_t)
            if not self.config.test_set_mode:
                rng.shuffle(t_indices)

            for t in t_indices:
                x = arr[t, :, :].astype("float32")  # [H,F]
                y = arr[t + k, :, :].astype("float32")
                batch_inputs.append(x)
                batch_targets.append(y)

                # Build context for time t from any available ctx_arrays
                if ctx_arrays:
                    ctx_t_list: List[np.ndarray] = []
                    max_F_t = 0
                    for ca in ctx_arrays:
                        if ca.ndim != 3:
                            # For this project, we only allow 3D VLM embeddings (N_frames, V_tokens, F_ctx)
                            # Any other shape is ignored to avoid ambiguous alignment.
                            continue
                        # Interpret as (N_frames, V_tokens, F_ctx)
                        N_frames, V_tokens, F_ctx = ca.shape
                        n_idx = min(t // self.config.context_frame_period, N_frames - 1)
                        c = ca[n_idx]  # [V_tokens, F_ctx]
                        c = c.astype('float32', copy=False)
                        max_F_t = max(max_F_t, c.shape[1])
                        ctx_t_list.append(c)
                    if ctx_t_list:
                        # Pad feature dim to max_F_t for concat
                        padded = []
                        for c in ctx_t_list:
                            pad_F = max_F_t - c.shape[1]
                            if pad_F:
                                c = np.pad(c, ((0,0),(0,pad_F)), mode='constant')
                            padded.append(c)
                        ctx_cat = np.concatenate(padded, axis=0)
                    else:
                        ctx_cat = np.zeros((0, 0), dtype='float32')
                    batch_contexts.append(ctx_cat)
                else:
                    # still keep alignment for emit_batch logic
                    batch_contexts.append(np.zeros((0, 0), dtype='float32'))

                if len(batch_inputs) == self.config.global_batch_size:
                    out = emit_batch()
                    if out is not None:
                        yield out

        # Flush remainder
        out = emit_batch()
        if out is not None:
            yield out



