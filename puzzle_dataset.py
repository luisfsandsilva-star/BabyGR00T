import os
import json
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Iterable
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel


def _read_npy_header_from_zip(zip_path: str, member: str) -> Tuple[Tuple[int, ...], np.dtype]:
    """Read the shape and dtype for ``member`` stored inside ``zip_path``."""

    with zipfile.ZipFile(zip_path, "r") as zf:
        if member not in zf.namelist():
            raise FileNotFoundError(f"Missing member '{member}' in '{zip_path}'.")
        with zf.open(member) as member_file:
            version = np.lib.format.read_magic(member_file)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(member_file)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(member_file)
            else:
                raise ValueError(f"Unsupported npy version {version} in '{zip_path}'.")
    return shape, np.dtype(dtype)


@dataclass
class EpisodeMetadata:
    set_name: str
    group_name: str
    puzzle_identifier: int


@dataclass
class Episode:
    npz_path: str
    metadata: EpisodeMetadata
    length: int

    @property
    def num_examples(self) -> int:
        return max(0, self.length - 1)


class EpisodeSetData:
    def __init__(self, info: "EpisodeDatasetInfo", set_name: str, episode_indices: Iterable[int]):
        self.info = info
        self.set_name = set_name
        self._episode_indices: List[int] = []

        group_to_episodes: Dict[str, List[int]] = defaultdict(list)
        for episode_index in episode_indices:
            episode = info.episodes[episode_index]
            group_to_episodes[episode.metadata.group_name].append(episode_index)

        self.group_names = sorted(group_to_episodes)
        self.group_indices = np.zeros(len(self.group_names) + 1, dtype=np.int32)

        for group_idx, group_name in enumerate(self.group_names):
            group_episode_indices = sorted(group_to_episodes[group_name])
            self._episode_indices.extend(group_episode_indices)
            self.group_indices[group_idx + 1] = len(self._episode_indices)

        self.puzzle_identifiers = np.zeros(len(self._episode_indices), dtype=np.int32)
        self.example_prefix = np.zeros(len(self._episode_indices) + 1, dtype=np.int64)
        for idx, episode_index in enumerate(self._episode_indices):
            episode = info.episodes[episode_index]
            self.puzzle_identifiers[idx] = episode.metadata.puzzle_identifier
            self.example_prefix[idx + 1] = self.example_prefix[idx] + episode.num_examples

        self.total_examples = int(self.example_prefix[-1])

    def normalize_indices(self, index) -> Tuple[np.ndarray, bool]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.total_examples)
            indices = np.arange(start, stop, step, dtype=np.int64)
            return indices, False

        if isinstance(index, (list, tuple)):
            arr = np.asarray(index, dtype=np.int64)
            return arr, False

        arr = np.asarray(index)
        if arr.ndim == 0:
            arr = arr.reshape(1)
            return arr.astype(np.int64), True
        return arr.astype(np.int64), False

    def gather(self, indices: np.ndarray, offset: int) -> np.ndarray:
        feature_dim = self.info.feature_dim
        result = np.empty((indices.size, feature_dim), dtype=self.info.dtype)
        for out_idx, value in enumerate(indices):
            episode_pos = int(np.searchsorted(self.example_prefix, value, side="right") - 1)
            if episode_pos < 0:
                raise IndexError("Index out of bounds for episode data")
            episode_index = self._episode_indices[episode_pos]
            local_index = int(value - self.example_prefix[episode_pos])
            latents = self.info.get_episode_latents(episode_index)
            result[out_idx] = latents[local_index + offset]
        return result


class EpisodeLatentField:
    def __init__(self, set_data: EpisodeSetData, offset: int):
        self.set_data = set_data
        self.offset = offset

    def __len__(self) -> int:
        return self.set_data.total_examples

    def __getitem__(self, index):
        indices, is_scalar = self.set_data.normalize_indices(index)
        gathered = self.set_data.gather(indices, self.offset)
        if is_scalar:
            return gathered[0]
        return gathered


class EpisodeMaskField:
    def __init__(self, set_data: EpisodeSetData):
        self.set_data = set_data

    def __len__(self) -> int:
        return self.set_data.total_examples

    def __getitem__(self, index):
        indices, is_scalar = self.set_data.normalize_indices(index)
        mask = np.ones((indices.size, self.set_data.info.feature_dim), dtype=np.bool_)
        if is_scalar:
            return mask[0]
        return mask


class EpisodeDatasetInfo:
    CACHE_SIZE = 8

    def __init__(self, dataset_path: str, split: str):
        self.dataset_path = dataset_path
        self.split = split
        split_latents_dir = os.path.join(dataset_path, split, "latents")
        root_latents_dir = os.path.join(dataset_path, "latents")
        if os.path.isdir(split_latents_dir):
            self.latents_dir = split_latents_dir
        elif os.path.isdir(root_latents_dir):
            self.latents_dir = root_latents_dir
        else:
            raise FileNotFoundError(
                f"Expected directory '{split_latents_dir}' or '{root_latents_dir}'."
            )

        split_metadata_dir = os.path.join(dataset_path, split, "metadata")
        root_metadata_dir = os.path.join(dataset_path, "metadata")
        if os.path.isdir(split_metadata_dir):
            self.metadata_dir = split_metadata_dir
        elif os.path.isdir(root_metadata_dir):
            self.metadata_dir = root_metadata_dir
        else:
            raise FileNotFoundError(
                f"Expected directory '{split_metadata_dir}' or '{root_metadata_dir}'."
            )
        self.episodes: List[Episode] = []
        self._set_to_episode_indices: Dict[str, List[int]] = defaultdict(list)
        self._episode_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self.identifier_offset: int = 0

        npz_files = sorted(
            filename
            for filename in os.listdir(self.latents_dir)
            if filename.startswith("episode_") and filename.endswith(".npz")
        )

        if not npz_files:
            raise FileNotFoundError(f"No episode archives found in '{self.latents_dir}'.")

        feature_dim: Optional[int] = None
        dtype: Optional[np.dtype] = None
        default_identifier = 0

        for filename in npz_files:
            npz_path = os.path.join(self.latents_dir, filename)
            shape, inferred_dtype = _read_npy_header_from_zip(npz_path, "latents.npy")
            if len(shape) < 1:
                raise ValueError(f"Latents stored in '{npz_path}' must be at least 1-D.")

            episode_length = int(shape[0])
            current_feature_dim = int(np.prod(shape[1:])) if len(shape) > 1 else 1

            if feature_dim is None:
                feature_dim = current_feature_dim
            elif feature_dim != current_feature_dim:
                raise ValueError("All episodes must share the same latent dimensionality.")

            if dtype is None:
                dtype = inferred_dtype
            elif dtype != inferred_dtype:
                raise ValueError("All episodes must share the same dtype.")

            metadata_stem = os.path.join(
                self.metadata_dir,
                filename.replace(".npz", ""),
            )
            metadata_json_path = metadata_stem + ".json"
            metadata_jsonl_path = metadata_stem + ".jsonl"

            metadata_dict: Dict[str, Optional[str]] = {}
            if os.path.isfile(metadata_jsonl_path):
                collected: Dict[str, Optional[str]] = {}
                with open(metadata_jsonl_path, "r", encoding="utf-8") as metadata_file:
                    for line in metadata_file:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(record, dict):
                            continue
                        for key in ("set", "group", "puzzle_identifier"):
                            if key not in collected and key in record:
                                collected[key] = record[key]
                        if all(key in collected for key in ("set", "group", "puzzle_identifier")):
                            break
                metadata_dict = collected
                if not metadata_dict and os.path.isfile(metadata_json_path):
                    with open(metadata_json_path, "r", encoding="utf-8") as metadata_file:
                        metadata_dict = json.load(metadata_file)
            elif os.path.isfile(metadata_json_path):
                with open(metadata_json_path, "r", encoding="utf-8") as metadata_file:
                    metadata_dict = json.load(metadata_file)

            set_name = metadata_dict.get("set", "episodes") if metadata_dict else "episodes"
            group_name_raw = metadata_dict.get("group", set_name) if metadata_dict else set_name
            group_name = str(group_name_raw)

            if metadata_dict and "puzzle_identifier" in metadata_dict:
                puzzle_identifier = int(metadata_dict["puzzle_identifier"])
                default_identifier = max(default_identifier, puzzle_identifier + 1)
            else:
                puzzle_identifier = default_identifier
                default_identifier += 1

            episode_metadata = EpisodeMetadata(
                set_name=set_name,
                group_name=group_name,
                puzzle_identifier=puzzle_identifier,
            )

            episode = Episode(
                npz_path=npz_path,
                metadata=episode_metadata,
                length=episode_length,
            )
            self.episodes.append(episode)
            self._set_to_episode_indices[set_name].append(len(self.episodes) - 1)

        self.feature_dim = feature_dim if feature_dim is not None else 1
        self.dtype = dtype if dtype is not None else np.float32
        self.latent_shape = (self.feature_dim,)

        self.sets: Dict[str, EpisodeSetData] = {
            set_name: EpisodeSetData(self, set_name, indices)
            for set_name, indices in self._set_to_episode_indices.items()
        }

    @property
    def total_examples(self) -> int:
        return sum(episode.num_examples for episode in self.episodes)

    @property
    def total_puzzles(self) -> int:
        return len(self.episodes)

    @property
    def total_groups(self) -> int:
        return sum(len(set_data.group_names) for set_data in self.sets.values())

    @property
    def metadata(self) -> PuzzleDatasetMetadata:
        total_examples = self.total_examples
        total_puzzles = self.total_puzzles
        denominator = total_puzzles if total_puzzles else 1
        mean_examples = total_examples / denominator
        unique_identifiers = len({episode.metadata.puzzle_identifier for episode in self.episodes})
        return PuzzleDatasetMetadata(
            pad_id=0,
            ignore_label_id=None,
            blank_identifier_id=0,
            vocab_size=0,
            seq_len=1,
            num_puzzle_identifiers=unique_identifiers,
            total_groups=self.total_groups,
            mean_puzzle_examples=mean_examples,
            total_puzzles=self.total_puzzles,
            sets=sorted(self.sets.keys()),
            task_type="regression",
            input_dim=self.feature_dim,
            target_dim=self.feature_dim,
            input_pad_value=0.0,
            target_pad_value=0.0,
        )

    def set_identifier_offset(self, offset: int) -> None:
        self.identifier_offset = offset

    def get_episode_latents(self, episode_index: int) -> np.ndarray:
        if episode_index in self._episode_cache:
            latents = self._episode_cache.pop(episode_index)
            self._episode_cache[episode_index] = latents
            return latents

        episode = self.episodes[episode_index]
        with np.load(episode.npz_path) as npz_file:
            if "latents" in npz_file.files:
                latents = npz_file["latents"].astype(self.dtype, copy=False)
            elif "arr_0" in npz_file.files:
                latents = npz_file["arr_0"].astype(self.dtype, copy=False)
            else:
                raise KeyError(f"Episode archive '{episode.npz_path}' does not contain latents array.")

        latents = latents.reshape(latents.shape[0], self.feature_dim)

        self._episode_cache[episode_index] = latents
        while len(self._episode_cache) > self.CACHE_SIZE:
            self._episode_cache.popitem(last=False)

        return latents

    def get_set_latent_field(self, set_name: str, offset: int) -> EpisodeLatentField:
        return EpisodeLatentField(self.sets[set_name], offset)

    def get_set_mask_field(self, set_name: str) -> EpisodeMaskField:
        return EpisodeMaskField(self.sets[set_name])

    def get_set_puzzle_identifiers(self, set_name: str) -> np.ndarray:
        base = self.sets[set_name].puzzle_identifiers + self.identifier_offset
        return base.astype(np.int32)

    def get_set_puzzle_indices(self, set_name: str) -> np.ndarray:
        return self.sets[set_name].example_prefix.astype(np.int32)

    def get_set_group_indices(self, set_name: str) -> np.ndarray:
        return self.sets[set_name].group_indices.astype(np.int32)

    @classmethod
    def from_path(cls, dataset_path: str, split: str) -> Optional["EpisodeDatasetInfo"]:
        dataset_json = os.path.join(dataset_path, split, "dataset.json")
        if os.path.isfile(dataset_json):
            return None

        split_latents_dir = os.path.join(dataset_path, split, "latents")
        root_latents_dir = os.path.join(dataset_path, "latents")
        if not (os.path.isdir(split_latents_dir) or os.path.isdir(root_latents_dir)):
            return None

        return cls(dataset_path, split)

def _sample_batch(rng: np.random.Generator, group_order: np.ndarray, puzzle_indices: np.ndarray, group_indices: np.ndarray, start_index: int, global_batch_size: int):
    # Pack examples into a full batch
    batch = []
    batch_puzzle_indices = []
    current_size = 0

    while (start_index < group_order.size) and (current_size < global_batch_size):
        # Pick a group and a puzzle from that group
        group_id = group_order[start_index]
        puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
        start_index += 1

        # Get range of the puzzle
        puzzle_start = puzzle_indices[puzzle_id]
        puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)

        append_size = min(puzzle_size, global_batch_size - current_size)

        # Put into batch
        batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
        batch.append(puzzle_start + np.random.choice(puzzle_size, append_size, replace=False))

        current_size += append_size

    return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)


class PuzzleDatasetConfig(pydantic.BaseModel):
    seed: int
    dataset_paths: List[str]
    global_batch_size: int
    test_set_mode: bool
    epochs_per_iter: int  # Batch X epochs in an iteration to reduce overhead.
    rank: int
    num_replicas: int

class PuzzleDataset(IterableDataset):
    def __init__(self, config: PuzzleDatasetConfig, split: str = "train"):
        super().__init__()
        self.config = config
        self.split = split
        self._episode_datasets: Dict[str, EpisodeDatasetInfo] = {}

        # Merge multiple metadata
        prev_seq_len = None
        prev_vocab_size = None
        prev_pad_id = None
        prev_ignore_label_id = None
        prev_blank_identifier_id = None
        prev_sets = None
        prev_num_identifiers = None
        prev_task_type = None
        prev_input_dim: Optional[int] = None
        prev_target_dim: Optional[int] = None
        prev_input_pad_value: Optional[float] = None
        prev_target_pad_value: Optional[float] = None
        mean_puzzle_examples = 0
        total_puzzles = 0
        total_groups = 0
        num_identifiers = 0
        for dataset_path in config.dataset_paths:
            current_metadata = self._load_metadata(dataset_path)
            episode_info = self._episode_datasets.get(dataset_path)
            if episode_info is not None:
                episode_info.set_identifier_offset(num_identifiers)
            if prev_seq_len is None:
                prev_seq_len = current_metadata.seq_len
                prev_vocab_size = current_metadata.vocab_size
                prev_pad_id = current_metadata.pad_id
                prev_ignore_label_id = current_metadata.ignore_label_id
                prev_blank_identifier_id = current_metadata.blank_identifier_id
                prev_sets = current_metadata.sets
                prev_num_identifiers = current_metadata.num_puzzle_identifiers
                prev_task_type = current_metadata.task_type
                prev_input_dim = current_metadata.input_dim
                prev_target_dim = current_metadata.target_dim
                prev_input_pad_value = current_metadata.input_pad_value
                prev_target_pad_value = current_metadata.target_pad_value
            else:
                assert prev_task_type == current_metadata.task_type
                if prev_seq_len is not None and current_metadata.seq_len is not None:
                    assert prev_seq_len == current_metadata.seq_len
                if prev_vocab_size is not None and current_metadata.vocab_size is not None:
                    assert prev_vocab_size == current_metadata.vocab_size
                if prev_pad_id is not None:
                    assert prev_pad_id == current_metadata.pad_id
                if prev_ignore_label_id is not None or current_metadata.ignore_label_id is not None:
                    assert prev_ignore_label_id == current_metadata.ignore_label_id
                assert prev_blank_identifier_id == current_metadata.blank_identifier_id
                assert prev_sets == current_metadata.sets
                if prev_input_dim is None and current_metadata.input_dim is not None:
                    prev_input_dim = current_metadata.input_dim
                elif current_metadata.input_dim is not None:
                    assert prev_input_dim == current_metadata.input_dim
                if prev_target_dim is None and current_metadata.target_dim is not None:
                    prev_target_dim = current_metadata.target_dim
                elif current_metadata.target_dim is not None:
                    assert prev_target_dim == current_metadata.target_dim
                if prev_input_pad_value is None and current_metadata.input_pad_value is not None:
                    prev_input_pad_value = current_metadata.input_pad_value
                elif current_metadata.input_pad_value is not None:
                    assert prev_input_pad_value == current_metadata.input_pad_value
                if prev_target_pad_value is None and current_metadata.target_pad_value is not None:
                    prev_target_pad_value = current_metadata.target_pad_value
                elif current_metadata.target_pad_value is not None:
                    assert prev_target_pad_value == current_metadata.target_pad_value
                if current_metadata.task_type != "regression":
                    assert prev_num_identifiers == current_metadata.num_puzzle_identifiers
            mean_puzzle_examples += current_metadata.mean_puzzle_examples*current_metadata.total_puzzles
            total_puzzles += current_metadata.total_puzzles
            total_groups += current_metadata.total_groups
            num_identifiers += current_metadata.num_puzzle_identifiers
        mean_puzzle_examples = mean_puzzle_examples / total_puzzles

        self.metadata = PuzzleDatasetMetadata(
            seq_len=prev_seq_len,
            vocab_size=prev_vocab_size,
            pad_id=prev_pad_id,
            ignore_label_id=prev_ignore_label_id,
            blank_identifier_id=prev_blank_identifier_id,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=mean_puzzle_examples,
            total_puzzles=total_puzzles,
            sets=prev_sets,
            task_type=prev_task_type or "classification",
            input_dim=prev_input_dim,
            target_dim=prev_target_dim,
            input_pad_value=prev_input_pad_value,
            target_pad_value=prev_target_pad_value
        )

        # Checks
        assert self.config.global_batch_size % self.config.num_replicas == 0, f"Global batch size {self.config.global_batch_size} must be multiples of nodes {self.config.num_replicas}."
        self.local_batch_size = self.config.global_batch_size // self.config.num_replicas

        # State
        self._data = None
        self._iters = 0

    def _load_metadata(self, dataset_path) -> PuzzleDatasetMetadata:
        episode_info = EpisodeDatasetInfo.from_path(dataset_path, self.split)
        if episode_info is not None:
            self._episode_datasets[dataset_path] = episode_info
            return episode_info.metadata

        with open(os.path.join(dataset_path, self.split, "dataset.json"), "r") as f:
            return PuzzleDatasetMetadata(**json.load(f))

    def _lazy_load_dataset(self):
        if self._data is not None:
            return

        field_mmap_modes = {
            "inputs": "r",

            # Keep indices in memory
            "puzzle_identifiers": None,
            "puzzle_indices": None,
            "group_indices": None,
        }
        if self.metadata.task_type == "regression":
            field_mmap_modes.update({
                "targets": "r",
                "target_mask": "r",
            })
        else:
            field_mmap_modes["labels"] = "r"

        # Load data
        self._data = {}
        for set_name in self.metadata.sets: # Load subset
            for i, dataset_path in enumerate(self.config.dataset_paths):
                if i > 0:
                    set_name_ = set_name + str(i)
                else:
                    set_name_ = set_name

                if dataset_path in self._episode_datasets:
                    episode_info = self._episode_datasets[dataset_path]
                    if set_name not in episode_info.sets:
                        continue
                    set_data = {
                        "inputs": episode_info.get_set_latent_field(set_name, offset=0),
                        "targets": episode_info.get_set_latent_field(set_name, offset=1),
                        "target_mask": episode_info.get_set_mask_field(set_name),
                        "puzzle_identifiers": episode_info.get_set_puzzle_identifiers(set_name),
                        "puzzle_indices": episode_info.get_set_puzzle_indices(set_name),
                        "group_indices": episode_info.get_set_group_indices(set_name),
                    }
                    self._data[set_name_] = set_data
                    continue

                set_data = {}
                for field_name, mmap_mode in field_mmap_modes.items():
                    base_path = os.path.join(
                        dataset_path,
                        self.split,
                        f"{set_name}__{field_name}",
                    )

                    npy_path = base_path + ".npy"
                    if os.path.exists(npy_path):
                        set_data[field_name] = np.load(npy_path, mmap_mode=mmap_mode)
                        continue

                    npz_path = base_path + ".npz"
                    if os.path.exists(npz_path):
                        with np.load(npz_path) as npz_file:
                            if len(npz_file.files) == 1:
                                array = npz_file[npz_file.files[0]]
                            else:
                                key = field_name
                                if key not in npz_file.files:
                                    raise KeyError(
                                        f"NPZ file '{npz_path}' does not contain expected key '{key}'."
                                    )
                                array = npz_file[key]
                        set_data[field_name] = array
                        continue

                    raise FileNotFoundError(
                        f"Missing dataset field '{field_name}' at '{npy_path}' or '{npz_path}'."
                    )

                self._data[set_name_] = set_data


    def _collate_batch(self, batch):
        converted = {}
        for key, value in batch.items():
            if key == "target_mask":
                converted[key] = value.astype(np.bool_)
            elif value.dtype.kind == "f":
                converted[key] = value.astype(np.float32, copy=False)
            else:
                converted[key] = value.astype(np.int32)

        batch = converted

        # Convert ignore label IDs to configured padding value for classification tasks
        if "labels" in batch and self.metadata.ignore_label_id is not None:
            target_pad_value = (
                self.metadata.target_pad_value
                if self.metadata.target_pad_value is not None
                else self.metadata.ignore_label_id
            )
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = target_pad_value

        # Pad to local batch size
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {
                "inputs": (
                    self.metadata.input_pad_value
                    if self.metadata.input_pad_value is not None
                    else self.metadata.pad_id
                ),
                "puzzle_identifiers": self.metadata.blank_identifier_id,
            }

            if "labels" in batch:
                pad_values["labels"] = (
                    self.metadata.target_pad_value
                    if self.metadata.target_pad_value is not None
                    else (
                        self.metadata.ignore_label_id
                        if self.metadata.ignore_label_id is not None
                        else 0
                    )
                )
            if "targets" in batch:
                pad_values["targets"] = float(
                    self.metadata.target_pad_value
                    if self.metadata.target_pad_value is not None
                    else 0.0
                )
            if "target_mask" in batch:
                pad_values["target_mask"] = False

            batch = {
                key: np.pad(
                    value,
                    ((0, pad_size),) + ((0, 0),) * (value.ndim - 1),
                    constant_values=pad_values[key],
                )
                for key, value in batch.items()
            }

        # To tensor
        return {k: torch.from_numpy(v) for k, v in batch.items()}
    
    def _iter_test(self):
        for set_i, (set_name, dataset) in enumerate(self._data.items()):  # type: ignore
            total_examples = len(dataset["inputs"])

            # Load examples one by one
            start_index = 0
            while start_index < total_examples:
                # Compute indices
                end_index = min(total_examples, start_index + self.config.global_batch_size)
                
                local_start = start_index + self.config.rank * self.local_batch_size
                local_end   = min(start_index + (self.config.rank + 1) * self.local_batch_size, end_index)
                
                # Get batch of examples, and also puzzle IDs
                puzzle_indices = []
                puzzle_index = np.searchsorted(dataset["puzzle_indices"], local_start, side="right") - 1
                for i in range(local_start, local_end):
                    while puzzle_index + 1 < len(dataset["puzzle_indices"]) and i >= dataset["puzzle_indices"][puzzle_index + 1]:
                        puzzle_index += 1

                    puzzle_indices.append(puzzle_index)
                
                batch_dict = {
                    "inputs": dataset["inputs"][local_start: local_end],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][puzzle_indices],
                }
                if "labels" in dataset:
                    batch_dict["labels"] = dataset["labels"][local_start: local_end]
                if "targets" in dataset:
                    batch_dict["targets"] = dataset["targets"][local_start: local_end]
                if "target_mask" in dataset:
                    batch_dict["target_mask"] = dataset["target_mask"][local_start: local_end]

                batch = self._collate_batch(batch_dict)

                yield set_name, batch, end_index - start_index
                
                # Advance to next batch
                start_index += self.config.global_batch_size

    def _iter_train(self):
        for set_name, dataset in self._data.items():  # type: ignore
            # Increase epoch count
            self._iters += 1

            # Randomly shuffle groups
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))

            group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1) for _i in range(self.config.epochs_per_iter)])
            start_index = 0
            
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = _sample_batch(
                    rng,
                    group_order=group_order,
                    puzzle_indices=dataset["puzzle_indices"],
                    group_indices=dataset["group_indices"],
                    start_index=start_index,
                    global_batch_size=self.config.global_batch_size,
                )

                # Select current rank and collate
                global_effective_batch_size = batch_puzzle_indices.size  # Global effective batch size, excluding pads

                # Drop last batch
                if global_effective_batch_size < self.config.global_batch_size:
                    break

                batch_indices        = batch_indices       [self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_puzzle_indices = batch_puzzle_indices[self.config.rank * self.local_batch_size: (self.config.rank + 1) * self.local_batch_size]
                batch_dict = {
                    "inputs": dataset["inputs"][batch_indices],
                    "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices],
                }
                if "labels" in dataset:
                    batch_dict["labels"] = dataset["labels"][batch_indices]
                if "targets" in dataset:
                    batch_dict["targets"] = dataset["targets"][batch_indices]
                if "target_mask" in dataset:
                    batch_dict["target_mask"] = dataset["target_mask"][batch_indices]

                batch = self._collate_batch(batch_dict)

                yield set_name, batch, global_effective_batch_size
                
    def __iter__(self):
        worker_info = get_worker_info()
        assert worker_info is None or worker_info.num_workers == 1, "Multithreaded data loading is not currently supported."
        
        self._lazy_load_dataset()
        
        # Iterate using specified mode
        if self.config.test_set_mode:
            yield from self._iter_test()
        else:
            yield from self._iter_train()

