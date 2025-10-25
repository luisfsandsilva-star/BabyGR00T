import os
import json
from typing import Tuple, List, Dict, Optional
import numpy as np
import pydantic

import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataset.common import PuzzleDatasetMetadata

from argdantic import ArgParser
from pydantic import BaseModel

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

