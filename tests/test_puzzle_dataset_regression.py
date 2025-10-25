import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def _write_regression_dataset(base_dir: Path) -> None:
    split_dir = base_dir / "train"
    split_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "pad_id": 0,
        "ignore_label_id": None,
        "blank_identifier_id": 0,
        "vocab_size": 10,
        "seq_len": 2,
        "num_puzzle_identifiers": 1,
        "total_groups": 1,
        "mean_puzzle_examples": 2,
        "total_puzzles": 1,
        "sets": ["regset"],
        "task_type": "regression",
        "input_dim": 2,
        "target_dim": 1,
        "input_pad_value": 0.0,
        "target_pad_value": 0.0,
    }
    with open(split_dir / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    set_name = "regset"

    inputs = np.array([[1, 2], [3, 4]], dtype=np.int32)
    np.save(split_dir / f"{set_name}__inputs.npy", inputs)

    puzzle_identifiers = np.array([7], dtype=np.int32)
    np.save(split_dir / f"{set_name}__puzzle_identifiers.npy", puzzle_identifiers)

    puzzle_indices = np.array([0, 2], dtype=np.int32)
    np.save(split_dir / f"{set_name}__puzzle_indices.npy", puzzle_indices)

    group_indices = np.array([0, 1], dtype=np.int32)
    np.save(split_dir / f"{set_name}__group_indices.npy", group_indices)

    targets = np.array([[1.5], [2.5]], dtype=np.float32)
    np.savez(split_dir / f"{set_name}__targets.npz", targets=targets)

    target_mask = np.array([[True], [False]], dtype=bool)
    np.savez(split_dir / f"{set_name}__target_mask.npz", target_mask=target_mask)


def test_targets_and_mask_loaded_from_npz(tmp_path):
    dataset_dir = tmp_path / "dataset"
    _write_regression_dataset(dataset_dir)

    config = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[str(dataset_dir)],
        global_batch_size=2,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    dataset = PuzzleDataset(config)
    set_name, batch, _ = next(iter(dataset))

    assert set_name == "regset"
    assert batch["targets"].dtype == torch.float32
    assert batch["targets"].tolist() == [[1.5], [2.5]]

    assert batch["target_mask"].dtype == torch.bool
    assert batch["target_mask"].tolist() == [[True], [False]]
