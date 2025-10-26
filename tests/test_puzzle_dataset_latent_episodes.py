import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def _write_episode_dataset(
    base_dir: Path,
    latents_sequences,
    group_names,
    puzzle_ids,
    set_name: str,
    *,
    use_jsonl: bool = False,
    extra_jsonl_records=None,
) -> None:
    split_dir = base_dir / "train"
    latents_dir = split_dir / "latents"
    metadata_dir = split_dir / "metadata"
    latents_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for idx, latents in enumerate(latents_sequences):
        episode_name = f"episode_{idx:03d}"
        np.savez(latents_dir / f"{episode_name}.npz", latents=latents)

        metadata = {
            "set": set_name,
            "group": group_names[idx],
            "puzzle_identifier": puzzle_ids[idx],
        }
        if use_jsonl:
            records = [metadata]
            if extra_jsonl_records:
                records.extend(extra_jsonl_records)
            with open(metadata_dir / f"{episode_name}.jsonl", "w", encoding="utf-8") as metadata_file:
                for record in records:
                    metadata_file.write(json.dumps(record) + "\n")
        else:
            with open(metadata_dir / f"{episode_name}.json", "w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file)


def test_episode_metadata_and_batching(tmp_path):
    dataset_a = tmp_path / "dataset_a"
    dataset_b = tmp_path / "dataset_b"

    latents_a = [
        np.array(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],
                [1.6, 1.7, 1.8],
                [1.9, 2.0, 2.1],
            ],
            dtype=np.float32,
        ),
    ]
    latents_b = [
        np.array(
            [
                [2.0, 2.1, 2.2],
                [2.3, 2.4, 2.5],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [3.0, 3.1, 3.2],
                [3.3, 3.4, 3.5],
                [3.6, 3.7, 3.8],
            ],
            dtype=np.float32,
        ),
    ]

    _write_episode_dataset(
        dataset_a,
        latents_sequences=latents_a,
        group_names=["group_a", "group_b"],
        puzzle_ids=[0, 1],
        set_name="latentset",
    )
    _write_episode_dataset(
        dataset_b,
        latents_sequences=latents_b,
        group_names=["group_a", "group_b"],
        puzzle_ids=[0, 1],
        set_name="latentset",
    )

    config = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[str(dataset_a), str(dataset_b)],
        global_batch_size=4,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    dataset = PuzzleDataset(config)

    # Metadata inferred from episodes
    assert dataset.metadata.task_type == "regression"
    assert dataset.metadata.input_dim == 3
    assert dataset.metadata.target_dim == 3
    assert dataset.metadata.input_pad_value == 0.0
    assert dataset.metadata.target_pad_value == 0.0
    assert dataset.metadata.num_puzzle_identifiers == 4

    # Load backing arrays lazily and inspect
    dataset._lazy_load_dataset()
    set_a = dataset._data["latentset"]
    set_b = dataset._data["latentset1"]

    total_examples_a = len(set_a["inputs"])
    total_examples_b = len(set_b["inputs"])
    assert total_examples_a == sum(len(latent) - 1 for latent in latents_a)
    assert total_examples_b == sum(len(latent) - 1 for latent in latents_b)

    indices_a = np.arange(total_examples_a)
    inputs_a = set_a["inputs"][indices_a]
    targets_a = set_a["targets"][indices_a]
    mask_a = set_a["target_mask"][indices_a]

    assert inputs_a.dtype == np.float32
    assert targets_a.dtype == np.float32
    assert mask_a.dtype == np.bool_
    assert inputs_a.shape == targets_a.shape == (total_examples_a, 3)
    assert mask_a.shape == (total_examples_a, 3)

    puzzle_indices_a = set_a["puzzle_indices"]
    for episode_idx in range(len(latents_a)):
        start, end = puzzle_indices_a[episode_idx], puzzle_indices_a[episode_idx + 1]
        expected_inputs = latents_a[episode_idx][:-1]
        expected_targets = latents_a[episode_idx][1:]
        np.testing.assert_allclose(inputs_a[start:end], expected_inputs)
        np.testing.assert_allclose(targets_a[start:end], expected_targets)

    # Puzzle identifiers are offset across datasets
    assert np.array_equal(set_a["puzzle_identifiers"], np.array([0, 1], dtype=np.int32))
    assert np.array_equal(set_b["puzzle_identifiers"], np.array([2, 3], dtype=np.int32))

    # Group and puzzle index dtypes
    assert set_a["puzzle_indices"].dtype == np.int32
    assert set_a["group_indices"].dtype == np.int32

    # Iterate to obtain batches and validate tensor types
    iterator_outputs = list(dataset)
    assert {set_name for set_name, _, _ in iterator_outputs} == {"latentset", "latentset1"}
    for set_name, batch, _ in iterator_outputs:
        assert set_name in {"latentset", "latentset1"}
        assert batch["inputs"].dtype == torch.float32
        assert batch["targets"].dtype == torch.float32
        assert batch["target_mask"].dtype == torch.bool
        assert batch["puzzle_identifiers"].dtype == torch.int32


def test_episode_metadata_from_jsonl(tmp_path):
    dataset_dir = tmp_path / "dataset_jsonl"
    latents_dir = dataset_dir / "train" / "latents"
    metadata_dir = dataset_dir / "train" / "metadata"
    latents_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    latents = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
        ],
        dtype=np.float32,
    )
    np.savez(latents_dir / "episode_000.npz", latents=latents)

    with open(metadata_dir / "episode_000.jsonl", "w", encoding="utf-8") as metadata_file:
        metadata_file.write(json.dumps({"set": "jsonlset", "group": "jsonl_group"}) + "\n")
        metadata_file.write(json.dumps({"puzzle_identifier": 42}) + "\n")

    config = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[str(dataset_dir)],
        global_batch_size=4,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    dataset = PuzzleDataset(config)

    episode_info = dataset._episode_datasets[str(dataset_dir)]
    assert episode_info.episodes[0].metadata.set_name == "jsonlset"
    assert episode_info.episodes[0].metadata.group_name == "jsonl_group"
    assert episode_info.episodes[0].metadata.puzzle_identifier == 42

    dataset._lazy_load_dataset()
    set_data = dataset._data["jsonlset"]
    assert np.array_equal(set_data["puzzle_identifiers"], np.array([42], dtype=np.int32))

    outputs = list(dataset)
    assert len(outputs) == 1
    set_name, batch, effective = outputs[0]
    assert set_name == "jsonlset"
    assert effective == 2
    assert batch["puzzle_identifiers"].dtype == torch.int32
    assert batch["puzzle_identifiers"][0:effective].tolist() == [42, 42]
    assert batch["puzzle_identifiers"][effective:].tolist() == [0, 0]
