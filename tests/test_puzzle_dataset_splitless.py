import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def test_splitless_episode_dataset(tmp_path):
    dataset_dir = tmp_path / "splitless_dataset"
    latents_dir = dataset_dir / "latents"
    metadata_dir = dataset_dir / "metadata"
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

    metadata = {
        "set": "splitless_set",
        "group": "splitless_group",
        "puzzle_identifier": 0,
    }
    with open(metadata_dir / "episode_000.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file)

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

    episode_info = dataset._episode_datasets[str(dataset_dir)]
    assert Path(episode_info.latents_dir) == latents_dir
    assert Path(episode_info.metadata_dir) == metadata_dir

    dataset._lazy_load_dataset()
    set_data = dataset._data[metadata["set"]]

    expected_inputs = latents[:-1]
    expected_targets = latents[1:]

    indices = np.arange(len(set_data["inputs"]))
    inputs = set_data["inputs"][indices]
    targets = set_data["targets"][indices]
    mask = set_data["target_mask"][indices]

    np.testing.assert_allclose(inputs, expected_inputs)
    np.testing.assert_allclose(targets, expected_targets)
    assert mask.dtype == np.bool_
    assert np.all(mask)

    set_name, batch, batch_size = next(iter(dataset))
    assert batch_size == config.global_batch_size
    assert set_name == metadata["set"]
    assert torch.allclose(batch["inputs"], torch.from_numpy(expected_inputs))
    assert torch.allclose(batch["targets"], torch.from_numpy(expected_targets))
    assert torch.all(batch["target_mask"])
