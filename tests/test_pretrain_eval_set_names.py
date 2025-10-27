import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pretrain import _resolve_eval_set_names
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig


def _write_episode_dataset(base_dir: Path, latents_sequences, set_name: str) -> None:
    split_dir = base_dir / "train"
    latents_dir = split_dir / "latents"
    metadata_dir = split_dir / "metadata"
    latents_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for idx, latents in enumerate(latents_sequences):
        episode_name = f"episode_{idx:03d}"
        np.savez(latents_dir / f"{episode_name}.npz", latents=latents)

        with open(metadata_dir / f"{episode_name}.json", "w", encoding="utf-8") as metadata_file:
            json.dump({"set": set_name}, metadata_file)


def test_resolve_eval_set_names_with_multiple_latent_roots(tmp_path):
    dataset_a = tmp_path / "dataset_a"
    dataset_b = tmp_path / "dataset_b"

    latents_a = [
        np.array([[0.0, 0.1], [0.2, 0.3]], dtype=np.float32),
        np.array([[0.4, 0.5], [0.6, 0.7], [0.8, 0.9]], dtype=np.float32),
    ]
    latents_b = [
        np.array([[1.0, 1.1], [1.2, 1.3]], dtype=np.float32),
    ]

    _write_episode_dataset(dataset_a, latents_sequences=latents_a, set_name="latentset")
    _write_episode_dataset(dataset_b, latents_sequences=latents_b, set_name="latentset")

    config = PuzzleDatasetConfig(
        seed=0,
        dataset_paths=[str(dataset_a), str(dataset_b)],
        global_batch_size=2,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )

    dataset = PuzzleDataset(config)

    # The metadata still lists the base set name.
    assert dataset.metadata.sets == ["latentset"]

    loader = SimpleNamespace(dataset=dataset)
    resolved = _resolve_eval_set_names(loader, dataset.metadata)

    assert resolved == ["latentset", "latentset1"]

    dataset._lazy_load_dataset()
    assert set(dataset._data.keys()) == {"latentset", "latentset1"}

    iter_names = {set_name for set_name, _, _ in dataset}
    assert iter_names == set(resolved)
