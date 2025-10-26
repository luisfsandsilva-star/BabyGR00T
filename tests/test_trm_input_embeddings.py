import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.recursive_reasoning.trm import (  # noqa: E402
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1_Inner,
)
from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig  # noqa: E402


def _write_episode_dataset(base_dir: Path, latents_sequences, puzzle_ids, set_name: str) -> None:
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
            "group": f"group_{idx}",
            "puzzle_identifier": int(puzzle_ids[idx]),
        }
        with open(metadata_dir / f"{episode_name}.json", "w", encoding="utf-8") as metadata_file:
            json.dump(metadata, metadata_file)


def test_input_embeddings_handles_missing_sequence_dim(tmp_path):
    dataset_dir = tmp_path / "episodes"

    latents_sequences = [
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=np.float32,
        ),
        np.array(
            [
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],
            ],
            dtype=np.float32,
        ),
    ]
    _write_episode_dataset(
        dataset_dir,
        latents_sequences=latents_sequences,
        puzzle_ids=[0, 1],
        set_name="latentset",
    )

    dataset_config = PuzzleDatasetConfig(
        seed=123,
        dataset_paths=[str(dataset_dir)],
        global_batch_size=2,
        test_set_mode=True,
        epochs_per_iter=1,
        rank=0,
        num_replicas=1,
    )
    dataset = PuzzleDataset(dataset_config)
    _set_name, batch, _ = next(iter(dataset))

    model_config = TinyRecursiveReasoningModel_ACTV1Config(
        batch_size=int(batch["inputs"].shape[0]),
        seq_len=1,
        puzzle_emb_ndim=5,
        num_puzzle_identifiers=int(dataset.metadata.num_puzzle_identifiers),
        vocab_size=1,
        H_cycles=1,
        L_cycles=1,
        H_layers=1,
        L_layers=1,
        hidden_size=4,
        latent_dim=int(dataset.metadata.input_dim),
        output_dim=int(dataset.metadata.target_dim),
        expansion=1.0,
        num_heads=1,
        pos_encodings="rope",
        halt_max_steps=1,
        halt_exploration_prob=0.0,
        forward_dtype="float32",
        mlp_t=False,
        puzzle_emb_len=2,
        no_ACT_continue=True,
        use_cross_attn=False,
        keep_act_halting_head=False,
        use_constant_cross_attn=False,
        cross_attn_constant_dim=0,
    )

    inner = TinyRecursiveReasoningModel_ACTV1_Inner(model_config)
    inner.eval()

    embeddings, _ = inner._input_embeddings(batch["inputs"], batch.get("puzzle_identifiers"))

    expected_shape = (
        batch["inputs"].shape[0],
        inner.config.seq_len + inner.puzzle_emb_len,
        inner.config.hidden_size,
    )
    assert embeddings.shape == expected_shape
    assert isinstance(embeddings, torch.Tensor)
