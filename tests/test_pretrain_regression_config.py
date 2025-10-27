from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from torch import nn
from torch.optim import Optimizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pretrain
from dataset.common import PuzzleDatasetMetadata
from models.losses import ACTRegressionLossHead


class _DummyDeviceContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class DummyRegressionModel(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        latent_dim = config_dict.get("latent_dim", 3) or 3
        output_dim = config_dict.get("output_dim", 2) or 2
        self.linear = nn.Linear(latent_dim, output_dim)
        self.config = SimpleNamespace(keep_act_halting_head=True)

    def initial_carry(self, batch):
        batch_size = batch["inputs"].shape[0]
        target_shape = (batch_size, 1, self.linear.out_features)
        zeros = torch.zeros(target_shape, dtype=batch["inputs"].dtype)
        return SimpleNamespace(
            current_data={
                "targets": zeros,
                "target_mask": torch.ones_like(zeros, dtype=torch.bool),
            },
            halted=torch.ones(batch_size, dtype=torch.bool),
            steps=torch.zeros(batch_size, dtype=torch.int64),
        )

    def forward(self, carry, batch):
        batch_size = batch["inputs"].shape[0]
        logits = self.linear(batch["inputs"]).unsqueeze(1)
        new_carry = SimpleNamespace(
            current_data={
                "targets": batch["targets"],
                "target_mask": batch["target_mask"],
            },
            halted=torch.ones(batch_size, dtype=torch.bool),
            steps=torch.zeros(batch_size, dtype=torch.int64),
        )
        return new_carry, {"logits": logits}


def test_create_model_with_regression_loss_filters_kwargs(monkeypatch):
    arch_name = "recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV1"

    original_loader = pretrain.load_model_class

    def fake_loader(identifier):
        if identifier == arch_name:
            return DummyRegressionModel
        return original_loader(identifier)

    class _DummyLion(Optimizer):
        def __init__(self, params, lr=0.0, **kwargs):
            super().__init__(params, defaults={"lr": lr})

        def step(self, closure=None):
            if closure is not None:
                closure()

    monkeypatch.setattr(pretrain, "Lion", _DummyLion)
    monkeypatch.setattr(pretrain, "load_model_class", fake_loader)
    monkeypatch.setattr(pretrain, "load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(pretrain.torch, "device", lambda *args, **kwargs: _DummyDeviceContext())
    monkeypatch.setenv("DISABLE_COMPILE", "1")

    loss_config = pretrain.LossConfig(
        name="losses@ACTRegressionLossHead",
        loss_type="stablemax_cross_entropy",
    )
    arch_config = pretrain.ArchConfig(
        name=arch_name,
        loss=loss_config,
        puzzle_emb_ndim=0,
    )

    config = pretrain.PretrainConfig(
        arch=arch_config,
        data_paths=["dummy"],
        global_batch_size=2,
        epochs=1,
        lr=1e-4,
        lr_min_ratio=0.1,
        lr_warmup_steps=1,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        puzzle_emb_lr=1e-4,
        puzzle_emb_weight_decay=0.0,
    )

    metadata = PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
        vocab_size=32,
        seq_len=1,
        num_puzzle_identifiers=1,
        total_groups=1,
        mean_puzzle_examples=1.0,
        total_puzzles=1,
        sets=["train"],
        task_type="regression",
        input_dim=3,
        target_dim=2,
        input_pad_value=0.0,
        target_pad_value=0.0,
    )

    model, optimizers, optimizer_lrs = pretrain.create_model(config, metadata, rank=0, world_size=1)

    assert isinstance(model, ACTRegressionLossHead)
    assert len(optimizers) == 1
    assert len(optimizer_lrs) == 1

    batch_size = config.global_batch_size
    inputs = torch.zeros((batch_size, metadata.input_dim), dtype=torch.float32)
    targets = torch.zeros((batch_size, 1, metadata.target_dim), dtype=torch.float32)
    target_mask = torch.ones_like(targets, dtype=torch.bool)

    carry = model.initial_carry({"inputs": inputs, "targets": targets, "target_mask": target_mask})
    new_carry, loss, metrics, outputs, halted = model(
        return_keys=("logits",),
        carry=carry,
        batch={"inputs": inputs, "targets": targets, "target_mask": target_mask},
    )

    assert torch.is_tensor(loss) and loss.ndim == 0
    assert halted.ndim == 0 and halted.dtype == torch.bool
    assert outputs["logits"].shape == targets.shape
    assert metrics["regression_loss"].shape == torch.Size([])
