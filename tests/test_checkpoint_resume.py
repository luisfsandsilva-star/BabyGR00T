from types import SimpleNamespace
from pathlib import Path
import sys

import torch
from torch import nn
from torch.optim import SGD

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pretrain


def _make_config(checkpoint_dir, *, resume=False, resume_from=None):
    checkpoint = SimpleNamespace(
        path=str(checkpoint_dir),
        resume=resume,
        resume_from=resume_from,
        monitor="val/loss",
        mode="min",
        keep_last=None,
    )
    return SimpleNamespace(
        checkpoint=checkpoint,
        checkpoint_path=str(checkpoint_dir),
        epochs=10,
        global_batch_size=1,
    )


def test_save_checkpoint_records_optimizer_and_best(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    config = _make_config(checkpoint_dir)

    model = nn.Linear(2, 2)
    optimizer = SGD(model.parameters(), lr=0.1)
    train_state = pretrain.TrainState(
        model=model,
        optimizers=[optimizer],
        optimizer_lrs=[0.1],
        carry=None,
        step=12,
        total_steps=100,
        epoch=5,
        best_val_metric=0.5,
    )

    train_state.best_val_metric = 0.4

    ema_state = {"weight": torch.ones_like(next(model.parameters()))}

    pretrain.save_train_state(
        config,
        train_state,
        monitor_value=0.4,
        is_best=True,
        ema_state=ema_state,
    )

    step_file = checkpoint_dir / "step_00000012.ckpt"
    best_file = checkpoint_dir / "best.ckpt"
    latest_file = checkpoint_dir / "latest.txt"

    assert step_file.exists()
    assert best_file.exists()
    assert latest_file.read_text().strip() == "step_00000012.ckpt"

    payload = torch.load(step_file)
    assert payload["step"] == 12
    assert payload["epoch"] == 5
    assert payload["best_val_metric"] == 0.4
    assert payload["monitor_value"] == 0.4
    torch.testing.assert_close(payload["ema_state"]["weight"], ema_state["weight"])
    assert payload["optimizer_states"][0] == optimizer.state_dict()


def test_load_checkpoint_restores_training_state(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "ckpt"
    config = _make_config(checkpoint_dir)

    base_model = nn.Linear(2, 2)
    optimizer = SGD(base_model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    train_state = pretrain.TrainState(
        model=base_model,
        optimizers=[optimizer],
        optimizer_lrs=[0.1],
        carry=None,
        step=7,
        total_steps=100,
        epoch=3,
        best_val_metric=0.6,
    )
    train_state.schedulers = [scheduler]
    scheduler.step()

    ema_state = {"weight": torch.full_like(next(base_model.parameters()), 2.0)}

    train_state.best_val_metric = 0.55

    pretrain.save_train_state(
        config,
        train_state,
        monitor_value=0.55,
        is_best=True,
        ema_state=ema_state,
    )

    resume_config = _make_config(checkpoint_dir, resume=True)
    payload = pretrain.load_checkpoint(resume_config)
    assert payload is not None

    def _dummy_create_model(config, train_metadata, rank, world_size, *, model_state=None):
        model = nn.Linear(2, 2)
        if model_state is not None:
            model.load_state_dict(model_state)
        optimizer = SGD(model.parameters(), lr=0.1)
        return model, [optimizer], [0.1]

    monkeypatch.setattr(pretrain, "create_model", _dummy_create_model)

    metadata = SimpleNamespace(total_groups=1, mean_puzzle_examples=1.0)

    resumed_state = pretrain.init_train_state(
        resume_config,
        metadata,
        rank=0,
        world_size=1,
        model_state=payload["model_state"],
        optimizer_states=payload["optimizer_states"],
        scheduler_states=payload["scheduler_state"],
        initial_step=payload["step"],
        initial_epoch=payload["epoch"],
        best_val_metric=payload["best_val_metric"],
        ema_state=payload["ema_state"],
    )

    assert resumed_state.step == 7
    assert resumed_state.epoch == 3
    assert resumed_state.best_val_metric == 0.55
    assert resumed_state.scheduler_states == payload["scheduler_state"]
    torch.testing.assert_close(resumed_state.ema_state["weight"], ema_state["weight"])

    torch.testing.assert_close(
        resumed_state.model.state_dict()["weight"],
        payload["model_state"]["weight"],
    )
    assert resumed_state.optimizers[0].state_dict()["state"] == payload["optimizer_states"][0]["state"]


def test_load_checkpoint_converts_legacy_best_metric(tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()

    model = nn.Linear(2, 2)
    optimizer_state = SGD(model.parameters(), lr=0.1).state_dict()

    legacy_payload = {
        "model_state": model.state_dict(),
        "optimizer_states": [optimizer_state],
        "scheduler_state": [{"step": 1}],
        "step": 1,
        "epoch": 1,
        "best_val_loss": 0.5,
        "monitor": "val/loss",
        "monitor_value": 0.5,
        "ema_state": None,
    }

    ckpt_path = checkpoint_dir / "step_00000001.ckpt"
    torch.save(legacy_payload, ckpt_path)
    (checkpoint_dir / "latest.txt").write_text(ckpt_path.name)

    config = _make_config(checkpoint_dir, resume=True)
    payload = pretrain.load_checkpoint(config)

    assert payload is not None
    assert payload["best_val_metric"] == legacy_payload["best_val_loss"]
    assert "best_val_loss" in payload
    assert payload["optimizer_states"][0] == optimizer_state
