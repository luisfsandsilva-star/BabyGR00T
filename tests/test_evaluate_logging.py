from types import SimpleNamespace

import numpy as np
import torch

import pretrain


class _DummyEvalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def initial_carry(self, batch):
        return SimpleNamespace()

    def forward(self, *, carry, batch, return_keys):
        metrics = {
            "loss": torch.tensor(1.0),
            "count": torch.tensor(2.0),
        }
        preds = {"preds": torch.zeros((batch["inputs"].shape[0], 1))}
        return carry, torch.tensor(0.0), metrics, preds, True


class _DummyEvaluator:
    required_outputs = ()

    def begin_eval(self):
        return None

    def update_batch(self, batch, preds):
        return None

    def result(self, save_path, rank, world_size, group):
        return {"val/loss": 0.5}


class _DummyEvalLoader:
    def __init__(self):
        self.dataset = SimpleNamespace(
            config=SimpleNamespace(global_batch_size=2),
            _data={"eval": {"inputs": np.zeros((4, 3), dtype=np.float32)}},
            _lazy_load_dataset=lambda: None,
        )

    def __iter__(self):
        batch = {
            "inputs": torch.zeros((2, 3), dtype=torch.float32),
        }
        for _ in range(2):
            yield "eval", batch.copy(), 2


def test_evaluate_default_logging_minimal_output(monkeypatch, capsys):
    monkeypatch.setattr(
        pretrain.torch.Tensor, "cuda", lambda self, *args, **kwargs: self, raising=False
    )

    real_zeros = torch.zeros

    def _zeros(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "device"}
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(pretrain.torch, "zeros", _zeros)

    class _DummyDeviceContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(pretrain.torch, "device", lambda *args, **kwargs: _DummyDeviceContext())

    config = pretrain.PretrainConfig(
        arch=pretrain.ArchConfig(
            name="losses@Dummy",  # Unused but required
            loss=pretrain.LossConfig(name="losses@Dummy"),
        ),
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

    metadata = pretrain.PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
        vocab_size=32,
        seq_len=1,
        num_puzzle_identifiers=1,
        total_groups=1,
        mean_puzzle_examples=1.0,
        total_puzzles=1,
        sets=["eval"],
        task_type="classification",
    )

    train_state = pretrain.TrainState(
        model=_DummyEvalModel(),
        optimizers=(),
        optimizer_lrs=(),
        carry=None,
        step=0,
        total_steps=1,
    )

    pretrain.evaluate(
        config,
        train_state,
        _DummyEvalLoader(),
        metadata,
        evaluators=[_DummyEvaluator()],
        rank=0,
        world_size=1,
        cpu_group=None,
        progress_bar=None,
    )

    captured = capsys.readouterr()
    combined = (captured.out + captured.err).splitlines()
    non_empty = [line for line in combined if line.strip()]
    assert len(non_empty) <= 1
