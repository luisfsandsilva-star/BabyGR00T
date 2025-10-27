import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.losses import ACTRegressionLossHead


class _DummyRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
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
        return SimpleNamespace(
            current_data={
                "targets": batch["targets"],
                "target_mask": batch["target_mask"],
            },
            halted=torch.ones(batch_size, dtype=torch.bool),
            steps=torch.arange(1, batch_size + 1, dtype=torch.int64),
        ), {"logits": logits}


def test_act_regression_loss_metrics_are_detached():
    torch.manual_seed(0)
    base_model = _DummyRegressionModel()
    loss_head = ACTRegressionLossHead(base_model)

    batch_size = 2
    inputs = torch.randn(batch_size, 3)
    targets = torch.randn(batch_size, 1, 2)
    target_mask = torch.tensor([[[True, False]], [[True, True]]])

    carry = loss_head.initial_carry(
        {"inputs": inputs, "targets": targets, "target_mask": target_mask}
    )

    _, loss, metrics, _, _ = loss_head(
        return_keys=("logits",),
        carry=carry,
        batch={"inputs": inputs, "targets": targets, "target_mask": target_mask},
    )

    loss.backward()

    assert all(isinstance(value, torch.Tensor) and not value.requires_grad for value in metrics.values())
