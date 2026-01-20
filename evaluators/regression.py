from typing import Dict, Optional

import torch


class Regression:
    """
    Minimal evaluator for regression metrics.
    Usage in config: name: regression@Regression
    """

    def __init__(self, data_path: str, eval_metadata, **kwargs):  # noqa: ARG002
        self.required_outputs = {"logits"}
        self.reset()

    def reset(self):
        self._sum_mse = 0.0
        self._sum_mae = 0.0
        self._sum_count = 0.0

    def begin_eval(self):
        self.reset()

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        y = batch["targets"]
        yhat = preds["logits"]
        mask = batch.get("mask")

        if yhat.dim() == 2 and y.dim() == 3:
            # collapse targets for point regression
            if mask is not None:
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
                y = (y * mask.unsqueeze(-1)).sum(dim=1) / denom
            else:
                y = y[:, -1]

        diff2 = (yhat - y).pow(2)
        diff1 = (yhat - y).abs()
        if mask is not None and yhat.dim() == 3:
            diff2 = diff2 * mask.unsqueeze(-1)
            diff1 = diff1 * mask.unsqueeze(-1)
            count = mask.sum().clamp_min(1).item()
        else:
            count = yhat.numel() / yhat.size(-1)

        self._sum_mse += diff2.sum().item()
        self._sum_mae += diff1.sum().item()
        self._sum_count += count

    def result(self, save_dir: Optional[str], rank: int, world_size: int, group=None):  # noqa: ARG002
        if self._sum_count == 0:
            return None
        return {
            "mse": self._sum_mse / self._sum_count,
            "mae": self._sum_mae / self._sum_count,
        }


