from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),

                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
            if "q_halt_logits" in outputs:
                metrics["q_halt_accuracy"] = (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum()
            else:
                metrics["q_halt_accuracy"] = torch.zeros((), dtype=torch.int64, device=valid_metrics.device)

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        metrics["lm_loss"] = lm_loss.detach()

        q_halt_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)
        if "q_halt_logits" in outputs:
            q_halt_loss = F.binary_cross_entropy_with_logits(
                outputs["q_halt_logits"],
                seq_is_correct.to(outputs["q_halt_logits"].dtype),
                reduction="sum",
            )
        metrics["q_halt_loss"] = q_halt_loss.detach()
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class ACTRegressionLossHead(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        mse_weight: float = 1.0,
        mae_weight: float = 0.0,
        keep_act_halting_head: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight

        base_keep_act = getattr(getattr(model, "config", None), "keep_act_halting_head", True)
        self.keep_act_halting_head = base_keep_act if keep_act_halting_head is None else keep_act_halting_head

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)

        predictions = outputs["logits"].to(torch.float32)
        targets = new_carry.current_data["targets"].to(torch.float32)
        target_mask = new_carry.current_data["target_mask"].to(torch.bool)

        batch_size = predictions.shape[0]
        flat_mask = target_mask.view(batch_size, -1)
        valid_counts = flat_mask.sum(-1)
        valid_sequences = new_carry.halted & (valid_counts > 0)

        valid_counts_clamped = valid_counts.clamp_min(1).to(predictions.dtype)
        diff = predictions - targets
        mse_errors = torch.where(target_mask, diff.square(), torch.zeros_like(diff))
        mae_errors = torch.where(target_mask, diff.abs(), torch.zeros_like(diff))

        mse_sum = mse_errors.view(batch_size, -1).sum(-1)
        mae_sum = mae_errors.view(batch_size, -1).sum(-1)

        per_example_mse = torch.where(
            valid_counts > 0,
            mse_sum / valid_counts_clamped,
            torch.zeros_like(valid_counts_clamped),
        )
        per_example_mae = torch.where(
            valid_counts > 0,
            mae_sum / valid_counts_clamped,
            torch.zeros_like(valid_counts_clamped),
        )

        mse_loss = per_example_mse.sum() if self.mse_weight != 0 else torch.zeros((), device=predictions.device)
        mae_loss = per_example_mae.sum() if self.mae_weight != 0 else torch.zeros((), device=predictions.device)

        total_loss = self.mse_weight * mse_loss + self.mae_weight * mae_loss

        with torch.no_grad():
            outputs["preds"] = predictions.detach()

        raw_metrics: Dict[str, torch.Tensor] = {
            "count": valid_sequences.sum(),
            "regression_loss": total_loss,
            "mse": torch.where(valid_sequences, per_example_mse, torch.zeros_like(per_example_mse)).sum(),
            "mae": torch.where(valid_sequences, per_example_mae, torch.zeros_like(per_example_mae)).sum(),
            "steps": torch.where(
                valid_sequences,
                new_carry.steps.to(torch.int64),
                torch.zeros_like(new_carry.steps.to(torch.int64)),
            ).sum(),
            "q_halt_loss": torch.zeros((), dtype=predictions.dtype, device=predictions.device),
            "q_continue_loss": torch.zeros((), dtype=predictions.dtype, device=predictions.device),
        }

        metrics = {key: value.detach() for key, value in raw_metrics.items()}

        if not self.keep_act_halting_head:
            outputs.pop("q_halt_logits", None)
            outputs.pop("q_continue_logits", None)
            outputs.pop("target_q_continue", None)

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

