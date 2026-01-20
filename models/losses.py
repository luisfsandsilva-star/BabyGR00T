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
        # Track, per slot in the batch, whether the halt-condition for this
        # reasoning episode has already been satisfied. We reset this state
        # whenever a new episode starts in that slot.
        self._halt_condition_met: Optional[torch.Tensor] = None
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits for one reasoning step.
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

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()

        # q_halt_loss:
        # We want to check the halt-condition at each *reasoning step* and,
        # for a given episode/sequence, only apply the loss once: on the
        # first step where the condition is satisfied.
        q_halt_logits = outputs["q_halt_logits"]
        batch_size = q_halt_logits.shape[0]

        # Lazily initialise / resize per-slot state.
        if (self._halt_condition_met is None) or (self._halt_condition_met.shape[0] != batch_size):
            self._halt_condition_met = torch.zeros(batch_size, dtype=torch.bool, device=q_halt_logits.device)
        else:
            # Reset state for slots where a new episode starts. We treat
            # steps == 1 as "first reasoning step" of a fresh sequence.
            new_episode = (new_carry.steps == 1)
            # Where new_episode is True, clear the flag; otherwise keep it.
            self._halt_condition_met = torch.where(
                new_episode.to(self._halt_condition_met.dtype).bool(),
                torch.zeros_like(self._halt_condition_met, dtype=torch.bool),
                self._halt_condition_met,
            )

        # Condition for this reasoning step: the full sequence prediction at
        # this step is exactly correct.
        step_condition = seq_is_correct
        # We trigger the loss for those slots where the condition becomes
        # true *for the first time* in this episode.
        new_trigger_mask = (~self._halt_condition_met) & step_condition

        if new_trigger_mask.any():
            # Compute q_halt_loss only on the newly triggered slots.
            q_halt_loss = F.binary_cross_entropy_with_logits(
                q_halt_logits[new_trigger_mask],
                step_condition[new_trigger_mask].to(q_halt_logits.dtype),
                reduction="sum",
            )
            # Mark these slots as having satisfied the halt-condition.
            self._halt_condition_met = self._halt_condition_met | new_trigger_mask
        else:
            # No new sequences satisfied the condition on this step.
            q_halt_loss = torch.tensor(0.0, device=q_halt_logits.device)

        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")

            metrics["q_continue_loss"] = q_continue_loss.detach()
        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()


class RegressionACTLossHead(nn.Module):
    """
    Loss head for regression task with ACT. Computes MSE or Huber on outputs["logits"].
    Supports optional q-head losses same as LM path to preserve ACT training dynamics.
    """
    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "mse",
        huber_delta: float = 1.0,
        halt_tolerance: float = 1e-2,
        halt_target_mode: str = "tolerance",  # "tolerance" | "improvement"
    ):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        # If per-(sequence,timestep) MSE <= halt_tolerance, we treat that (b, l)
        # position as "correct" for q_halt in tolerance mode.
        self.halt_tolerance = float(halt_tolerance)
        # Mode for defining q_halt targets:
        #  - "tolerance": 1 si el MSE en el timestep físico actual está por debajo de halt_tolerance.
        #  - "improvement": 1 sólo si el MSE local actual ha mejorado respecto al reasoning step anterior.
        self.halt_target_mode = halt_target_mode
        # Histórico de MSE por (batch, timestep físico) para modo "improvement".
        self._prev_mse_per_l: Optional[torch.Tensor] = None  # [B, L]
        # Track, per (batch, timestep físico), whether the halt-condition has
        # already been satisfied for the current reasoning episode.
        self._halt_condition_met: Optional[torch.Tensor] = None  # [B, L]

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def _regression_loss(self, pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor | None):
        if self.loss_type == "huber":
            loss = F.smooth_l1_loss(pred, tgt, reduction="none", beta=self.huber_delta)
        else:
            loss = (pred - tgt).pow(2)
        if mask is not None:
            # mask: [B, L] for sequence, broadcast over features
            loss = loss * mask.unsqueeze(-1)
        return loss.mean()

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)
        preds = outputs["logits"]

        # Targets and optional mask
        targets = new_carry.current_data.get("targets")
        if targets is None:
            raise RuntimeError("RegressionACTLossHead expects 'targets' in batch/current_data.")
        mask = new_carry.current_data.get("mask")

        # Enforce sequence regression only (no point regression). Expect [B, L, F] for preds and targets.
        if not (preds.dim() == 3 and targets.dim() == 3):
            raise RuntimeError("RegressionACTLossHead expects sequence regression with preds and targets of shape [B, L, F].")

        # RevIN target normalization
        # If model outputs are normalized (revin_apply_on_outputs=True), we must normalize targets to match.
        if getattr(self.model.config, "revin_apply_on_outputs", False):
            revin_ctx = outputs.get("revin_ctx")
            if revin_ctx is not None:
                # Use the revin module from the model to normalize targets
                if hasattr(self.model.inner, "revin") and self.model.inner.revin is not None:
                    targets = self.model.inner.revin.normalize_in(targets, revin_ctx)

        # Teacher forcing next-step option: compare ŷ_t with y_{t+1} over sequence
        # Controlled via arch.teacher_forcing_shift=true (non-breaking; defaults to False)
        try:
            tf_shift = bool(getattr(self.model.config, 'teacher_forcing_shift', False))  # type: ignore[attr-defined]
        except Exception:
            tf_shift = False

        if tf_shift and preds.dim() == 3 and targets.dim() == 3 and preds.size(1) >= 2 and targets.size(1) >= 2:
            # Align by one step forward along sequence dimension
            preds = preds[:, :-1]
            targets = targets[:, 1:]
            if mask is not None and mask.dim() == 2:
                mask = mask[:, 1:]

        reg_loss = self._regression_loss(preds, targets, mask)

        # Metrics and q-head losses (if present)
        # "count" se usa para promediar en pretrain.py; "steps" permite loggear reasoning steps promedio.
        metrics: Dict[str, torch.Tensor] = {
            "count": torch.tensor(preds.shape[0], device=preds.device, dtype=torch.float32),
            "steps": new_carry.steps.to(torch.float32).sum(),
        }
        q_halt_logits = outputs.get("q_halt_logits")
        if q_halt_logits is not None:
            with torch.no_grad():
                # Per-element squared error [B, L, F]
                sq_err_full = (preds - targets).pow(2)
                B, L, Fdim = sq_err_full.shape
                device = preds.device

                # Para el target de halt, cada timestep físico L se trata de
                # forma independiente. Definimos un MSE local por (b, l)
                # promediando sólo sobre la dimensión de features F, nunca
                # sobre L ni sobre reasoning steps.
                if mask is not None and mask.dim() == 2:
                    # mask: [B, L] -> broadcast a [B, L, 1]
                    m = mask.unsqueeze(-1)
                    sq_err_valid = sq_err_full * m
                    valid_counts = m.sum(dim=-1).clamp_min(1)  # [B, L]
                    mse_per_l = sq_err_valid.sum(dim=-1) / valid_counts  # [B, L]
                    valid_pos = mask.to(torch.bool)  # [B, L]
                else:
                    mse_per_l = sq_err_full.mean(dim=-1)  # [B, L]
                    valid_pos = torch.ones((B, L), dtype=torch.bool, device=device)

                if self.halt_target_mode == "tolerance":
                    # Target de halt por (b, l): 1 si el MSE local de ese
                    # timestep físico está por debajo de la tolerancia.
                    target_halt_per_l = torch.zeros(
                        (B, L), dtype=q_halt_logits.dtype, device=device
                    )
                    target_halt_per_l[valid_pos] = (
                        mse_per_l[valid_pos] <= self.halt_tolerance
                    ).to(dtype=q_halt_logits.dtype)

                elif self.halt_target_mode == "improvement":
                    # Target = 1 sólo cuando el MSE local actual es menor que
                    # el MSE local del reasoning step anterior, por (b, l).
                    if (
                        self._prev_mse_per_l is None
                        or self._prev_mse_per_l.shape != mse_per_l.shape
                    ):
                        # Primera vez o cambio de tamaño: no consideramos mejora.
                        improved = torch.zeros(
                            (B, L), dtype=q_halt_logits.dtype, device=device
                        )
                    else:
                        # Resetear histórico para episodios nuevos (steps == 1).
                        new_episode = (new_carry.steps == 1).view(B, 1).expand(-1, L)
                        prev = torch.where(
                            new_episode,
                            mse_per_l,
                            self._prev_mse_per_l.to(device=mse_per_l.device),
                        )
                        improved = (mse_per_l < prev).to(dtype=q_halt_logits.dtype)

                    self._prev_mse_per_l = mse_per_l.detach()
                    
                    # Fix: also halt if we are below tolerance, even if not strictly improving
                    # (e.g. we converged to a good solution).
                    is_good_enough = (mse_per_l <= self.halt_tolerance).to(dtype=q_halt_logits.dtype)
                    target_halt_per_l = torch.maximum(improved, is_good_enough)

                else:
                    raise ValueError(f"Unknown halt_target_mode: {self.halt_target_mode}")

            # Igual que en ACTLossHead (LM), queremos que para cada (b, l) la
            # q_halt_loss se aplique sólo en el primer reasoning step en el que
            # se satisface la condición local. A partir de ahí, ese (b, l)
            # deja de contribuir a la loss.

            # Inicializar / reajustar el estado por (batch, L) si cambia tamaño.
            if (
                self._halt_condition_met is None
                or self._halt_condition_met.shape != (B, L)
            ):
                self._halt_condition_met = torch.zeros(
                    (B, L), dtype=torch.bool, device=q_halt_logits.device
                )
            else:
                # Resetear episodios nuevos (primer paso de razonamiento) para
                # todos los timesteps físicos de esas secuencias.
                new_episode = (new_carry.steps == 1).view(B, 1).expand(-1, L)
                self._halt_condition_met = torch.where(
                    new_episode,
                    torch.zeros_like(self._halt_condition_met, dtype=torch.bool),
                    self._halt_condition_met,
                )

            # Condición de parada local en este step (0/1 en float).
            # step_condition used to filter ONLY positives. We want to train on all valid steps until halt.
            
            # Train on all slots that have NOT yet met the halt condition
            train_mask = (~self._halt_condition_met) & valid_pos
            
            if train_mask.any():
                # Expandir logits [B] -> [B, L] para aplicar BCE por-timestep.
                q_logits_exp = q_halt_logits.unsqueeze(1).expand(-1, L)
                q_halt_loss = F.binary_cross_entropy_with_logits(
                    q_logits_exp[train_mask],
                    target_halt_per_l[train_mask],
                    reduction="sum",
                )
                
                # Update state: Mark the slots that HAVE satisfied the condition (target=1)
                # We only stop tracking a slot once it has successfully halted (improved).
                halt_update_mask = train_mask & (target_halt_per_l > 0.5)
                self._halt_condition_met = self._halt_condition_met | halt_update_mask
            else:
                q_halt_loss = torch.tensor(0.0, device=q_halt_logits.device)

            metrics.update({"q_halt_loss": q_halt_loss.detach()})
            total_loss = reg_loss + 0.5 * q_halt_loss
        else:
            total_loss = reg_loss

        # Simple metrics on preds vs targets (final)
        with torch.no_grad():
            diff = (preds - targets).abs()  # [B, L, F]
            sq_err = (preds - targets).pow(2)
            if mask is not None and preds.dim() == 3:
                # Apply mask per token and broadcast over feature dim
                m = mask.unsqueeze(-1)  # [B, L, 1]
                diff = diff * m
                sq_err = sq_err * m
                # Match reg_loss normalization: mean over B, L, F of valid positions
                denom = (mask.sum().clamp_min(1) * preds.size(-1)).to(dtype=torch.float32)
            else:
                # All positions valid: B * L * F
                denom = torch.tensor(preds.numel(), device=preds.device, dtype=torch.float32)
            metrics.update({
                "mse": (sq_err.sum() / denom),
                "mae": (diff.sum() / denom),
            })

        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}
        return new_carry, total_loss, metrics, detached_outputs, new_carry.halted.all()

