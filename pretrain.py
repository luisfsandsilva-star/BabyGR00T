from typing import Optional, Any, Sequence, List, Dict, Tuple, Literal
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import inspect
import logging
import re
from pathlib import Path
from numbers import Number

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from lion_pytorch import Lion

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class CheckpointConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")

    path: Optional[str] = None
    resume: bool = False
    resume_from: Optional[str] = None
    monitor: str = "val/loss"
    mode: Literal["min", "max"] = "min"
    keep_last: Optional[int] = None


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint: CheckpointConfig = CheckpointConfig()

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings

    # Logging
    use_wandb: bool = True
    log_dir: str = "logs"
    log_file: str = "pretrain.log"
    skip_sanity_checks: bool = False

    @pydantic.model_validator(mode="after")
    def _sync_checkpoint_config(self) -> "PretrainConfig":
        if self.checkpoint.path is None and self.checkpoint_path is not None:
            object.__setattr__(self.checkpoint, "path", self.checkpoint_path)
        if self.checkpoint.resume_from is None and self.load_checkpoint is not None:
            object.__setattr__(self.checkpoint, "resume_from", self.load_checkpoint)
        return self



def setup_file_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("pretrain_run")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(file_handler)

    return logger



def _write_console_message(message: str, progress_bar: Optional[tqdm.tqdm] = None) -> None:
    if progress_bar is not None:
        tqdm.tqdm.write(message)
    else:
        print(message)


def _maybe_to_number(value: Any) -> Optional[float]:
    if isinstance(value, Number):
        return float(value)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return float(item())
        except Exception:
            return None

    return None



def _flatten_metrics(metrics: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in metrics.items():
        new_key = f"{parent_key}/{key}" if parent_key else str(key)
        if isinstance(value, dict):
            items.update(_flatten_metrics(value, new_key))
        else:
            items[new_key] = value
    return items



def _format_metrics_string(metrics: Dict[str, Any]) -> str:
    flat_metrics = _flatten_metrics(metrics)
    parts = []
    for key in sorted(flat_metrics.keys()):
        value = flat_metrics[key]
        numeric = _maybe_to_number(value)
        if numeric is None:
            parts.append(f"{key}={value}")
        else:
            if math.isnan(numeric) or math.isinf(numeric):
                parts.append(f"{key}={numeric}")
            else:
                parts.append(f"{key}={numeric:.6f}")
    return ", ".join(parts)



def _log_metrics(
    logger: Optional[logging.Logger],
    label: str,
    metrics: Optional[Dict[str, Any]],
    print_to_console: bool = False,
    progress_bar: Optional[tqdm.tqdm] = None,
) -> None:
    if metrics is None:
        return

    formatted = _format_metrics_string(metrics)
    message = f"{label}: {formatted}"

    if logger is not None:
        logger.info(message)
    if print_to_console:
        _write_console_message(message, progress_bar)



def _dataset_total_examples(loader: DataLoader) -> Optional[int]:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return None

    lazy_loader = getattr(dataset, "_lazy_load_dataset", None)
    if callable(lazy_loader):
        lazy_loader()

    data = getattr(dataset, "_data", None)
    if not isinstance(data, dict) or not data:
        return None

    total = 0
    for set_data in data.values():
        if isinstance(set_data, dict) and "inputs" in set_data:
            inputs = set_data["inputs"]
            try:
                total += int(len(inputs))
            except Exception:
                size = getattr(inputs, "shape", None)
                if isinstance(size, tuple) and size:
                    total += int(size[0])
    return total


def _validate_tensor(
    *,
    name: str,
    tensor: torch.Tensor,
    expected_dtype: Optional[torch.dtype] = None,
    expected_last_dim: Optional[int] = None,
) -> Dict[str, Any]:
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise TypeError(
            f"Expected {name} dtype {expected_dtype} but found {tensor.dtype}."
        )

    if expected_last_dim is not None and tensor.ndim >= 1:
        if tensor.shape[-1] != expected_last_dim:
            raise ValueError(
                f"Expected {name} last dimension {expected_last_dim} but found {tensor.shape[-1]}."
            )

    finite = True
    if tensor.is_floating_point():
        finite = bool(torch.isfinite(tensor).all().item())
        if not finite:
            raise ValueError(f"Tensor '{name}' contains NaN or inf values.")

    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
    }


def _run_dataloader_sanity_checks(
    *,
    loader: DataLoader,
    metadata: PuzzleDatasetMetadata,
    loader_name: str,
    rank: int,
    expected_batch_size: Optional[int],
) -> None:
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise RuntimeError(f"{loader_name} loader does not expose a dataset instance.")

    if metadata.total_groups <= 0:
        raise RuntimeError(
            f"{loader_name} metadata reports no groups (total_groups={metadata.total_groups})."
        )

    total_examples = _dataset_total_examples(loader)
    if total_examples is None or total_examples <= 0:
        raise RuntimeError(
            f"{loader_name} dataset does not contain any examples (total_examples={total_examples})."
        )

    data = getattr(dataset, "_data", None)
    if isinstance(data, dict) and metadata.sets:
        missing_sets = [set_name for set_name in metadata.sets if set_name not in data]
        if missing_sets:
            raise RuntimeError(
                f"{loader_name} dataset is missing splits listed in metadata: {missing_sets}."
            )

    iterator = iter(loader)
    try:
        set_name, batch, global_batch_size = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"{loader_name} loader returned no batches for sanity checks.") from exc

    if "inputs" not in batch:
        raise RuntimeError(f"{loader_name} batch is missing required 'inputs' tensor.")

    inputs = batch["inputs"]
    batch_size = inputs.shape[0]
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise RuntimeError(
            f"{loader_name} batch size mismatch: expected {expected_batch_size}, found {batch_size}."
        )

    expected_inputs_dtype = torch.float32 if metadata.task_type == "regression" else torch.int32
    summary: Dict[str, Any] = {
        "set_name": set_name,
        "global_batch_size": int(global_batch_size),
        "inputs": _validate_tensor(
            name="inputs",
            tensor=inputs,
            expected_dtype=expected_inputs_dtype,
            expected_last_dim=metadata.input_dim,
        ),
    }

    targets = batch.get("targets")
    if targets is not None:
        summary["targets"] = _validate_tensor(
            name="targets",
            tensor=targets,
            expected_dtype=torch.float32,
            expected_last_dim=metadata.target_dim,
        )
    else:
        if metadata.task_type == "regression":
            raise RuntimeError(f"{loader_name} batch is missing required 'targets' tensor.")
        summary["targets"] = "missing"

    target_mask = batch.get("target_mask")
    if target_mask is not None:
        summary["target_mask"] = _validate_tensor(
            name="target_mask",
            tensor=target_mask,
            expected_dtype=torch.bool,
            expected_last_dim=metadata.target_dim,
        )
        if targets is not None and target_mask.shape != targets.shape:
            raise ValueError(
                f"{loader_name} target_mask shape {tuple(target_mask.shape)} does not match targets {tuple(targets.shape)}."
            )
    else:
        if metadata.task_type == "regression":
            raise RuntimeError(f"{loader_name} batch is missing required 'target_mask' tensor.")
        summary["target_mask"] = "missing"

    if rank == 0:
        print(f"[Sanity Check] {loader_name} loader preview:")
        print(f"  set='{summary['set_name']}', global_batch_size={summary['global_batch_size']}")

        def _fmt_entry(value: Any) -> str:
            if isinstance(value, dict):
                finite = value.get("finite")
                finite_str = "finite" if finite else "non-finite"
                return (
                    f"shape={value.get('shape')}, dtype={value.get('dtype')}, {finite_str}"
                )
            return str(value)

        for key in ("inputs", "targets", "target_mask"):
            print(f"  {key}: {_fmt_entry(summary[key])}")


def _extract_validation_loss(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if metrics is None:
        return None

    flat_metrics = _flatten_metrics(metrics)
    losses = []
    for key, value in flat_metrics.items():
        metric_name = key.split("/")[-1]
        if metric_name.endswith("loss") and not key.startswith("train/"):
            numeric = _maybe_to_number(value)
            if numeric is not None:
                losses.append(numeric)

    if losses:
        return sum(losses) / len(losses)
    return None



def generate_loss_plot(log_path: Path, output_dir: Path, logger: Optional[logging.Logger] = None) -> None:

    if not log_path.exists():
        return

    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - pandas might be unavailable in tests
        message = f"Unable to load pandas for loss plot: {exc}"
        if logger is not None:
            logger.warning(message)
        else:
            print(message)
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib might be unavailable in tests
        message = f"Unable to create loss plot: {exc}"
        if logger is not None:
            logger.warning(message)
        else:
            print(message)
        return

    summary_pattern = re.compile(
        r"EPOCH_SUMMARY epoch=(?P<epoch>\d+) train_loss=(?P<train>[-+\deE\.NA]+) val_loss=(?P<val>[-+\deE\.NA]+)"
    )

    records: List[Dict[str, Any]] = []
    try:
        with log_path.open("r", encoding="utf-8") as log_file:
            for line in log_file:
                match = summary_pattern.search(line)
                if match is None:
                    continue

                def _safe_value(key: str) -> Optional[str]:
                    value = match.group(key)
                    if value.upper() == "NA":
                        return None
                    return value

                records.append(
                    {
                        "epoch": match.group("epoch"),
                        "train_loss": _safe_value("train"),
                        "val_loss": _safe_value("val"),
                    }
                )
    except FileNotFoundError:
        return

    if not records:
        return

    df = pd.DataFrame.from_records(records)
    if df.empty:
        return

    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df = df.dropna(subset=["epoch"])
    if df.empty:
        return

    df["epoch"] = df["epoch"].astype(int)
    for column in ("train_loss", "val_loss"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df.sort_values("epoch", inplace=True)
    df = df.drop_duplicates(subset=["epoch"], keep="last")

    has_train = "train_loss" in df.columns and df["train_loss"].notna().any()
    has_val = "val_loss" in df.columns and df["val_loss"].notna().any()

    if not has_train and not has_val:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "train_vs_val_loss.png"

    plt.figure(figsize=(8, 5))
    if has_train:
        plt.plot(df["epoch"], df["train_loss"], label="train_loss", marker="o")
    if has_val:
        plt.plot(df["epoch"], df["val_loss"], label="val_loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

    if logger is not None:
        logger.info("LOSS_PLOT_SAVED path=%s", figure_path)
    else:
        print(f"Saved loss plot to {figure_path}")




@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    epoch: int = 0
    best_val_loss: Optional[float] = None
    schedulers: Sequence[Any] = ()
    scheduler_states: Optional[Sequence[Dict[str, Any]]] = None
    ema_state: Optional[Dict[str, Any]] = None


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    *,
    model_state: Optional[Dict[str, Any]] = None,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=False  # Non-autoregressive
    )

    if train_metadata.input_dim is not None:
        model_cfg["latent_dim"] = train_metadata.input_dim
    if train_metadata.target_dim is not None:
        model_cfg["output_dim"] = train_metadata.target_dim

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    loss_kwargs = dict(getattr(config.arch.loss, "__pydantic_extra__", {}) or {})

    try:
        loss_init_sig = inspect.signature(loss_head_cls.__init__)
    except (TypeError, ValueError):
        pass
    else:
        accepts_var_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in loss_init_sig.parameters.values()
        )
        if not accepts_var_kwargs:
            allowed_params = {
                name
                for name, param in loss_init_sig.parameters.items()
                if name != "self" and param.kind != inspect.Parameter.VAR_POSITIONAL
            }
            loss_kwargs = {k: v for k, v in loss_kwargs.items() if k in allowed_params}

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **loss_kwargs)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load model weights if provided
        if model_state is not None and rank == 0:
            print("Loading model weights from checkpoint")
            puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
            expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
            if puzzle_emb_name in model_state:
                puzzle_emb = model_state[puzzle_emb_name]
                if puzzle_emb.shape != expected_shape:
                    print(
                        "Resetting puzzle embedding as shape is different. "
                        f"Found {puzzle_emb.shape}, Expected {expected_shape}"
                    )
                    model_state[puzzle_emb_name] = (
                        torch.mean(puzzle_emb, dim=0, keepdim=True)
                        .expand(expected_shape)
                        .contiguous()
                    )
            model.load_state_dict(model_state, assign=True)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            Lion(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            Lion(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(
    config: PretrainConfig,
    train_metadata: PuzzleDatasetMetadata,
    rank: int,
    world_size: int,
    *,
    model_state: Optional[Dict[str, Any]] = None,
    optimizer_states: Optional[Sequence[Dict[str, Any]]] = None,
    scheduler_states: Optional[Sequence[Dict[str, Any]]] = None,
    initial_step: int = 0,
    initial_epoch: int = 0,
    best_val_loss: Optional[float] = None,
    ema_state: Optional[Dict[str, Any]] = None,
):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(
        config,
        train_metadata,
        rank=rank,
        world_size=world_size,
        model_state=model_state,
    )

    if optimizer_states is not None:
        for optimizer, state in zip(optimizers, optimizer_states):
            optimizer.load_state_dict(state)

    return TrainState(
        step=initial_step,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
        epoch=initial_epoch,
        best_val_loss=best_val_loss,
        scheduler_states=scheduler_states,
        ema_state=ema_state,
    )


def _get_checkpoint_dir(config: PretrainConfig) -> Optional[Path]:
    checkpoint_dir = config.checkpoint.path or config.checkpoint_path
    if checkpoint_dir is None:
        return None
    return Path(checkpoint_dir)


def _format_checkpoint_name(step: int) -> str:
    return f"step_{step:08d}.ckpt"


def _collect_scheduler_states(train_state: TrainState) -> Optional[List[Dict[str, Any]]]:
    collected: List[Dict[str, Any]] = []
    for scheduler in getattr(train_state, "schedulers", ()) or ():
        state_dict = getattr(scheduler, "state_dict", None)
        if callable(state_dict):
            collected.append(state_dict())
    if collected:
        return collected
    if train_state.scheduler_states is not None:
        return list(train_state.scheduler_states)
    return None


def _get_monitor_value(metrics: Optional[Dict[str, Any]], monitor_key: Optional[str]) -> Optional[float]:
    if metrics is None or monitor_key is None:
        return None

    flat_metrics = _flatten_metrics(metrics)
    if monitor_key not in flat_metrics:
        return None

    return _maybe_to_number(flat_metrics[monitor_key])


def _is_improvement(mode: str, current: Optional[float], best: Optional[float]) -> bool:
    if current is None:
        return False

    if math.isnan(current):
        return False

    if best is None or math.isnan(best):
        return True

    normalized_mode = mode.lower()
    if normalized_mode == "max":
        return current > best

    return current < best


def save_train_state(
    config: PretrainConfig,
    train_state: TrainState,
    *,
    monitor_value: Optional[float] = None,
    is_best: bool = False,
    ema_state: Optional[Dict[str, Any]] = None,
):
    checkpoint_dir = _get_checkpoint_dir(config)
    if checkpoint_dir is None:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer_states = [optimizer.state_dict() for optimizer in train_state.optimizers]
    scheduler_states = _collect_scheduler_states(train_state)
    ema_payload = ema_state if ema_state is not None else train_state.ema_state

    checkpoint_payload = {
        "model_state": train_state.model.state_dict(),
        "optimizer_states": optimizer_states,
        "scheduler_state": scheduler_states,
        "step": train_state.step,
        "epoch": train_state.epoch,
        "best_val_loss": train_state.best_val_loss,
        "monitor": config.checkpoint.monitor,
        "monitor_value": monitor_value,
        "ema_state": ema_payload,
    }

    step_filename = _format_checkpoint_name(train_state.step)
    step_path = checkpoint_dir / step_filename
    torch.save(checkpoint_payload, step_path)

    # Track latest checkpoint for easy resume
    latest_path = checkpoint_dir / "latest.txt"
    latest_path.write_text(step_filename)

    if is_best:
        shutil.copyfile(step_path, checkpoint_dir / "best.ckpt")

    # Maintain cascading checkpoints
    keep_last = config.checkpoint.keep_last
    if keep_last is not None and keep_last >= 0:
        limit = max(keep_last, 1)
        numbered = sorted(checkpoint_dir.glob("step_*.ckpt"))
        if len(numbered) > limit:
            remove = numbered[:-limit]
        else:
            remove = []
        for old_path in remove:
            old_path.unlink(missing_ok=True)

    train_state.scheduler_states = scheduler_states
    train_state.ema_state = ema_payload


def _resolve_checkpoint_path(config: PretrainConfig) -> Optional[Path]:
    checkpoint_dir = _get_checkpoint_dir(config)
    checkpoint_cfg = config.checkpoint

    if checkpoint_cfg.resume_from is not None:
        explicit_path = Path(checkpoint_cfg.resume_from)
        if not explicit_path.is_absolute() and checkpoint_dir is not None:
            explicit_path = checkpoint_dir / explicit_path
        return explicit_path

    if not checkpoint_cfg.resume:
        return None

    if checkpoint_dir is None:
        return None

    latest_file = checkpoint_dir / "latest.txt"
    if latest_file.exists():
        relative = latest_file.read_text().strip()
        candidate = checkpoint_dir / relative
        if candidate.exists():
            return candidate

    numbered = sorted(checkpoint_dir.glob("step_*.ckpt"))
    if numbered:
        return numbered[-1]

    best_path = checkpoint_dir / "best.ckpt"
    if best_path.exists():
        return best_path

    return None


def load_checkpoint(config: PretrainConfig) -> Optional[Dict[str, Any]]:
    checkpoint_path = _resolve_checkpoint_path(config)
    if checkpoint_path is None:
        return None

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")

    print(f"Loading checkpoint {checkpoint_path}")
    map_location = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    raw_checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(raw_checkpoint, dict) and "model_state" in raw_checkpoint:
        checkpoint_payload = dict(raw_checkpoint)
    else:
        checkpoint_payload = {
            "model_state": raw_checkpoint,
            "optimizer_states": None,
            "scheduler_state": None,
            "step": 0,
            "epoch": 0,
            "best_val_loss": None,
            "monitor": None,
            "monitor_value": None,
            "ema_state": None,
        }

    checkpoint_payload["checkpoint_path"] = str(checkpoint_path)
    return checkpoint_payload


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

def _resolve_eval_set_names(eval_loader: DataLoader, eval_metadata: PuzzleDatasetMetadata) -> List[str]:
    """Return the ordered list of set names yielded by ``eval_loader``.

    When multiple dataset paths share the same base set name the loader appends
    an integer suffix (e.g. ``episodes1``) to keep them distinct.  The metadata
    still only tracks the base names, which causes evaluation to fail when it
    looks up metrics by the suffixed key.  We inspect the loader to retrieve the
    actual keys populated in ``PuzzleDataset._data`` and fall back to the
    metadata when that information is unavailable (e.g. for custom loaders).
    """

    dataset = getattr(eval_loader, "dataset", None)
    if dataset is not None:
        lazy_loader = getattr(dataset, "_lazy_load_dataset", None)
        if callable(lazy_loader):
            lazy_loader()
        data = getattr(dataset, "_data", None)
        if isinstance(data, dict) and len(data):
            return list(data.keys())

    return list(eval_metadata.sets)


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_names = _resolve_eval_set_names(eval_loader, eval_metadata)
        set_ids = {k: idx for idx, k in enumerate(set_names)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_names), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        checkpoint_dir = _get_checkpoint_dir(config)
        if checkpoint_dir is not None and len(save_preds):
            # Each rank save predictions independently
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                save_preds, checkpoint_dir / f"step_{train_state.step}_all_preds.{rank}"
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_names)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if checkpoint_dir is not None:
                evaluator_save_path = checkpoint_dir / f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}"
                evaluator_save_path.mkdir(exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    checkpoint_dir = _get_checkpoint_dir(config)
    if checkpoint_dir is None or wandb.run is None:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, checkpoint_dir / code_name)

    # Dump config as yaml
    config_file = checkpoint_dir / "all_config.yaml"
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(str(checkpoint_dir))


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        checkpoint_dir = config.checkpoint.path or config.checkpoint_path
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join("checkpoints", config.project_name, config.run_name)
        config.checkpoint_path = checkpoint_dir
        object.__setattr__(config.checkpoint, "path", checkpoint_dir)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore



@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    log_dir_path: Optional[Path] = None
    log_path: Optional[Path] = None
    run_logger: Optional[logging.Logger] = None
    if RANK == 0:
        resolved_dir = Path(config.log_dir).expanduser()
        config.log_dir = str(resolved_dir.resolve())
        if not config.use_wandb:
            resolved_dir.mkdir(parents=True, exist_ok=True)
            log_dir_path = resolved_dir
            log_path = log_dir_path / config.log_file
            run_logger = setup_file_logger(log_path)
            run_logger.info("LOGGING_INITIALIZED log_path=%s", log_path)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=train_epochs_per_iter,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    if not config.skip_sanity_checks:
        train_dataset = getattr(train_loader, "dataset", None)
        expected_local_batch = getattr(train_dataset, "local_batch_size", None)
        _run_dataloader_sanity_checks(
            loader=train_loader,
            metadata=train_metadata,
            loader_name="train",
            rank=RANK,
            expected_batch_size=expected_local_batch,
        )
        train_loader, train_metadata = create_dataloader(
            config,
            "train",
            test_set_mode=False,
            epochs_per_iter=train_epochs_per_iter,
            global_batch_size=config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
        )

    eval_loader = None
    eval_metadata = None
    try:
        potential_eval_loader, potential_eval_metadata = create_dataloader(
            config,
            "test",
            test_set_mode=True,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
            rank=RANK,
            world_size=WORLD_SIZE,
        )
    except Exception:
        print("NO EVAL DATA FOUND")
    else:
        if not config.skip_sanity_checks:
            eval_dataset = getattr(potential_eval_loader, "dataset", None)
            expected_eval_batch = getattr(eval_dataset, "local_batch_size", None)
            _run_dataloader_sanity_checks(
                loader=potential_eval_loader,
                metadata=potential_eval_metadata,
                loader_name="eval",
                rank=RANK,
                expected_batch_size=expected_eval_batch,
            )
            potential_eval_loader, potential_eval_metadata = create_dataloader(
                config,
                "test",
                test_set_mode=True,
                epochs_per_iter=1,
                global_batch_size=config.global_batch_size,
                rank=RANK,
                world_size=WORLD_SIZE,
            )

        eval_loader = potential_eval_loader
        eval_metadata = potential_eval_metadata

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        print("No evaluator found")
        evaluators = []

    # Train state
    checkpoint_payload = None
    if RANK == 0:
        checkpoint_payload = load_checkpoint(config)

    if WORLD_SIZE > 1:
        shared_payload: List[Any] = [checkpoint_payload]
        dist.broadcast_object_list(shared_payload, src=0)
        checkpoint_payload = shared_payload[0]

    model_state = None
    optimizer_states = None
    scheduler_states = None
    initial_step = 0
    initial_epoch = 0
    best_val_loss = None
    ema_state = None

    if checkpoint_payload is not None:
        model_state = checkpoint_payload.get("model_state")
        if config.checkpoint.resume:
            optimizer_states = checkpoint_payload.get("optimizer_states")
            scheduler_states = checkpoint_payload.get("scheduler_state")
            initial_step = int(checkpoint_payload.get("step", 0) or 0)
            initial_epoch = int(checkpoint_payload.get("epoch", 0) or 0)
            best_val_loss = checkpoint_payload.get("best_val_loss")
            ema_state = checkpoint_payload.get("ema_state")

    train_state = init_train_state(
        config,
        train_metadata,
        rank=RANK,
        world_size=WORLD_SIZE,
        model_state=model_state,
        optimizer_states=optimizer_states,
        scheduler_states=scheduler_states,
        initial_step=initial_step,
        initial_epoch=initial_epoch,
        best_val_loss=best_val_loss,
        ema_state=ema_state,
    )

    # Progress bar and logger
    progress_bar: Optional[tqdm.tqdm] = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)
        num_params = sum(x.numel() for x in train_state.model.parameters())
        if run_logger is not None:
            run_logger.info("MODEL num_params=%d", num_params)
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.model_dump(),
                settings=wandb.Settings(_disable_stats=True),
            )  # type: ignore
            wandb.log({"num_params": num_params}, step=0)
            save_code_and_config(config)
        else:
            info_message = f"W&B disabled. Logging metrics to {log_path}" if log_path is not None else "W&B disabled."
            _write_console_message(info_message, progress_bar)
            if run_logger is not None:
                run_logger.info(info_message)
            _write_console_message(f"Model parameters: {num_params}", progress_bar)

    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)
        if train_state.ema_state is not None:
            ema_helper.load_state_dict(train_state.ema_state)

    # Training Loop
    start_iter = 0
    if train_epochs_per_iter > 0:
        start_iter = train_state.epoch // train_epochs_per_iter

    if start_iter:
        resumed_from = checkpoint_payload.get("checkpoint_path") if checkpoint_payload else None
        resume_message = (
            f"Resuming training from step {train_state.step} epoch {train_state.epoch}"
            + (f" using {resumed_from}" if resumed_from else "")
        )
        if RANK == 0:
            if run_logger is not None:
                run_logger.info(resume_message)
            if not config.use_wandb:
                _write_console_message(resume_message, progress_bar)

    for _iter_id in range(start_iter, total_iters):
        epoch_num = _iter_id + 1
        global_epoch = _iter_id * train_epochs_per_iter
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {global_epoch}")

        epoch_train_losses: List[float] = []
        epoch_val_loss: Optional[float] = None

        if RANK == 0:
            if run_logger is not None:
                run_logger.info("EPOCH_START epoch=%d global_epoch=%d", epoch_num, global_epoch)
            if not config.use_wandb:
                _write_console_message(f"EPOCH_START epoch={epoch_num} global_epoch={global_epoch}", progress_bar)
            if progress_bar is not None:
                progress_bar.set_description(f"Epoch {epoch_num}/{total_iters}")

        ############ Train Iter
        if RANK == 0:
            _write_console_message("TRAIN", progress_bar)
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                if config.use_wandb:
                    wandb.log(metrics, step=train_state.step)
                else:
                    _log_metrics(run_logger, f"TRAIN step={train_state.step}", metrics, print_to_console=True, progress_bar=progress_bar)
                if progress_bar is not None:
                    progress_bar.update(train_state.step - progress_bar.n)
                train_loss_val = _maybe_to_number(metrics.get("train/loss")) if isinstance(metrics, dict) else None
                if train_loss_val is not None:
                    epoch_train_losses.append(train_loss_val)
            if config.ema:
                ema_helper.update(train_state.model)

        monitor_value: Optional[float] = None

        if _iter_id >= config.min_eval_interval:
            metrics = None
            if RANK == 0:
                _write_console_message("EVALUATE", progress_bar)
            if config.ema:
                _write_console_message("SWITCH TO EMA", progress_bar)
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()

            if eval_loader is not None and eval_metadata is not None:
                metrics = evaluate(
                    config,
                    train_state_eval,
                    eval_loader,
                    eval_metadata,
                    evaluators,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                    cpu_group=CPU_PROCESS_GROUP,
                )
                if RANK == 0 and metrics is not None:
                    if config.use_wandb:
                        wandb.log(metrics, step=train_state.step)
                    else:
                        _log_metrics(run_logger, f"VAL step={train_state.step}", metrics, print_to_console=True, progress_bar=progress_bar)
                    epoch_val_loss = _extract_validation_loss(metrics)
                monitor_value = _get_monitor_value(metrics, config.checkpoint.monitor)
            else:
                if RANK == 0:
                    skip_msg = "Skipping evaluation because no evaluation loader is available."
                    if run_logger is not None:
                        run_logger.info(skip_msg)
                    if not config.use_wandb:
                        _write_console_message(skip_msg, progress_bar)

            ############ Checkpointing
            if RANK == 0:
                _write_console_message("SAVE CHECKPOINT", progress_bar)
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                ema_state = ema_helper.state_dict() if config.ema else None
                improved = False
                if config.checkpoint.monitor is not None:
                    if _is_improvement(config.checkpoint.mode, monitor_value, train_state.best_val_loss):
                        train_state.best_val_loss = monitor_value
                        improved = monitor_value is not None
                save_train_state(
                    config,
                    train_state,
                    monitor_value=monitor_value,
                    is_best=improved,
                    ema_state=ema_state,
                )

            if config.ema:
                del train_state_eval

        # Update the current epoch counter after each iteration
        train_state.epoch = (_iter_id + 1) * train_epochs_per_iter

        if RANK == 0:
            train_loss_avg = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else None
            train_loss_str = f"{train_loss_avg:.6f}" if train_loss_avg is not None else "NA"
            val_loss_str = f"{epoch_val_loss:.6f}" if epoch_val_loss is not None else "NA"
            summary_message = f"EPOCH_SUMMARY epoch={epoch_num} train_loss={train_loss_str} val_loss={val_loss_str}"
            if run_logger is not None:
                run_logger.info(summary_message)
            if not config.use_wandb:
                _write_console_message(summary_message, progress_bar)
            if progress_bar is not None:
                progress_bar.set_postfix_str(f"train_loss={train_loss_str} val_loss={val_loss_str}")

    if RANK == 0:
        if progress_bar is not None:
            progress_bar.close()
        if log_path is not None:
            checkpoint_dir = _get_checkpoint_dir(config)
            output_dir = checkpoint_dir if checkpoint_dir is not None else Path(config.log_dir)
            generate_loss_plot(log_path, output_dir, run_logger)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    if config.use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    launch()
