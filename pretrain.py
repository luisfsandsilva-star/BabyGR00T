from typing import Optional, Any, Sequence, List, Dict
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
) -> None:
    if metrics is None:
        return

    formatted = _format_metrics_string(metrics)
    message = f"{label}: {formatted}"

    if logger is not None:
        logger.info(message)
    if print_to_console:
        print(message)



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

    summary_pattern = re.compile(
        r"EPOCH_SUMMARY epoch=(?P<epoch>\d+) train_loss=(?P<train>[-+\deE\.NA]+) val_loss=(?P<val>[-+\deE\.NA]+)"
    )

    epochs: List[int] = []
    train_losses: List[Optional[float]] = []
    val_losses: List[Optional[float]] = []

    try:
        with log_path.open("r", encoding="utf-8") as log_file:
            for line in log_file:
                match = summary_pattern.search(line)
                if match is None:
                    continue

                epoch = int(match.group("epoch"))

                def _safe_parse(value: str) -> Optional[float]:
                    if value.upper() == "NA":
                        return None
                    try:
                        parsed = float(value)
                    except ValueError:
                        return None
                    if math.isnan(parsed) or math.isinf(parsed):
                        return None
                    return parsed

                epochs.append(epoch)
                train_losses.append(_safe_parse(match.group("train")))
                val_losses.append(_safe_parse(match.group("val")))
    except FileNotFoundError:
        return

    if not epochs:
        return

    has_train = any(value is not None for value in train_losses)
    has_val = any(value is not None for value in val_losses)

    if not has_train and not has_val:
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

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "train_vs_val_loss.png"

    def _to_series(values: List[Optional[float]]) -> List[float]:
        return [value if value is not None else math.nan for value in values]

    plt.figure(figsize=(8, 5))
    if has_train:
        plt.plot(epochs, _to_series(train_losses), label="train_loss", marker="o")
    if has_val:
        plt.plot(epochs, _to_series(val_losses), label="val_loss", marker="o")
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


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
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

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

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


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


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
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
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
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

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
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

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
        log_dir_path = Path(config.log_dir).expanduser()
        log_dir_path.mkdir(parents=True, exist_ok=True)
        config.log_dir = str(log_dir_path.resolve())
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
    try:
        eval_loader, eval_metadata = create_dataloader(
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
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar: Optional[tqdm.tqdm] = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
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
            print(info_message)
            if run_logger is not None:
                run_logger.info(info_message)
            print(f"Model parameters: {num_params}")

    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        epoch_num = _iter_id + 1
        global_epoch = _iter_id * train_epochs_per_iter
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {global_epoch}")

        epoch_train_losses: List[float] = []
        epoch_val_loss: Optional[float] = None

        if RANK == 0:
            if run_logger is not None:
                run_logger.info("EPOCH_START epoch=%d global_epoch=%d", epoch_num, global_epoch)
            if not config.use_wandb:
                print(f"EPOCH_START epoch={epoch_num} global_epoch={global_epoch}")
            if progress_bar is not None:
                progress_bar.set_description(f"Epoch {epoch_num}/{total_iters}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                if config.use_wandb:
                    wandb.log(metrics, step=train_state.step)
                else:
                    _log_metrics(run_logger, f"TRAIN step={train_state.step}", metrics, print_to_console=True)
                if progress_bar is not None:
                    progress_bar.update(train_state.step - progress_bar.n)
                train_loss_val = _maybe_to_number(metrics.get("train/loss")) if isinstance(metrics, dict) else None
                if train_loss_val is not None:
                    epoch_train_losses.append(train_loss_val)
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            metrics = None
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
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
                        _log_metrics(run_logger, f"VAL step={train_state.step}", metrics, print_to_console=True)
                    epoch_val_loss = _extract_validation_loss(metrics)
            else:
                if RANK == 0:
                    skip_msg = "Skipping evaluation because no evaluation loader is available."
                    if run_logger is not None:
                        run_logger.info(skip_msg)
                    if not config.use_wandb:
                        print(skip_msg)

            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

        if RANK == 0:
            train_loss_avg = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else None
            train_loss_str = f"{train_loss_avg:.6f}" if train_loss_avg is not None else "NA"
            val_loss_str = f"{epoch_val_loss:.6f}" if epoch_val_loss is not None else "NA"
            summary_message = f"EPOCH_SUMMARY epoch={epoch_num} train_loss={train_loss_str} val_loss={val_loss_str}"
            if run_logger is not None:
                run_logger.info(summary_message)
            if not config.use_wandb:
                print(summary_message)
            if progress_bar is not None:
                progress_bar.set_postfix_str(f"train_loss={train_loss_str} val_loss={val_loss_str}")

    if RANK == 0:
        if progress_bar is not None:
            progress_bar.close()
        if log_path is not None:
            generate_loss_plot(log_path, Path(config.log_dir), run_logger)

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    if config.use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    launch()
