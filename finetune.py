from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import json
import getpass

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import coolname
import hydra
import pydantic
from omegaconf import DictConfig

try:
    import wandb  # type: ignore
except Exception:  # noqa: BLE001
    wandb = None  # type: ignore

# Use Lion optimizer (prefer official, fallback to local)
try:
    from lion_pytorch import Lion as _OptimCls  # type: ignore
except Exception:  # noqa: BLE001
    # Fallback to local lightweight Lion
    from utils.lion import Lion as _OptimCls  # type: ignore

from dataset.common import PuzzleDatasetMetadata
from dataset.latent_npz_dataset import LatentNPZDataset, LatentNPZDatasetConfig
from utils.functions import load_model_class, get_model_source_path
from models.ema import EMAHelper
from models.layers import ScaleNorm

# Optional fallback dataset (only used if no .npz files found)
try:
    from dataset.regression_dataset import RegressionDataset, RegressionDatasetConfig
except ImportError:
    RegressionDataset = None  # type: ignore
    RegressionDatasetConfig = None  # type: ignore


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

    # Names / logging
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None
    use_wandb: bool = False  # If True, log to Weights & Biases in addition to terminal
    log_interval: int = 10   # How often (in steps) to print train metrics to terminal

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    checkpoint_interval_steps: Optional[int] = 1000  # periodic step-based checkpointing (None -> disable)
    # Evaluation scheduling (in training steps)
    eval_interval: Optional[int] = None  # if set, run eval every N train steps
    min_eval_interval: Optional[int] = 0 # minimum step at which to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    # Dataset strictness
    require_latents: bool = False  # If true, error if no .npz latents are found when task=regression
    
    # VLM context directories (optional)
    vlm_context_dirs: List[str] = []  # Explicit list of VLM context directories
    vlm_context_root: Optional[str] = None  # Auto-discover VLM context under this root

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def generate_plots_from_metrics(checkpoint_path: str):
    """
    Read the train_metrics.jsonl file and automatically generate PNG plots
    for all scalar metrics (loss, lr, etc.) as a function of step.
    """
    metrics_path = os.path.join(checkpoint_path, "train_metrics.jsonl")
    if not os.path.isfile(metrics_path):
        print(f"[Plots] No train_metrics.jsonl found under {checkpoint_path}; skipping plot generation.")
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[Plots] matplotlib is not available; cannot generate plots automatically: {e}")
        return

    steps: List[int] = []
    series: dict[str, List[float]] = {}

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                step = rec.get("step")
                if step is None:
                    continue
                try:
                    step_int = int(step)
                except Exception:
                    continue

                steps.append(step_int)
                for k, v in rec.items():
                    if k == "step":
                        continue
                    # Try to coerce to float; skip non-scalars
                    try:
                        val = float(v)
                    except Exception:
                        continue
                    series.setdefault(k, []).append(val)
    except Exception as e:  # noqa: BLE001
        print(f"[Plots] Failed to read metrics from {metrics_path}: {e}")
        return

    if not steps or not series:
        print(f"[Plots] No numeric metrics found in {metrics_path}; skipping plot generation.")
        return

    os.makedirs(checkpoint_path, exist_ok=True)

    for name, values in series.items():
        if len(values) != len(steps):
            # Skip inconsistent series lengths to avoid confusing plots
            continue
        try:
            plt.figure()
            plt.plot(steps, values, label=name)
            plt.xlabel("step")
            plt.ylabel(name)
            plt.title(name)
            plt.grid(True, alpha=0.3)
            plt.legend()
            safe_name = name.replace("/", "_").replace(" ", "_")
            out_path = os.path.join(checkpoint_path, f"{safe_name}.png")
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            plt.close()
            print(f"[Plots] Saved plot for '{name}' to {out_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[Plots] Failed to generate plot for {name}: {e}")


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    # Choose dataset based on task
    task = (config.arch.__pydantic_extra__ or {}).get("task", "language_modeling")
    if task == "regression":
        selected_paths = config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths
        # Auto-detect .npz latents
        has_npz = False
        try:
            for p in selected_paths:
                for _root, _dirs, files in os.walk(p):
                    if any(f.endswith('.npz') for f in files):
                        has_npz = True
                        break
                if has_npz:
                    break
        except Exception:
            has_npz = False

        if has_npz:
            # VLM context directories from config or environment
            vlm_dirs: List[str] = []
            
            # Option 1: Explicit directories from config
            if config.vlm_context_dirs:
                for pth in config.vlm_context_dirs:
                    if os.path.isdir(pth):
                        vlm_dirs.append(pth)
                        print(f"[Data] Using VLM context dir from config: {pth}")
            
            # Option 2: Auto-discover under root directory
            elif config.vlm_context_root and os.path.isdir(config.vlm_context_root):
                root = config.vlm_context_root
                print(f"[Data] Auto-discovering VLM context dirs under: {root}")
                for dirpath, _dirnames, filenames in os.walk(root):
                    if any(f.endswith('.npz') for f in filenames):
                        vlm_dirs.append(dirpath)
            
            # Option 3: Environment variable fallback
            elif 'VLM_CONTEXT_ROOT' in os.environ:
                root = os.environ['VLM_CONTEXT_ROOT']
                if os.path.isdir(root):
                    print(f"[Data] Auto-discovering VLM context dirs from $VLM_CONTEXT_ROOT: {root}")
                    for dirpath, _dirnames, filenames in os.walk(root):
                        if any(f.endswith('.npz') for f in filenames):
                            vlm_dirs.append(dirpath)
            
            if len(vlm_dirs):
                print(f"[Data] Using {len(vlm_dirs)} VLM context dir(s): {vlm_dirs[:3]}{'...' if len(vlm_dirs) > 3 else ''}")
            dataset = LatentNPZDataset(LatentNPZDatasetConfig(
                seed=config.seed,
                dataset_paths=selected_paths,
                global_batch_size=kwargs["global_batch_size"],
                test_set_mode=kwargs.get("test_set_mode", False),
                epochs_per_iter=kwargs.get("epochs_per_iter", 1),
                rank=rank,
                num_replicas=world_size,
                time_offset=1,
                vlm_context_dirs=vlm_dirs if len(vlm_dirs) else None,
                context_frame_period=12,
            ), split=split)
            # Ensure metadata is available before returning
            try:
                dataset._discover_files()  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                num_eps = len(dataset._episode_files)  # type: ignore[attr-defined]
            except Exception:
                num_eps = 0
            print(f"[Data] Using LatentNPZDataset: found {num_eps} episode(s) under: {selected_paths}")
        else:
            if config.require_latents:
                raise RuntimeError(f"[Data] require_latents=True but no .npz episodes found under: {selected_paths}")
            if RegressionDataset is None:
                raise RuntimeError(f"[Data] No .npz files found and RegressionDataset not available (fallback dataset not imported)")
            dataset = RegressionDataset(RegressionDatasetConfig(
                seed=config.seed,
                dataset_paths=selected_paths,
                global_batch_size=kwargs["global_batch_size"],
                test_set_mode=kwargs.get("test_set_mode", False),
                epochs_per_iter=kwargs.get("epochs_per_iter", 1),
                rank=rank,
                num_replicas=world_size,
            ), split=split)
            # Ensure metadata is available before returning
            try:
                dataset._lazy_load_dataset()  # type: ignore[attr-defined]
            except Exception:
                pass
            print(f"[Data] No .npz episodes found; using RegressionDataset fallback for: {selected_paths}")
    else:
        raise RuntimeError(f"[Data] Unsupported task: {task}. Only 'regression' task is supported.")
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

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        # Print parameter count for the base model
        num_params_base = sum(p.numel() for p in model.parameters())
        print(f"[Model] Base parameters: {num_params_base:,}")

        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        # Print total parameter count including loss head
        num_params_total = sum(p.numel() for p in model.parameters())
        print(f"[Model] Total parameters (with loss head): {num_params_total:,}")

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

    # Build parameter groups: do not apply weight decay to ScaleNorm.g parameters
    scale_norm_param_ids = set()
    for module in model.modules():
        if isinstance(module, ScaleNorm):
            for p in module.parameters(recurse=False):
                scale_norm_param_ids.add(id(p))

    def build_param_groups(main_lr_weight_decay: float):
        decay_params = []
        no_decay_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in scale_norm_param_ids:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = []
        if decay_params:
            param_groups.append(
                {
                    "params": decay_params,
                    "weight_decay": main_lr_weight_decay,
                }
            )
        if no_decay_params:
            param_groups.append(
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                }
            )
        return param_groups

    # Optimizer (Lion)
    optimizers = [
        _OptimCls(
            build_param_groups(config.weight_decay),
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )
    ]
    optimizer_lrs = [config.lr]

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
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)
    checkpoint = {
        "model": train_state.model.state_dict(),
        "optimizers": [opt.state_dict() for opt in train_state.optimizers],
        "optimizer_lrs": list(train_state.optimizer_lrs),
        "step": int(train_state.step),
        "total_steps": int(train_state.total_steps),
    }
    torch.save(checkpoint, os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load raw object (can be either a state_dict or a full training checkpoint)
        loaded_obj = torch.load(config.load_checkpoint, map_location="cuda")

        # If this is a full training checkpoint produced by save_train_state, extract the model state_dict.
        if isinstance(loaded_obj, dict) and "model" in loaded_obj and isinstance(loaded_obj["model"], dict):
            state_dict = loaded_obj["model"]
        else:
            # Backwards-compatible: assume the file directly contains the model state_dict.
            state_dict = loaded_obj

        model.load_state_dict(state_dict, assign=True)


def load_train_state_if_available(config: PretrainConfig, train_state: TrainState) -> TrainState:
    """
    Optionally restore full training state (model + optimizers + step) from config.load_checkpoint
    if the checkpoint was produced by save_train_state. If the file only contains raw model
    weights, this is a no-op (model weights were already loaded in create_model via load_checkpoint).
    """
    if config.load_checkpoint is None:
        return train_state

    ckpt_path = config.load_checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location="cuda")
    except Exception as e:  # noqa: BLE001
        print(f"[Checkpoint] Failed to load full train state from {ckpt_path}: {e}")
        return train_state

    # Expect a dict with at least model + optimizers to consider it a full training checkpoint.
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "optimizers" not in checkpoint:
        # Probably a raw state_dict file – nothing to do here.
        return train_state

    print(f"[Checkpoint] Restoring full training state from {ckpt_path}")

    # Restore model weights
    model_state = checkpoint["model"]
    train_state.model.load_state_dict(model_state, assign=True)

    # Restore optimizers
    opt_states = checkpoint.get("optimizers", [])
    for opt, opt_state in zip(train_state.optimizers, opt_states):
        opt.load_state_dict(opt_state)

    # Restore step (do NOT override total_steps; that is recomputed from current config)
    if "step" in checkpoint:
        train_state.step = int(checkpoint["step"])

    return train_state


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    """
    Linear warmup + linear decay schedule (in *steps*), independent of base_lr:
      - Warmup: from 5e-06 to 1e-03 over the first 1% of total_steps.
      - Decay:  from 1e-03 back down to 5e-06 over the remaining 99%.
    """
    # Fixed endpoints requested by user
    lr_start = 5e-6
    lr_peak = 1e-3
    lr_end = 5e-6

    total_steps = max(int(train_state.total_steps), 1)
    warmup_steps = max(int(0.01 * total_steps), 1)
    decay_steps = max(total_steps - warmup_steps, 1)

    # Clamp step to [1, total_steps] for scheduling purposes.
    step = int(train_state.step)
    if step < 1:
        step = 1
    if step > total_steps:
        step = total_steps

    if warmup_steps == 1:
        # Degenerate warmup: jump to peak on first step.
        if step == 1:
            return lr_peak
    if step <= warmup_steps:
        # Linear warmup: lr_start -> lr_peak
        # step=1 -> lr_start, step=warmup_steps -> lr_peak
        denom = max(warmup_steps - 1, 1)
        t = (step - 1) / denom
        return lr_start + t * (lr_peak - lr_start)

    # Linear decay: lr_peak -> lr_end over [warmup_steps+1, total_steps]
    if decay_steps <= 1:
        return lr_end
    t = (step - warmup_steps) / decay_steps
    return lr_peak + t * (lr_end - lr_peak)



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
    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init / reset carry if needed (e.g., batch size changed)
    batch_size = batch["inputs"].shape[0]
    if (
        train_state.carry is None
        or train_state.carry.halted.shape[0] != batch_size  # type: ignore[union-attr]
    ):
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(train_state.model.parameters(), max_norm=3.0)

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
    eval_pbar = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        if rank == 0:
            # Evaluation progress bar (batches)
            eval_pbar = tqdm.tqdm(desc="Eval", unit="batch", dynamic_ncols=True, leave=True, disable=False)

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
                if eval_pbar is not None:
                    eval_pbar.update(1)
            
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
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
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
                    for set_id, set_name in enumerate(set_ids)
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
            if eval_pbar is not None:
                eval_pbar.close()
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if (
        config.checkpoint_path is None
        or not config.use_wandb
        or wandb is None
        or wandb.run is None
    ):
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


def _discover_checkpoint_root() -> str:
    """
    Heuristic to choose a default checkpoint root:
      1) Respect CHECKPOINT_ROOT env var if set.
      2) Otherwise, look for writable mounts under /media/$USER.
      3) Fallback to a local 'checkpoints' directory in the current workspace.
    """
    # 1) Explicit override
    env_root = os.environ.get("CHECKPOINT_ROOT")
    if env_root:
        return env_root

    # 2) Try to auto-detect an external disk under /media/$USER
    try:
        user = getpass.getuser()
        media_root = os.path.join("/media", user)
        if os.path.isdir(media_root):
            candidates: List[str] = []
            for name in sorted(os.listdir(media_root)):
                path = os.path.join(media_root, name)
                # Prefer mounted, writable paths
                if os.path.ismount(path) and os.access(path, os.W_OK):
                    candidates.append(path)
            if candidates:
                # Pick the first stable candidate
                return candidates[0]
    except Exception:
        pass

    # 3) Fallback: local relative path
    return os.path.join(os.getcwd(), "checkpoints")


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"

        # Save checkpoints to outputs directory (mounted from host)
        base_ckpt_root = "/workspace/outputs"
        # Preserve per-project/run substructure under outputs
        config.checkpoint_path = os.path.join(
            base_ckpt_root,
            config.project_name,
            config.run_name,
        )

        # Create the directory
        try:
            os.makedirs(config.checkpoint_path, exist_ok=True)
            print(f"[Config] Using checkpoint_path: {config.checkpoint_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[Config] Warning: Failed to create checkpoint_path: {e}")
            # Fallback to current directory
            config.checkpoint_path = os.path.join(
                os.getcwd(),
                "outputs",
                config.project_name,
                config.run_name,
            )
            os.makedirs(config.checkpoint_path, exist_ok=True)
            print(f"[Config] Using fallback checkpoint_path: {config.checkpoint_path}")

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

    # Dataset
    # Standard behavior: single train loader, single eval loader.
    # Evaluation is scheduled based on training *epochs* using config.eval_interval.
    #
    # Construimos un único dataloader de entrenamiento.
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=config.epochs,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except:
        print("No evaluator found")
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Compute steps per epoch analíticamente a partir del tamaño del dataset,
    # sin iterar un dataloader extra. Esto asegura que:
    #  - La barra de progreso tiene un total realista.
    #  - El schedule de LR (compute_lr) usa el nº real de pasos de entrenamiento.
    import math

    steps_per_epoch = 0
    try:
        from dataset.latent_npz_dataset import LatentNPZDataset  # type: ignore
        import numpy as np  # type: ignore[import]

        train_dataset = getattr(train_loader, "dataset", None)
        if isinstance(train_dataset, LatentNPZDataset):
            # Latent NPZ episodic dataset: contamos pares (t, t+time_offset)
            train_dataset._discover_files()
            k = max(1, int(train_dataset.config.time_offset))
            total_pairs = 0
            for ep in train_dataset._episode_files:
                arr = np.load(ep, allow_pickle=True)["latents"]  # (T, H, F)
                T = arr.shape[0]
                total_pairs += max(0, T - k)
            steps_per_epoch = max(1, math.ceil(total_pairs / config.global_batch_size))
        else:
            # Fallback genérico: usar metadata (total_groups * mean_puzzle_examples)
            total_examples = float(train_metadata.total_groups) * float(  # type: ignore[attr-defined]
                getattr(train_metadata, "mean_puzzle_examples", 1.0)
            )
            steps_per_epoch = max(1, math.ceil(total_examples / config.global_batch_size))
    except Exception:
        # Si algo falla, mantener un valor de respaldo razonable basado en el
        # antiguo cálculo de total_steps.
        steps_per_epoch = max(1, train_state.total_steps // max(config.epochs, 1))

    # Actualizar total_steps para que el LR schedule y cualquier lógica que lo
    # use vean el nº real de pasos de entrenamiento.
    train_state.total_steps = steps_per_epoch * config.epochs

    # Si el checkpoint contiene un estado de entrenamiento completo (modelo + optimizadores + paso),
    # restáuralo ahora. Si el archivo sólo tiene pesos del modelo, esto no hará nada y
    # sólo se habrán cargado los pesos en create_model/load_checkpoint.
    train_state = load_train_state_if_available(config, train_state)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        # Main training progress bar (steps)
        progress_bar = tqdm.tqdm(
            total=steps_per_epoch * config.epochs,
            initial=train_state.step,
            desc="Train",
            unit="step",
            dynamic_ncols=True,
            leave=True,
            disable=False,
        )
        # Optional Weights & Biases logging
        if config.use_wandb and wandb is not None:
            wandb.init(
                project=config.project_name,
                name=config.run_name,
                config=config.model_dump(),
                settings=wandb.Settings(_disable_stats=True),  # type: ignore[attr-defined]
            )
            if wandb.run is not None:
                wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
            save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop - epoch-based over full dataset with standard eval scheduling
    if RANK == 0:
        print(f"[Rank {RANK}, World Size {WORLD_SIZE}]: Starting training for {config.epochs} epoch(s)")

    try:
        for epoch in range(config.epochs):
            if RANK == 0:
                print(f"TRAIN epoch {epoch+1}/{config.epochs}")
            train_state.model.train()

            for set_name, batch, global_batch_size in train_loader:
                metrics = train_batch(
                    config,
                    train_state,
                    batch,
                    global_batch_size,
                    rank=RANK,
                    world_size=WORLD_SIZE,
                )

                if RANK == 0 and metrics is not None:
                    # Persist training metrics (incl. loss) to disk for later plotting
                    if config.checkpoint_path is not None:
                        try:
                            os.makedirs(config.checkpoint_path, exist_ok=True)
                            metrics_path = os.path.join(config.checkpoint_path, "train_metrics.jsonl")
                            # Convert all metric values to plain Python floats where posible
                            serializable_metrics = {}
                            for k, v in metrics.items():
                                try:
                                    serializable_metrics[k] = float(v)
                                except Exception:
                                    # Fallback: store repr if it can't be converted cleanly
                                    serializable_metrics[k] = repr(v)
                            record = {
                                "step": int(train_state.step),
                                **serializable_metrics,
                            }
                            with open(metrics_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(record) + "\n")
                        except Exception as e:  # noqa: BLE001
                            print(f"[Metrics] Failed to write train_metrics.jsonl: {e}")

                    # Optional wandb logging
                    if config.use_wandb and wandb is not None and wandb.run is not None:
                        wandb.log(metrics, step=train_state.step)

                    # Update progress bar (one tick per training step)
                    if progress_bar is not None:
                        progress_bar.update(1)  # type: ignore

                    # Simple terminal logging
                    step = train_state.step
                    if step == 1 or step % config.log_interval == 0:
                        msg_parts = [f"[Train] step {step}"]
                        # Show a few key metrics
                        for k in sorted(metrics.keys()):
                            v = metrics[k]
                            if isinstance(v, (int, float)) or hasattr(v, "__float__"):
                                if (
                                    "loss" in k
                                    or k.endswith("/lr")
                                    or k.endswith("/mae")
                                    or k.endswith("/mse")
                                    or k.endswith("/steps")
                                ):
                                    msg_parts.append(f"{k}={float(v):.4g}")
                        print("  ".join(msg_parts))

                if config.ema:
                    ema_helper.update(train_state.model)

                # Periodic checkpointing every N training steps (step-based, not epoch-based).
                # Controlled via config.checkpoint_interval_steps (default: 1000).
                if (
                    RANK == 0
                    and config.checkpoint_path is not None
                    and config.checkpoint_interval_steps is not None
                    and (train_state.step % config.checkpoint_interval_steps == 0)
                ):
                    print(f"SAVE CHECKPOINT (periodic, every {config.checkpoint_interval_steps} steps)")
                    save_train_state(config, train_state)
                    # Also refresh plots so this behaves like a regular checkpoint.
                    try:
                        generate_plots_from_metrics(config.checkpoint_path)
                    except Exception as e:  # noqa: BLE001
                        print(f"[Checkpoint] Failed to generate plots at step {train_state.step}: {e}")

            # End for over train_loader (one full epoch)

            # Epoch-based eval scheduling: every N epochs, after min_eval_interval (interpreted in epochs)
                if (
                    eval_loader is not None
                    and evaluators is not None
                    and config.eval_interval is not None
                and (epoch + 1) >= (config.min_eval_interval or 0)
                and (epoch + 1) % config.eval_interval == 0
                ):
                    if RANK == 0:
                        print(f"EVALUATE (epoch {epoch+1})")
                    if config.ema:
                        print("SWITCH TO EMA")
                        train_state_eval = copy.deepcopy(train_state)
                        train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
                    else:
                        train_state_eval = train_state
                    train_state_eval.model.eval()
                    metrics_eval = evaluate(
                        config,
                        train_state_eval,
                        eval_loader,
                        eval_metadata,
                        evaluators,
                        rank=RANK,
                        world_size=WORLD_SIZE,
                        cpu_group=CPU_PROCESS_GROUP,
                    )

                    if RANK == 0 and metrics_eval is not None:
                        # Optional wandb logging
                        if config.use_wandb and wandb is not None and wandb.run is not None:
                            wandb.log(metrics_eval, step=train_state.step)

                        # Human-readable eval summary
                        print(f"[Eval] epoch {epoch+1}, step {train_state.step}")
                        for key, value in metrics_eval.items():
                            # Metrics may be nested dicts (e.g., per split)
                            if isinstance(value, dict):
                                print(f"  {key}:")
                                for sub_k, sub_v in value.items():
                                    print(f"    {sub_k}: {sub_v}")
                            else:
                                print(f"  {key}: {value}")

                    # Checkpointing on eval
                    if RANK == 0:
                        print("SAVE CHECKPOINT")
                    if RANK == 0 and config.checkpoint_every_eval:
                        save_train_state(config, train_state_eval)

                    if config.ema and train_state_eval is not train_state:
                        del train_state_eval


    except KeyboardInterrupt:
        if RANK == 0:
            print("[Train] Interrupted by user (KeyboardInterrupt). Finalizing...")
    finally:
        # Always try to save a final checkpoint and generate plots if possible
        if RANK == 0 and config.checkpoint_path is not None:
            try:
                print("[Finalize] Saving final training state...")
                save_train_state(config, train_state)
            except Exception as e:  # noqa: BLE001
                print(f"[Finalize] Failed to save final training state: {e}")

            try:
                print("[Finalize] Generating training plots from metrics...")
                generate_plots_from_metrics(config.checkpoint_path)
            except Exception as e:  # noqa: BLE001
                print(f"[Finalize] Failed to generate plots: {e}")

        # finalize distributed/wandb
        if dist.is_initialized():
            dist.destroy_process_group()
        if config.use_wandb and wandb is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    launch()
