#!/usr/bin/env python3
"""
Build GR1 OmniVLM Dense LV-Embeddings Dataset (No Frames)

This CLI downloads the GR00T-X Embodiment (sim) dataset subsets whose task
directories match a regex (default: ^gr1_), enumerates episodes, and extracts
per-frame dense hidden states [L, H] from OmniVLM conditioned on the episode
prompt(s). It writes per-episode NPZ shards and global manifests. Images are
never saved.

Key outputs per episode NPZ:
  - hidden: [N_frames, L, H] (default float16)
  - frame_idxs: [N_frames] int32
  - t_img: int32 (image tokens)
  - t_text: int32 (text tokens)
  - attention_mask: [L] uint8 (representative mask)
  - prompt_ids: [K] int32 (prompt IDs for this episode)

Global artifacts (under --out-root):
  - prompts.jsonl: unique prompt catalog (stable prompt_id)
  - manifest.jsonl: one record per episode shard with metadata
  - skipped.jsonl: records of skipped episodes with reasons
  - failures.jsonl: records of failures with traceback summary
  - build.log: detailed logs

This tool is idempotent with --resume, and supports --dry-run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import traceback
import signal
from pathlib import Path
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _lazy_imports():
    # Defer heavy imports until needed
    import torch  # type: ignore
    import decord  # type: ignore
    from huggingface_hub import snapshot_download, HfApi  # type: ignore
    from transformers import AutoModel, AutoProcessor, AutoTokenizer  # type: ignore
    import transformers  # type: ignore
    return torch, decord, snapshot_download, HfApi, AutoModel, AutoProcessor, AutoTokenizer, transformers

# Global debug state
DEBUG_STATE = {
    "current_episode": None,
    "current_frame_batch": None,
    "model_loaded": False,
    "last_log_time": time.time(),
    "hang_detected": False
}

def debug_log(message: str, level: str = "INFO"):
    """Enhanced debug logging with timestamps and state tracking"""
    global DEBUG_STATE
    current_time = time.time()
    elapsed = current_time - DEBUG_STATE["last_log_time"]
    DEBUG_STATE["last_log_time"] = current_time
    
    # Check for hangs (no log for >30 seconds)
    if elapsed > 30:
        DEBUG_STATE["hang_detected"] = True
        print(f"🚨 HANG DETECTED: No logs for {elapsed:.1f}s - {message}")
        sys.stdout.flush()
    
    state_info = f"[Episode: {DEBUG_STATE['current_episode']}, Batch: {DEBUG_STATE['current_frame_batch']}]"
    timestamp = time.strftime("%H:%M:%S")
    print(f"🔍 [{timestamp}] DEBUG {level} {state_info}: {message}")
    sys.stdout.flush()  # Force immediate output
    
    # Also log to file with immediate flush
    logging.getLogger().info(f"DEBUG {level} {state_info}: {message}")
    sys.stdout.flush()

def progress_log(message: str, progress: str = ""):
    """Progress logging with immediate output"""
    global DEBUG_STATE
    current_time = time.time()
    timestamp = time.strftime("%H:%M:%S")
    state_info = f"[Episode: {DEBUG_STATE['current_episode']}, Batch: {DEBUG_STATE['current_frame_batch']}]"
    
    if progress:
        print(f"📊 [{timestamp}] PROGRESS {state_info}: {message} ({progress})")
    else:
        print(f"📊 [{timestamp}] PROGRESS {state_info}: {message}")
    sys.stdout.flush()
    
    # Also log to file
    logging.getLogger().info(f"PROGRESS {state_info}: {message} ({progress})")
    sys.stdout.flush()

def debug_timer(func_name: str):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            debug_log(f"Starting {func_name}")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                debug_log(f"Completed {func_name} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                debug_log(f"ERROR in {func_name} after {elapsed:.2f}s: {e}", "ERROR")
                raise
        return wrapper
    return decorator

def timeout_handler(signum, frame):
    """Signal handler for timeout detection"""
    global DEBUG_STATE
    debug_log("TIMEOUT DETECTED! Process has been running too long", "ERROR")
    debug_log(f"Current state: Episode={DEBUG_STATE['current_episode']}, Batch={DEBUG_STATE['current_frame_batch']}", "ERROR")
    raise TimeoutError("Process timeout - likely hanging")

# Set up timeout signal
signal.signal(signal.SIGALRM, timeout_handler)


DATASET_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
DEFAULT_ALLOW_PATTERNS = [
    "gr1_*/meta/episodes.jsonl",
    "gr1_*/videos/**/observation.images.ego_view/episode_*.mp4",
]


@dataclasses.dataclass
class EpisodeMeta:
    episode_index: int
    prompts: List[str]
    length: Optional[int] = None


@dataclasses.dataclass
class EpisodePlan:
    task_dir: str  # relative to dataset root
    video_relpath: str  # relative to dataset root
    meta_relpath: str  # relative to dataset root
    episode_index: int
    prompts: List[str]
    joined_prompt: str
    prompt_ids: List[int]


def setup_logger(out_root: Path, verbose: bool = True) -> logging.Logger:
    out_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("build_gr1_omnivlm_embeddings")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    fh = logging.FileHandler(out_root / "build.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    print("🚀 STARTING SCRIPT - Initializing...")
    sys.stdout.flush()
    
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--model-id", type=str, default="NexaAI/OmniVLM-968M", help="Model ID on Hugging Face")
    parser.add_argument("--hf-token", type=str, default="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", help="Hugging Face token (default set for testing)")
    parser.add_argument("--stride", type=int, default=1, help="Frame sampling stride")
    parser.add_argument("--dtype", type=str, choices=["float16", "float32"], default="float16", help="Hidden dtype")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for frames per forward pass")
    parser.add_argument("--tasks-regex", type=str, default=r"^gr1_", help="Regex to filter task directories")
    parser.add_argument("--max-episodes-per-task", type=int, default=None, help="Limit episodes per task directory")
    parser.add_argument("--workers", type=int, default=0, help="Reserved for future parallelism (currently sequential)")
    parser.add_argument("--resume", action="store_true", help="Resume and skip existing shards")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing shards even if resume")
    parser.add_argument("--device", type=str, default="cuda" if torch_cuda_available() else "cpu", help="Device: cuda or cpu")
    parser.add_argument("--prompts-joiner", type=str, default=" || ", help="String to join multiple prompts")
    parser.add_argument("--include-pattern", type=str, nargs="*", default=None, help="Additional allow_patterns for dataset download")
    parser.add_argument("--exclude-pattern", type=str, nargs="*", default=None, help="Ignore patterns for dataset download")
    parser.add_argument("--dry-run", action="store_true", help="Plan and list episodes without running inference")
    parser.add_argument("--strict-metadata", action="store_true", help="Skip episodes if filename index mismatches metadata")
    parser.add_argument("--limit", type=int, default=None, help="Process only N episodes total for smoke tests")
    parser.add_argument("--frames-limit", type=int, default=None, help="Cut per-episode frames to first K frames after stride")
    parser.add_argument("--max-episode-mb", type=float, default=4096.0, help="Skip or downcast if estimated shard exceeds this size (MB)")
    parser.add_argument("--oversize-policy", type=str, choices=["skip", "force-f16"], default="force-f16", help="What to do when estimate exceeds max size")
    parser.add_argument("--assume-image-tokens", type=int, default=None, help="Override inferred image token count when needed")
    parser.add_argument("--zarr", action="store_true", help="Use Zarr instead of NPZ (disabled by default)")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Use an existing local dataset snapshot (skip download)")
    parser.add_argument("--skip-download", action="store_true", help="Reuse existing local snapshot under out-root if present")
    parser.add_argument("--local-model-dir", type=Path, default=None, help="Path to a local model directory to load offline")
    parser.add_argument("--local-files-only", action="store_true", help="Do not make any network calls when loading model/tokenizer")
    parser.add_argument("--concat-vision", action="store_true", help="Prepend projected vision tokens to last hidden states and save the combined sequence")
    print("🔧 PARSING ARGUMENTS...")
    sys.stdout.flush()
    args = parser.parse_args(argv)
    
    print(f"✅ ARGUMENTS PARSED: {args}")
    sys.stdout.flush()
    return args


def torch_cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


def get_versions() -> Dict[str, str]:
    versions = {}
    with contextlib.suppress(Exception):
        import transformers  # type: ignore
        versions["transformers_version"] = getattr(transformers, "__version__", "unknown")
    with contextlib.suppress(Exception):
        import torch  # type: ignore
        versions["torch_version"] = getattr(torch, "__version__", "unknown")
    return versions


def snapshot_gr1_dataset(out_root: Path, hf_token: Optional[str], include_patterns: Optional[List[str]], exclude_patterns: Optional[List[str]], logger: logging.Logger) -> Path:
    torch, decord, snapshot_download, HfApi, AutoModel, AutoProcessor, AutoTokenizer, transformers = _lazy_imports()
    allow_patterns = list(DEFAULT_ALLOW_PATTERNS)
    if include_patterns:
        allow_patterns.extend(include_patterns)
    # snapshot_download ignores None patterns
    logger.info("Downloading dataset snapshot %s (filtered)...", DATASET_ID)
    dataset_dir = snapshot_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        allow_patterns=allow_patterns,
        ignore_patterns=exclude_patterns,
        local_dir=str(out_root / "_datasets" / DATASET_ID.replace("/", "__")),
        token=hf_token,
    )
    logger.info("Dataset snapshot at %s", dataset_dir)
    return Path(dataset_dir)


def read_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[dict], append: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def normalize_prompt(text: str) -> str:
    # Trim and canonicalize whitespace
    return re.sub(r"\s+", " ", text.strip())


def safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def build_prompt_tables(prompts_path: Path) -> Tuple[Dict[str, int], List[dict]]:
    prompt_to_id: Dict[str, int] = {}
    rows: List[dict] = []
    if prompts_path.exists():
        for rec in read_jsonl(prompts_path):
            text = rec.get("text", "")
            pid = int(rec["prompt_id"])  # stable
            prompt_to_id[text] = pid
            rows.append(rec)
    return prompt_to_id, rows


def assign_prompt_ids(prompts: List[str], prompt_to_id: Dict[str, int], existing_rows: List[dict], tokenizer_info: Dict[str, str], tokenizer_encode_len_fn) -> List[int]:
    prompt_ids: List[int] = []
    # Existing IDs start at max(existing)+1
    next_id = (max(prompt_to_id.values()) + 1) if prompt_to_id else 0
    for p in prompts:
        p_norm = normalize_prompt(p)
        if p_norm in prompt_to_id:
            prompt_ids.append(prompt_to_id[p_norm])
            continue
        n_text_tokens = int(tokenizer_encode_len_fn(p_norm))
        row = {
            "prompt_id": next_id,
            "text": p_norm,
            "tokenizer": tokenizer_info["tokenizer_id"],
            "tokenizer_commit": tokenizer_info.get("tokenizer_commit"),
            "n_text_tokens": n_text_tokens,
        }
        prompt_to_id[p_norm] = next_id
        existing_rows.append(row)
        prompt_ids.append(next_id)
        next_id += 1
    return prompt_ids


def find_task_dirs(dataset_root: Path, tasks_regex: str) -> List[Path]:
    pat = re.compile(tasks_regex)
    return [p for p in sorted(dataset_root.iterdir()) if p.is_dir() and pat.match(p.name)]


def parse_episodes_jsonl(meta_path: Path) -> Dict[int, EpisodeMeta]:
    episodes: Dict[int, EpisodeMeta] = {}
    if not meta_path.exists():
        return episodes
    for rec in read_jsonl(meta_path):
        idx = int(rec.get("episode_index"))
        tasks = rec.get("tasks", [])
        if not isinstance(tasks, list):
            tasks = [str(tasks)]
        prompts = [normalize_prompt(str(t)) for t in tasks if str(t).strip()]
        episodes[idx] = EpisodeMeta(episode_index=idx, prompts=prompts, length=rec.get("length"))
    return episodes


def glob_videos(task_dir: Path) -> List[Path]:
    # Look for episode_*.mp4 under videos/**/observation.images.ego_view/
    return sorted(task_dir.glob("videos/**/observation.images.ego_view/episode_*.mp4"))


def derive_episode_index_from_filename(path: Path) -> Optional[int]:
    m = re.search(r"episode_(\d+)\.mp4$", path.name)
    if not m:
        return None
    return safe_int(m.group(1))


def estimate_episode_mb(n_frames: int, seq_len: int, hidden_size: int, dtype: str) -> float:
    bytes_per = 2 if dtype == "float16" else 4
    total_bytes = n_frames * seq_len * hidden_size * bytes_per
    return total_bytes / (1024.0 * 1024.0)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _move_to_device(obj, device, dtype):
    try:
        import torch  # type: ignore
    except Exception:
        return obj
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=dtype)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_move_to_device(v, device, dtype) for v in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    return obj


def main(argv: Optional[Sequence[str]] = None) -> int:
    print("🎯 ENTERING MAIN FUNCTION")
    sys.stdout.flush()
    
    args = parse_args(argv)
    print("📋 MAIN: Arguments parsed successfully")
    sys.stdout.flush()
    print("📝 MAIN: Setting up logger...")
    sys.stdout.flush()
    logger = setup_logger(args.out_root)
    print("✅ MAIN: Logger setup complete")
    sys.stdout.flush()

    print("🔍 MAIN: Getting versions...")
    sys.stdout.flush()
    versions = get_versions()
    print("✅ MAIN: Versions retrieved")
    sys.stdout.flush()
    
    print("📊 MAIN: Logging configuration...")
    sys.stdout.flush()
    logger.info("Config: %s", json.dumps({
        "out_root": str(args.out_root),
        "model_id": args.model_id,
        "tasks_regex": args.tasks_regex,
        "batch_size": args.batch_size,
        "stride": args.stride,
        "dtype": args.dtype,
        "device": args.device,
        "limit": args.limit,
        "frames_limit": args.frames_limit,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "strict_metadata": args.strict_metadata,
        "max_episode_mb": args.max_episode_mb,
        "oversize_policy": args.oversize_policy,
        "assume_image_tokens": args.assume_image_tokens,
        **versions,
    }, ensure_ascii=False))

    print("🗂️ MAIN: Resolving dataset root...")
    sys.stdout.flush()
    # Resolve dataset root, optionally skip download
    default_snapshot_dir = args.out_root / "_datasets" / DATASET_ID.replace("/", "__")
    if args.dataset_root is not None:
        print(f"📁 MAIN: Using provided dataset root: {args.dataset_root}")
        sys.stdout.flush()
        dataset_root = args.dataset_root
        logger.info("Using provided dataset root: %s", dataset_root)
        if not dataset_root.exists():
            print(f"❌ MAIN: Provided dataset root does not exist: {dataset_root}")
            sys.stdout.flush()
            logger.error("Provided --dataset-root does not exist: %s", dataset_root)
            return 2
    elif args.skip_download and default_snapshot_dir.exists():
        print(f"📁 MAIN: Reusing existing snapshot at {default_snapshot_dir}")
        sys.stdout.flush()
        dataset_root = default_snapshot_dir
        logger.info("Reusing existing snapshot at %s (skip-download)", dataset_root)
    else:
        print("📥 MAIN: Downloading dataset snapshot...")
        sys.stdout.flush()
        # Download dataset snapshot (filtered)
        dataset_root = snapshot_gr1_dataset(
            out_root=args.out_root,
            hf_token=args.hf_token,
            include_patterns=args.include_pattern,
            exclude_patterns=args.exclude_pattern,
            logger=logger,
        )
        print(f"✅ MAIN: Dataset downloaded to: {dataset_root}")
        sys.stdout.flush()

    print("📝 MAIN: Building prompts table...")
    sys.stdout.flush()
    # Build/Load prompts table
    prompts_path = args.out_root / "prompts.jsonl"
    prompt_to_id, prompt_rows = build_prompt_tables(prompts_path)
    print(f"✅ MAIN: Prompts table built with {len(prompt_to_id)} prompts")
    sys.stdout.flush()

    # Optional heavy deps; delay until needed (skip for dry-run)
    device = None
    dtype_torch = None
    processor = None
    model = None
    tokenizer = None
    model_commit = None
    processor_commit = None
    tokenizer_info = {"tokenizer_id": args.model_id, "tokenizer_commit": None}
    # Resolve token from env if not provided
    if args.hf_token is None:
        args.hf_token = os.environ.get("HF_TOKEN")

    if not args.dry_run:
        print("📦 MAIN: Loading heavy dependencies...")
        sys.stdout.flush()
        torch, decord, snapshot_download, HfApi, AutoModel, AutoProcessor, AutoTokenizer, transformers = _lazy_imports()
        print("✅ MAIN: Dependencies loaded")
        sys.stdout.flush()
        
        print(f"🎯 MAIN: Setting device to {args.device}")
        sys.stdout.flush()
        device = torch.device(args.device)
        dtype_torch = torch.float16 if args.dtype == "float16" else torch.float32

        print(f"🤖 MAIN: Loading model and processor: {args.model_id}")
        sys.stdout.flush()
        logger.info("Loading model and processor: %s", args.model_id)
        print("🔧 MAIN: Loading processor...")
        sys.stdout.flush()
        model_source = str(args.local_model_dir) if args.local_model_dir is not None else args.model_id
        processor = AutoProcessor.from_pretrained(
            model_source,
            trust_remote_code=True,
            token=args.hf_token,
            local_files_only=bool(args.local_files_only),
        )
        print("✅ MAIN: Processor loaded")
        sys.stdout.flush()
        print("🤖 MAIN: Loading model...")
        sys.stdout.flush()
        # Prefer AutoModelForCausalLM for multimodal chat models (e.g., LLaVA/Qwen-VL)
        model = None
        try:
            from transformers import AutoModelForCausalLM as _AutoModelForCausalLM  # type: ignore
            print("🔧 MAIN: Using AutoModelForCausalLM...")
            sys.stdout.flush()
            model = _AutoModelForCausalLM.from_pretrained(
                model_source,
                trust_remote_code=True,
                torch_dtype=dtype_torch,
                token=args.hf_token,
                local_files_only=bool(args.local_files_only),
            )
        except Exception:
            print("🔧 MAIN: AutoModelForCausalLM failed, trying AutoModel...")
            sys.stdout.flush()
            model = AutoModel.from_pretrained(
                model_source,
                trust_remote_code=True,
                torch_dtype=dtype_torch,
                token=args.hf_token,
                local_files_only=bool(args.local_files_only),
            )
        print("✅ MAIN: Model loaded successfully")
        sys.stdout.flush()
        
        print("🎯 MAIN: Setting model to eval mode and moving to device...")
        sys.stdout.flush()
        model.eval()
        model.to(device)
        print("✅ MAIN: Model moved to device")
        sys.stdout.flush()
        # Align custom model attribute used internally by some LLaVA forks
        with contextlib.suppress(Exception):
            setattr(model, "device", device)

        print("🔧 MAIN: Setting up vision tower and device alignment...")
        sys.stdout.flush()
        # Best-effort: ensure the vision tower is materialized and on the target device
        with contextlib.suppress(Exception):
            base = getattr(model, "get_model", lambda: model)()
            # Some custom models rely on `self.device`; align it with requested device
            try:
                setattr(base, "device", device)
            except Exception:
                pass
            vt = None
            if hasattr(base, "get_vision_tower"):
                vt = base.get_vision_tower()
            elif hasattr(base, "vision_tower"):
                vt = getattr(base, "vision_tower")
            elif hasattr(model, "vision_tower"):
                vt = getattr(model, "vision_tower")
            if vt is not None:
                if hasattr(vt, "to"):
                    vt.to(device=device, dtype=dtype_torch)
                # Some wrappers hold the model under .model or .vision_tower
                inner = getattr(vt, "model", None) or getattr(vt, "vision_tower", None)
                if inner is not None and hasattr(inner, "to"):
                    inner.to(device=device, dtype=dtype_torch)

        # Tokenizer for text-only length measurement
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                trust_remote_code=True,
                token=args.hf_token,
                local_files_only=bool(args.local_files_only),
            )

        # Model/processor commit SHA (best-effort)
        api = None
        if not args.local_files_only:
            api = HfApi()
            with contextlib.suppress(Exception):
                model_info = api.model_info(args.model_id)
                model_commit = getattr(model_info, "sha", None)
        processor_commit = model_commit
        tokenizer_info = {
            "tokenizer_id": args.model_id,
            "tokenizer_commit": model_commit,
        }

    print("🔍 MAIN: Finding task directories...")
    sys.stdout.flush()
    # Enumerate tasks and episodes
    tasks = find_task_dirs(dataset_root, args.tasks_regex)
    print(f"✅ MAIN: Found {len(tasks)} task directories matching {args.tasks_regex}")
    sys.stdout.flush()
    logger.info("Found %d task directories matching %s", len(tasks), args.tasks_regex)

    # Files for manifests and logs
    manifest_path = args.out_root / "manifest.jsonl"
    skipped_path = args.out_root / "skipped.jsonl"
    failures_path = args.out_root / "failures.jsonl"

    episodes_processed = 0
    total_gb_written = 0.0
    failures = 0
    skipped = 0

    if not args.dry_run:
        def tokenizer_len_fn(text: str) -> int:
            ids = tokenizer(text, return_tensors="pt").input_ids
            return int(ids.shape[1])
    else:
        def tokenizer_len_fn(text: str) -> int:  # type: ignore
            return 0

    # Iterate tasks
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        def tqdm(x, **kwargs):  # type: ignore
            return x
    print("🎬 MAIN: Starting task processing loop...")
    sys.stdout.flush()
    for task_dir in tasks:
        print(f"📁 MAIN: Processing task directory: {task_dir.name}")
        sys.stdout.flush()
        if args.limit is not None and episodes_processed >= args.limit:
            print(f"⏹️ MAIN: Reached limit of {args.limit} episodes")
            sys.stdout.flush()
            break
        meta_rel = Path(task_dir.name) / "meta" / "episodes.jsonl"
        meta_path = dataset_root / meta_rel
        if not meta_path.exists():
            print(f"⚠️ MAIN: Missing metadata: {meta_rel}")
            sys.stdout.flush()
            logger.warning("Missing metadata: %s", meta_rel)
            write_jsonl(skipped_path, [{"task_dir": str(task_dir.name), "reason": "missing episodes.jsonl"}])
            skipped += 1
            continue
        print(f"📊 MAIN: Loading episodes metadata from {meta_path}")
        sys.stdout.flush()
        episodes_meta = parse_episodes_jsonl(meta_path)
        print(f"📹 MAIN: Finding video files in {task_dir}")
        sys.stdout.flush()
        video_paths = glob_videos(task_dir)
        if args.max_episodes_per_task is not None:
            video_paths = video_paths[: args.max_episodes_per_task]
        print(f"✅ MAIN: Task {task_dir.name}: {len(video_paths)} candidate episodes")
        sys.stdout.flush()
        logger.info("Task %s: %d candidate episodes", task_dir.name, len(video_paths))

        total_episodes = len(video_paths if args.limit is None else video_paths[: max(0, args.limit - episodes_processed)])
        print(f"🎬 MAIN: Starting episode processing: {total_episodes} episodes total")
        sys.stdout.flush()
        
        for episode_idx, vid_path in enumerate(tqdm(video_paths if args.limit is None else video_paths[: max(0, args.limit - episodes_processed)], desc=f"{task_dir.name} episodes")):
            if args.limit is not None and episodes_processed >= args.limit:
                break
            
            # Update debug state
            DEBUG_STATE["current_episode"] = vid_path.name
            episode_progress = f"Episode {episode_idx + 1}/{total_episodes} ({((episode_idx + 1) / total_episodes) * 100:.1f}%)"
            debug_log(f"Processing episode: {vid_path.name} - {episode_progress}")
            progress_log(f"Starting episode: {vid_path.name}", episode_progress)
            
            ep_idx = derive_episode_index_from_filename(vid_path)
            if ep_idx is None:
                logger.warning("Cannot parse episode index from %s", vid_path)
                write_jsonl(skipped_path, [{"video_relpath": str(vid_path.relative_to(dataset_root)), "reason": "bad filename"}])
                skipped += 1
                episodes_processed += 1
                continue
            meta = episodes_meta.get(ep_idx)
            if meta is None:
                msg = "episode_index not found in metadata"
                logger.warning("Skipping %s: %s", vid_path.name, msg)
                write_jsonl(skipped_path, [{
                    "video_relpath": str(vid_path.relative_to(dataset_root)),
                    "episode_index": ep_idx,
                    "reason": msg,
                }])
                skipped += 1
                episodes_processed += 1
                continue

            # Prompts handling
            prompts = meta.prompts or []
            # If user set empty joiner, use first prompt only
            if args.prompts_joiner == "":
                joined_prompt = prompts[0] if prompts else ""
            else:
                joined_prompt = args.prompts_joiner.join(prompts) if prompts else ""
            # Assign prompt IDs and extend prompts table if needed
            prompt_ids = assign_prompt_ids(prompts, prompt_to_id, prompt_rows, tokenizer_info, tokenizer_len_fn)

            video_relpath = str(vid_path.relative_to(dataset_root))
            task_rel = vid_path.relative_to(dataset_root).parts[0]

            # Output path mirrors source tree
            embeddings_relpath = Path(video_relpath).with_suffix(".npz")
            embeddings_out_path = args.out_root / embeddings_relpath
            ensure_parent(embeddings_out_path)

            # Resume/overwrite policy
            if embeddings_out_path.exists() and args.resume and not args.overwrite:
                logger.info("Skip existing (resume): %s", embeddings_relpath)
                episodes_processed += 1
                continue

            if args.dry_run:
                logger.info("DRY-RUN: would process %s (episode %d, %d prompts)", video_relpath, ep_idx, len(prompts))
                episodes_processed += 1
                continue

            # Decode frames and extract hidden states
            try:
                debug_log("Loading video with decord")
                # Decord import (only when not dry-run)
                torch, decord, snapshot_download, HfApi, AutoModel, AutoProcessor, AutoTokenizer, transformers = _lazy_imports()
                vr = decord.VideoReader(str(vid_path))
                debug_log(f"Video loaded: {len(vr)} frames, {vr.get_avg_fps()} fps")
            except Exception as e:
                logger.error("Failed to open video %s: %s", video_relpath, e)
                write_jsonl(failures_path, [{
                    "video_relpath": video_relpath,
                    "episode_index": ep_idx,
                    "error": f"open_video: {type(e).__name__}: {e}",
                }])
                failures += 1
                episodes_processed += 1
                continue

            num_frames = int(len(vr))
            fps = float(vr.get_avg_fps() or 0.0)
            frame_idxs = list(range(0, num_frames, max(1, args.stride)))
            if args.frames_limit is not None:
                frame_idxs = frame_idxs[: args.frames_limit]
            
            debug_log(f"Frame processing: {num_frames} total, {len(frame_idxs)} selected frames")
            if not frame_idxs:
                logger.warning("No frames after stride/limit for %s", video_relpath)
                write_jsonl(skipped_path, [{
                    "video_relpath": video_relpath,
                    "episode_index": ep_idx,
                    "reason": "no_frames_after_stride",
                }])
                skipped += 1
                episodes_processed += 1
                continue

            hidden_chunks: List[np.ndarray] = []
            rep_attention_mask: Optional[np.ndarray] = None
            attention_policy = "first"
            seq_len: Optional[int] = None
            hidden_size: Optional[int] = None

            # Process batches of frames
            total_batches = (len(frame_idxs) + args.batch_size - 1) // args.batch_size
            debug_log(f"Starting batch processing: {len(frame_idxs)} frames in {total_batches} batches of {args.batch_size}")
            print(f"🎬 MAIN: Processing {len(frame_idxs)} frames in {total_batches} batches")
            sys.stdout.flush()
            
            for i in tqdm(range(0, len(frame_idxs), args.batch_size), desc=f"frames {video_relpath}", leave=False):
                batch_idx = i // args.batch_size + 1
                batch_progress = f"Batch {batch_idx}/{total_batches} ({batch_idx/total_batches*100:.1f}%)"
                frame_progress = f"Frames {i+1}-{min(i + args.batch_size, len(frame_idxs))}/{len(frame_idxs)}"
                DEBUG_STATE["current_frame_batch"] = f"{batch_idx}/{total_batches}"
                debug_log(f"Processing batch {batch_idx}: frames {i}-{min(i + args.batch_size, len(frame_idxs))}")
                progress_log(f"Processing batch: {frame_progress}", batch_progress)
                
                batch_indices = frame_idxs[i : i + args.batch_size]
                try:
                    debug_log("Decoding video frames")
                    batch_nd = vr.get_batch(batch_indices)  # (B, H, W, 3) RGB uint8 (decord NDArray)
                    batch_np = batch_nd.asnumpy()
                    images: List[np.ndarray] = [batch_np[j] for j in range(batch_np.shape[0])]
                    debug_log(f"Decoded {len(images)} frames, shapes: {[img.shape for img in images]}")
                except Exception as e:
                    logger.error("Failed to decode frames for %s at %s: %s", video_relpath, batch_indices, e)
                    write_jsonl(failures_path, [{
                        "video_relpath": video_relpath,
                        "episode_index": ep_idx,
                        "error": f"decode_frames: {type(e).__name__}: {e}",
                    }])
                    failures += 1
                    images = []
                if not images:
                    debug_log("No images after decode, skipping batch")
                    episodes_processed += 1
                    continue

                debug_log(f"Building inputs for {len(images)} images")
                texts = [joined_prompt] * len(images)

                # Build inputs depending on model API (LLaVA/NanoLLaVA often require process_images + chat template)
                inputs = None
                try:
                    debug_log("Attempting to build model inputs")
                    # First, prefer model.process_images when available (NanoLLaVA/Qwen2-VL custom API)
                    if hasattr(model, "process_images"):
                        debug_log("Using model.process_images")
                        try:
                            from PIL import Image  # type: ignore
                        except Exception as _:
                            Image = None  # type: ignore
                        debug_log("Converting images to PIL format")
                        pil_images = []
                        for arr in images:
                            if Image is not None:
                                pil_images.append(Image.fromarray(arr))
                        debug_log(f"Processed {len(pil_images)} PIL images")
                        debug_log("Calling model.process_images")
                        vision = model.process_images(pil_images, getattr(model, "config", None))
                        debug_log("Moving vision features to device")
                        vision = _move_to_device(vision, device, dtype_torch)
                        debug_log("Building text prompts")
                        base = f"USER: <image>\n{joined_prompt or ''}\nASSISTANT:"
                        prompt_texts: List[str] = [base] * len(images)
                        debug_log("Tokenizing text prompts")
                        tok_out = tokenizer(prompt_texts, return_tensors="pt", padding=True)
                        inputs = {
                            "input_ids": tok_out.input_ids.to(device),
                            "attention_mask": tok_out.attention_mask.to(device) if hasattr(tok_out, "attention_mask") else None,
                            "images": vision,
                        }
                        if inputs.get("attention_mask", None) is None:
                            inputs.pop("attention_mask", None)
                        debug_log("Inputs built successfully")
                    # Next, prefer processor.image_processor to produce pixel_values
                    else:
                        image_proc = getattr(processor, "image_processor", None)
                        if image_proc is not None:
                            pixel = image_proc(images=images, return_tensors="pt")
                            pixel_values = pixel.get("pixel_values")
                            pixel_values = _move_to_device(pixel_values, device, dtype_torch)
                            base = f"USER: <image>\n{joined_prompt or ''}\nASSISTANT:"
                            prompt_texts: List[str] = [base] * len(images)
                            tok_out = tokenizer(prompt_texts, return_tensors="pt", padding=True)
                            inputs = {
                                "input_ids": tok_out.input_ids.to(device),
                                "attention_mask": tok_out.attention_mask.to(device) if hasattr(tok_out, "attention_mask") else None,
                                # Llava/Qwen-VL forward expects `images` when it handles encoding internally
                                "images": pixel_values,
                            }
                            if inputs.get("attention_mask", None) is None:
                                inputs.pop("attention_mask", None)
                        else:
                            # Fallback: generic processor call (may fail on older processors)
                            inputs = processor(images=images, text=texts, return_tensors="pt", padding=True)
                            for k, v in list(inputs.items()):
                                if hasattr(v, "to"):
                                    inputs[k] = v.to(device)
                except Exception as e:
                    logger.error("Processor failed for %s: %s", video_relpath, e)
                    write_jsonl(failures_path, [{
                        "video_relpath": video_relpath,
                        "episode_index": ep_idx,
                        "error": f"processor: {type(e).__name__}: {e}",
                    }])
                    failures += 1
                    episodes_processed += 1
                    continue

                # Optionally extract projected vision tokens to concatenate later
                vis_tokens = None
                if args.concat_vision:
                    try:
                        base_model = getattr(model, "get_model", lambda: model)()
                        images_tensor = inputs.get("images", None)
                        if images_tensor is not None and hasattr(base_model, "encode_images"):
                            # Try to encode on the fly; align projector device as needed
                            try:
                                vis_tokens = base_model.encode_images(images_tensor)
                            except RuntimeError as e:
                                if "Expected all tensors to be on the same device" in str(e):
                                    mm_proj = getattr(base_model, "mm_projector", None) or getattr(model, "mm_projector", None)
                                    if mm_proj is not None:
                                        for mod in mm_proj.modules():
                                            if hasattr(mod, "to"):
                                                mod.to(device=images_tensor.device)
                                        vis_tokens = base_model.encode_images(images_tensor)
                                if vis_tokens is None:
                                    raise
                        if vis_tokens is not None:
                            # Ensure dtype/device
                            vis_tokens = vis_tokens.to(device=device, dtype=dtype_torch)
                    except Exception as e:
                        logger.warning("concat_vision failed to extract vision tokens for %s: %s", video_relpath, e)
                        vis_tokens = None

                debug_log("Starting model forward pass")
                with torch.no_grad():
                    try:
                        # Set timeout for model forward pass (60 seconds)
                        signal.alarm(60)
                        debug_log("Calling model(**inputs) with output_hidden_states=True")
                        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                        signal.alarm(0)  # Cancel timeout
                        debug_log("Model forward pass completed successfully")
                    except TypeError:
                        signal.alarm(0)  # Cancel timeout
                        debug_log("TypeError in model call, trying degraded options")
                        # Some models might not accept use_cache/output_hidden_states; try degraded
                        try:
                            signal.alarm(60)  # Set timeout again
                            debug_log("Trying model call without use_cache")
                            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                            signal.alarm(0)  # Cancel timeout
                            debug_log("Model call succeeded without use_cache")
                        except TypeError:
                            signal.alarm(0)  # Cancel timeout
                            signal.alarm(60)  # Set timeout again
                            debug_log("Trying model call with minimal options")
                            outputs = model(**inputs, return_dict=True)
                            signal.alarm(0)  # Cancel timeout
                            debug_log("Model call succeeded with minimal options")
                    except RuntimeError as e:
                        # Handle device mismatches between image features and mm_projector (common in LLaVA forks)
                        msg = str(e)
                        if "Expected all tensors to be on the same device" in msg:
                            try:
                                base = getattr(model, "get_model", lambda: model)()
                                mm_proj = getattr(base, "mm_projector", None) or getattr(model, "mm_projector", None)
                                if mm_proj is not None:
                                    # Move mm_projector to CPU to match encode_images(image_features) device
                                    for mod in mm_proj.modules():
                                        if hasattr(mod, "to"):
                                            mod.to(device="cpu")
                                    outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
                                else:
                                    raise e
                            except Exception:
                                raise e
                        else:
                            raise e
                # Extract hidden states
                debug_log("Extracting hidden states from model outputs")
                last_hs = getattr(outputs, "last_hidden_state", None)
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is not None:
                    debug_log(f"Found hidden_states with {len(hidden_states)} layers")
                    last = hidden_states[-1]
                    debug_log(f"Last hidden state shape: {last.shape}")
                elif last_hs is not None:
                    debug_log("Using last_hidden_state from model")
                    last = last_hs
                    debug_log(f"Last hidden state shape: {last.shape}")
                else:
                    debug_log("ERROR: No hidden states found in model outputs", "ERROR")
                    logger.error("Model outputs lack hidden states for %s", video_relpath)
                    write_jsonl(failures_path, [{
                        "video_relpath": video_relpath,
                        "episode_index": ep_idx,
                        "error": "no_hidden_states",
                    }])
                    failures += 1
                    episodes_processed += 1
                    continue

                # last: [B, L, H]
                debug_log(f"Validating hidden state shape: {last.shape}")
                if last.dim() != 3:
                    debug_log(f"ERROR: Unexpected hidden shape {tuple(last.shape)}", "ERROR")
                    logger.error("Unexpected hidden shape %s for %s", tuple(last.shape), video_relpath)
                    write_jsonl(failures_path, [{
                        "video_relpath": video_relpath,
                        "episode_index": ep_idx,
                        "error": f"bad_hidden_shape:{tuple(last.shape)}",
                    }])
                    failures += 1
                    episodes_processed += 1
                    continue

                # Optionally prepend vision tokens along sequence dimension
                if args.concat_vision and vis_tokens is not None:
                    # vis_tokens: [B, V, H]
                    try:
                        if vis_tokens.dim() == 3 and vis_tokens.size(0) == last.size(0):
                            # Some models may return pooled features [B, H]; guard
                            if vis_tokens.size(-1) == last.size(-1):
                                last = torch.cat([vis_tokens, last], dim=1)
                            else:
                                logger.warning("Skip concat_vision: hidden size mismatch %s vs %s for %s", tuple(vis_tokens.shape), tuple(last.shape), video_relpath)
                        else:
                            logger.warning("Skip concat_vision: unexpected vision shape %s for %s", tuple(vis_tokens.shape) if hasattr(vis_tokens, 'shape') else None, video_relpath)
                    except Exception as e:
                        logger.warning("concat_vision failed during concat for %s: %s", video_relpath, e)

                bsz, L, H = int(last.shape[0]), int(last.shape[1]), int(last.shape[2])
                seq_len = seq_len or L
                hidden_size = hidden_size or H
                if seq_len != L or hidden_size != H:
                    logger.error("Inconsistent sequence dims across batches: expected (%d,%d) got (%d,%d)", seq_len, hidden_size, L, H)
                    write_jsonl(failures_path, [{
                        "video_relpath": video_relpath,
                        "episode_index": ep_idx,
                        "error": "inconsistent_seq_dims",
                    }])
                    failures += 1
                    continue

                # Attention mask handling
                attn = inputs.get("attention_mask", None)
                if attn is not None:
                    attn_np = attn.detach().to("cpu").numpy().astype(np.uint8)  # [B, L]
                    # If we prepended vision tokens, extend the mask with ones for that many tokens
                    if args.concat_vision and vis_tokens is not None and vis_tokens.dim() == 3:
                        V = int(vis_tokens.shape[1])
                        if V > 0:
                            ones = np.ones((attn_np.shape[0], V), dtype=np.uint8)
                            attn_np = np.concatenate([ones, attn_np], axis=1)
                    if attn_np.shape[0] > 1:
                        first = attn_np[0]
                        all_equal = np.all(attn_np == first[None, :])
                        if not all_equal:
                            rep = np.bitwise_or.reduce(attn_np, axis=0)
                            rep_attention_mask = rep
                            attention_policy = "or"
                        else:
                            rep_attention_mask = first
                    else:
                        rep_attention_mask = attn_np[0]

                # Collect hidden states to CPU numpy
                last_cpu = last.detach().to("cpu")
                if dtype_torch == torch.float16:
                    last_cpu = last_cpu.half()
                else:
                    last_cpu = last_cpu.float()
                hidden_chunks.append(last_cpu.numpy())

                # Free GPU memory
                del outputs, last, last_cpu
                for k in list(inputs.keys()):
                    del inputs[k]
                if torch.cuda.is_available() and args.device.startswith("cuda"):
                    torch.cuda.empty_cache()

            # After batches
            debug_log(f"Batch processing complete. Hidden chunks: {len(hidden_chunks)}")
            if not hidden_chunks:
                debug_log("ERROR: No hidden states produced", "ERROR")
                logger.warning("No hidden states produced for %s", video_relpath)
                write_jsonl(skipped_path, [{
                    "video_relpath": video_relpath,
                    "episode_index": ep_idx,
                    "reason": "no_hidden_produced",
                }])
                skipped += 1
                episodes_processed += 1
                continue

            debug_log("Concatenating hidden state chunks")
            hidden = np.concatenate(hidden_chunks, axis=0)  # [N, L, H]
            n_frames_out = hidden.shape[0]
            debug_log(f"Concatenated hidden states: shape {hidden.shape}")
            if n_frames_out != len(frame_idxs):
                debug_log(f"WARNING: Frame count mismatch hidden({n_frames_out}) vs idxs({len(frame_idxs)})", "WARNING")
                logger.warning("Frame count mismatch hidden(%d) vs idxs(%d) for %s", n_frames_out, len(frame_idxs), video_relpath)

            L = int(hidden.shape[1])
            H = int(hidden.shape[2])

            # Token counts
            try:
                n_text_tokens = tokenizer_len_fn(joined_prompt)
            except Exception:
                n_text_tokens = 0
            if args.assume_image_tokens is not None:
                t_img = int(args.assume_image_tokens)
                t_text = L - t_img
            else:
                t_text = int(n_text_tokens)
                t_img = L - t_text
                if t_img <= 0 or t_text <= 0:
                    msg = f"invalid token split: t_img={t_img}, t_text={t_text}, L={L}"
                    logger.error("%s for %s", msg, video_relpath)
                    write_jsonl(failures_path, [{
                        "video_relpath": video_relpath,
                        "episode_index": ep_idx,
                        "error": msg,
                    }])
                    failures += 1
                    episodes_processed += 1
                    continue

            # Size guardrail
            est_mb = estimate_episode_mb(n_frames_out, L, H, args.dtype)
            write_dtype = args.dtype
            if est_mb > args.max_episode_mb:
                if args.oversize_policy == "skip":
                    logger.warning("Skip %s: est %.1f MB > %.1f MB", video_relpath, est_mb, args.max_episode_mb)
                    write_jsonl(skipped_path, [{
                        "video_relpath": video_relpath,
                        "episode_index": ep_idx,
                        "reason": f"oversize:{est_mb:.1f}MB",
                    }])
                    skipped += 1
                    episodes_processed += 1
                    continue
                elif args.oversize_policy == "force-f16" and args.dtype != "float16":
                    logger.info("Downcasting to float16 for %s due to size (%.1f MB)", video_relpath, est_mb)
                    hidden = hidden.astype(np.float16, copy=False)
                    write_dtype = "float16"

            # Save NPZ
            debug_log("Preparing to save NPZ file")
            # Use a temporary file that still ends with .npz to avoid numpy appending another .npz
            tmp_path = embeddings_out_path.with_name(embeddings_out_path.stem + ".tmp.npz")
            debug_log(f"Saving to: {embeddings_out_path}")
            try:
                debug_log("Writing NPZ file with compressed format")
                np.savez_compressed(
                    tmp_path,
                    hidden=hidden.astype(np.float16 if write_dtype == "float16" else np.float32, copy=False),
                    frame_idxs=np.asarray(frame_idxs, dtype=np.int32),
                    t_img=np.int32(t_img),
                    t_text=np.int32(t_text),
                    attention_mask=(rep_attention_mask.astype(np.uint8) if rep_attention_mask is not None else np.zeros((L,), dtype=np.uint8)),
                    prompt_ids=np.asarray(prompt_ids, dtype=np.int32),
                )
                debug_log("NPZ file written successfully")
                # Rename atomically
                if embeddings_out_path.exists():
                    embeddings_out_path.unlink()
                tmp_path.replace(embeddings_out_path)
                debug_log("NPZ file renamed atomically")
                
                # Completion logging
                episode_progress = f"Episode {episode_idx + 1}/{total_episodes} ({((episode_idx + 1) / total_episodes) * 100:.1f}%)"
                progress_log(f"✅ Episode completed: {vid_path.name}", episode_progress)
                print(f"🎉 MAIN: Episode {vid_path.name} completed successfully!")
                sys.stdout.flush()
            except Exception as e:
                debug_log(f"ERROR: Failed to write NPZ: {e}", "ERROR")
                logger.error("Failed to write NPZ %s: %s", embeddings_relpath, e)
                with contextlib.suppress(Exception):
                    if tmp_path.exists():
                        tmp_path.unlink()
                write_jsonl(failures_path, [{
                    "video_relpath": video_relpath,
                    "episode_index": ep_idx,
                    "error": f"write_npz: {type(e).__name__}: {e}",
                }])
                failures += 1
                episodes_processed += 1
                continue

            # Attention policy sidecar
            sidecar = embeddings_out_path.with_suffix(".meta.json")
            with sidecar.open("w", encoding="utf-8") as f:
                json.dump({"attention_policy": attention_policy}, f)

            # Manifest entry
            episode_uid = str(Path(task_rel) / Path(video_relpath).name)
            manifest_row = {
                "episode_uid": episode_uid,
                "task_dir": str(task_rel),
                "video_relpath": video_relpath,
                "meta_relpath": str(meta_rel),
                "embeddings_relpath": str(embeddings_relpath),
                "episode_index": ep_idx,
                "n_frames": n_frames_out,
                "fps": fps,
                "t_img": t_img,
                "t_text": t_text,
                "L": L,
                "H": H,
                "prompt_ids": prompt_ids,
                "model_id": args.model_id,
                "model_commit": model_commit,
                "processor_commit": processor_commit,
                "transformers_version": versions.get("transformers_version"),
                "dtype": write_dtype,
            }
            write_jsonl(manifest_path, [manifest_row])

            debug_log(f"Episode {vid_path.name} completed successfully")
            episodes_processed += 1
            total_gb_written += (os.path.getsize(embeddings_out_path) / (1024**3))
            logger.info("Wrote %s [N=%d, L=%d, H=%d] (%.2f GB total)", embeddings_relpath, n_frames_out, L, H, total_gb_written)

    # Write/extend prompts.jsonl
    if prompt_rows:
        # We may have duplicates in memory (shouldn't), but write_jsonl appends; ensure unique by filtering against disk
        existing_texts = set()
        if prompts_path.exists():
            for rec in read_jsonl(prompts_path):
                existing_texts.add(rec.get("text"))
        new_rows = [r for r in prompt_rows if r.get("text") not in existing_texts]
        if new_rows:
            write_jsonl(prompts_path, new_rows)

    logger.info("Done. Episodes processed: %d, skipped: %d, failures: %d, total written: %.2f GB", episodes_processed, skipped, failures, total_gb_written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


