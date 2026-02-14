#!/usr/bin/env python3
"""
BabyGR00T end-to-end pipeline runner.

This script keeps the repo "research-y" (it shells out to the existing entrypoints)
but makes it much more plug-and-play by:
  - creating required output directories
  - running steps in a consistent sequence with one set of CLI flags
  - setting a portable checkpoint root (CHECKPOINT_ROOT)

Steps (optional):
  1) Build VLM embeddings:     visual_embedding_builder.py
  2) Distill teacher latents:  gr00t_distiller.py
  3) Train TRM student:        pretrain.py (Hydra overrides)
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _abs(p: str | Path) -> str:
    return str(Path(p).expanduser().resolve())


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _run(cmd: List[str], *, env: Dict[str, str], cwd: Optional[str] = None) -> None:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[Pipeline] Running:\n  {printable}\n", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Workspace layout
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repo root (used to place default ./data and ./outputs dirs).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base data dir (defaults to <root>/data).",
    )
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=None,
        help="Base outputs dir (defaults to <root>/outputs).",
    )

    # Step toggles
    p.add_argument("--skip-vlm", action="store_true", help="Skip VLM embedding build.")
    p.add_argument("--skip-distill", action="store_true", help="Skip teacher latent distillation.")
    p.add_argument("--skip-train", action="store_true", help="Skip TRM training.")

    # HF / dataset options
    p.add_argument("--hf-token", default=None, help="Hugging Face token (also read from $HF_TOKEN).")

    # VLM step (visual_embedding_builder.py)
    p.add_argument("--vlm-out", type=Path, default=None, help="Output root for VLM embeddings.")
    p.add_argument("--vlm-model-id", default="NexaAI/NanoLLaVA", help="Hugging Face model id for VLM embeddings.")
    p.add_argument("--vlm-device", default="cuda", help="Device for VLM embedding build (cuda/cpu).")
    p.add_argument("--vlm-tasks-regex", default=r"^gr1_", help="Tasks regex for VLM embedding build.")
    p.add_argument("--vlm-batch-size", type=int, default=16, help="Frame batch size for VLM embedding build.")
    p.add_argument("--vlm-stride", type=int, default=1, help="Frame stride for VLM embedding build.")
    p.add_argument("--vlm-resume", action="store_true", help="Resume VLM embedding build.")
    p.add_argument("--vlm-limit", type=int, default=None, help="Process only N episodes total (smoke test).")

    # Distill step (gr00t_distiller.py)
    p.add_argument("--distill-out", type=Path, default=None, help="Output dir for distilled teacher latents.")
    p.add_argument("--distill-repo", default="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim", help="HF dataset repo id.")
    p.add_argument("--distill-model", default="nvidia/GR00T-N1.5-3B", help="Teacher model id or local path.")
    p.add_argument("--distill-embodiment", default="gr1", help="Embodiment tag for GR00T policy.")
    p.add_argument("--distill-tasks-filter", default=None, help="Regex filter for GR1 task dirs.")
    p.add_argument("--distill-max-episodes-per-task", type=int, default=None, help="Cap episodes per task.")
    p.add_argument("--distill-start-at-episode", type=int, default=0, help="Skip episodes < this index.")
    p.add_argument("--distill-require-video", action="store_true", help="Require video frames (skip if missing).")
    p.add_argument("--distill-video-key", default=None, help="Override video key (e.g., observation.images.front_view).")

    # Train step (pretrain.py / Hydra overrides)
    p.add_argument("--train-epochs", type=int, default=10, help="Training epochs.")
    p.add_argument("--train-global-batch-size", type=int, default=16, help="Global batch size.")
    p.add_argument("--train-lr", type=float, default=1e-4, help="Learning rate (base lr; internal schedule may override).")
    p.add_argument("--train-eval-interval", type=int, default=10, help="Eval every N epochs.")
    p.add_argument("--train-seed", type=int, default=0, help="Random seed.")
    p.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument(
        "--train-vlm-context-root",
        type=Path,
        default=None,
        help="If set, pass as vlm_context_root=... to training (cross-attn context).",
    )
    p.add_argument(
        "--hydra",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides passed through to pretrain.py (e.g. arch.hidden_size=1024).",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()

    root = args.root.expanduser().resolve()
    data_dir = (args.data_dir or (root / "data")).expanduser().resolve()
    outputs_dir = (args.outputs_dir or (root / "outputs")).expanduser().resolve()

    # Default step outputs
    vlm_out = (args.vlm_out or (data_dir / "vlm_embeddings")).expanduser().resolve()
    distill_out = (args.distill_out or (data_dir / "distill_out")).expanduser().resolve()

    # Create directories
    _ensure_dir(data_dir)
    _ensure_dir(outputs_dir)
    if not args.skip_vlm:
        _ensure_dir(vlm_out)
    if not args.skip_distill:
        _ensure_dir(distill_out)

    # Build environment for child scripts
    env = dict(os.environ)
    env["PYTHONPATH"] = _abs(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CHECKPOINT_ROOT"] = _abs(outputs_dir)

    # HF token propagation
    hf_token = args.hf_token or env.get("HF_TOKEN")
    if hf_token:
        env["HF_TOKEN"] = hf_token

    python = sys.executable

    # 1) VLM embeddings
    if not args.skip_vlm:
        cmd = [
            python,
            _abs(root / "visual_embedding_builder.py"),
            "--out-root",
            _abs(vlm_out),
            "--model-id",
            args.vlm_model_id,
            "--device",
            args.vlm_device,
            "--tasks-regex",
            args.vlm_tasks_regex,
            "--batch-size",
            str(args.vlm_batch_size),
            "--stride",
            str(args.vlm_stride),
        ]
        if hf_token:
            cmd += ["--hf-token", hf_token]
        if args.vlm_resume:
            cmd.append("--resume")
        if args.vlm_limit is not None:
            cmd += ["--limit", str(args.vlm_limit)]
        _run(cmd, env=env, cwd=_abs(root))

    # 2) Teacher latents
    if not args.skip_distill:
        cmd = [
            python,
            _abs(root / "gr00t_distiller.py"),
            "--repo",
            args.distill_repo,
            "--model",
            args.distill_model,
            "--embodiment",
            args.distill_embodiment,
            "--out",
            _abs(distill_out),
            "--start-at-episode",
            str(args.distill_start_at_episode),
        ]
        if args.distill_tasks_filter:
            cmd += ["--tasks-filter", args.distill_tasks_filter]
        if args.distill_max_episodes_per_task is not None:
            cmd += ["--max-episodes-per-task", str(args.distill_max_episodes_per_task)]
        if args.distill_require_video:
            cmd.append("--require-video")
        if args.distill_video_key:
            cmd += ["--video-key", args.distill_video_key]
        _run(cmd, env=env, cwd=_abs(root))

    # 3) Train TRM (Hydra)
    if not args.skip_train:
        hydra_overrides = [
            f"data_paths=[{_abs(distill_out)}]",
            f"epochs={int(args.train_epochs)}",
            f"global_batch_size={int(args.train_global_batch_size)}",
            f"lr={float(args.train_lr)}",
            f"eval_interval={int(args.train_eval_interval)}",
            f"seed={int(args.train_seed)}",
            f"use_wandb={'True' if args.use_wandb else 'False'}",
        ]
        if args.train_vlm_context_root is not None:
            hydra_overrides.append(f"vlm_context_root={_abs(args.train_vlm_context_root)}")
        # Convenience: if user built VLM embeddings but didn't explicitly pass root,
        # default training context root to that output.
        elif (not args.skip_vlm) and vlm_out.exists():
            hydra_overrides.append(f"vlm_context_root={_abs(vlm_out)}")

        if args.hydra:
            hydra_overrides.extend(args.hydra)

        cmd = [python, _abs(root / "pretrain.py"), *hydra_overrides]
        _run(cmd, env=env, cwd=_abs(root))

    print("\n[Pipeline] Done.", flush=True)
    print(f"[Pipeline] data_dir     = {_abs(data_dir)}", flush=True)
    print(f"[Pipeline] outputs_dir  = {_abs(outputs_dir)}", flush=True)
    print(f"[Pipeline] distill_out  = {_abs(distill_out)}", flush=True)
    if not args.skip_vlm:
        print(f"[Pipeline] vlm_out      = {_abs(vlm_out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



