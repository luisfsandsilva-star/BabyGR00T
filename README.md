# BabyGR00T: Making Foundational Robotic Models Small, Fast, and Scalable (work in progress)

This repo contains a **teacher→student distillation pipeline** built around:

- **`gr00t_distiller.py`**: runs the **GR00T N1.5** policy on GR1 episodes and saves per-timestep **teacher latents** to `.npz`.
- **`pretrain.py` / `finetune.py`**: trains a **Tiny Recursive Reasoning Model (TRM)** to do **next-step latent prediction** (regression) on those `.npz` episodes.
- **`visual_embedding_builder.py`** (optional): builds per-episode **VLM embeddings** (e.g. NanoLLaVA) that can be fed to TRM via **cross-attention** as extra context.

- ## Example Cross-Attention Map

![TRM Cross-Attention Map](assets/attn_map.jpeg)


The goal is to experiment with making a smaller student learn to “track” a larger teacher’s internal representations. This codebase is **not** a turnkey deployment project.

## What’s in the repo

- **Training entrypoint**: `pretrain.py` (Hydra config: `config/cfg_pretrain.yaml`)
- **Student model**: `models/recursive_reasoning/trm.py` (`TinyRecursiveReasoningModel_ACTV1`)
- **Latent dataset loader**: `dataset/latent_npz_dataset.py` (`LatentNPZDataset`)
- **Teacher latent dumper**: `gr00t_distiller.py`
- **Optional VLM context builder**: `visual_embedding_builder.py`

## Setup

### Python environment

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

### GR00T python package dependency

`gr00t_distiller.py` requires a python package that provides `gr00t.model.policy.Gr00tPolicy`.
This package is not vendored in this repo yet, so you’ll need to install it separately (or add it once it’s included).

## Step 1 — (Optional) Build VLM embedding context

If you want TRM to d over per-episode VLM embeddings, generate them first:

```bash
python visual_embedding_builder.py \
  --out-root /abs/path/to/vlm_embeddings \
  --tasks-regex '^gr1_' \
  --device cuda \
  --resume
```
**TRM (Tiny Recursive Model)**  
A ~140M-parameter recursive reasoning model developed by Samsung SAIT Montréal.  
It updates a latent state *z* and candidate output *y* through multiple refinement steps given input *x*, achieving high reasoning quality without large-scale architectures.

**Encoder–Decoder Interface**  
Encodes state and action tokens into TRM latents, then decodes refined latents back into next-state or action predictions, enabling closed-loop policy reasoning.

This writes per-episode `.npz` files whose main array is typically:

- `hidden`: `[N_frames, V_tokens, F_ctx]`

During training, these are loaded and provided to the TRM blocks as `cross_context_raw`.

## Step 2 — Distill teacher latents from GR00T N1.5

Run the teacher over GR1 episodes and dump latents:

```bash
python gr00t_distiller.py \
  --repo nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --model nvidia/GR00T-N1.5-3B \
  --embodiment gr1 \
  --out /abs/path/to/distill_out
```

Output layout (per task directory):

- `distill_out/<task>/latents/episode_000123.npz` (teacher latents)
- `distill_out/<task>/metadata/episode_000123.jsonl` (pointers + shapes)

Each `episode_*.npz` is expected to contain:

- `latents`: typically shaped `[T, H, F]` (episode length × token length × feature dim)

## Step 3 — Train TRM on the latent episodes (regression)

Edit `config/cfg_pretrain.yaml` and set `data_paths` to your distilled output root(s),
or override via Hydra:

```bash
python pretrain.py \
  data_paths=[/abs/path/to/distill_out] \
  global_batch_size=16 \
  epochs=10
```

## One-command runner (recommended)

If you want a single script that creates directories and runs the steps in sequence:

```bash
python pipeline.py \
  --train-epochs 10 \
  --train-global-batch-size 16 \
  --vlm-resume
```

Common variations:

- Skip VLM embeddings:

```bash
python pipeline.py --skip-vlm
```
| Component | Role | Key Traits |
|------------|------|------------|
| **TRM** | Recursive reasoning | 140 M parameters · latent refinement · minimal compute |
| **nanoLLaVA-1B** | Visual–language grounding | 1 B parameters · SigLIP encoder · edge-friendly |
| **Distillation Pipeline** | Teacher–student training | Representation + behavioral transfer |

- Smoke test (few episodes) for the VLM builder:

```bash
python pipeline.py --vlm-limit 2
```

### Optional: enable VLM cross-tion context

Provide VLM context directories (or a root to auto-discover) via config overrides:

```bash
python pretrain.py \
  data_paths=[/abs/path/to/distill_out] \
  vlm_context_root=/abs/path/to/vlm_embeddings \
  global_batch_size=16 \
  epochs=10
```

## Outputs / logging

- **Checkpoints + logs**: `pretrain.py` writes to `/workspace/outputs/<project>/<run_name>/...` by default (designed for Docker mounts).
- **Metrics**: `train_metrics.jsonl` (JSONL, one line per step)
- **Plots**: if `matplotlib` is available, plots are auto-generated next to `train_metrics.jsonl`.

## Notes / known rough edges

- **Docker is not plug-and-play in this snapshot**: `Dockerfile` expects a `grutito/` directory (a GR00T python package) which is not present here.
- If you fix up the Docker image locally, `scripts/docker-run.sh distill` runs `gr00t_distiller.py` inside the container.

## Citation

If you use this code, please cite the upstream works you build on (GR00T, TRM, NanoLLaVA) and clearly document your dataset provenance and distillation setup.
If you use BabyGR00T in research or development, please cite:

```bibtex
@article{sandoval2025babygr00t,
  title  = {BabyGR00T: Making Foundational Robotic Models Small, Fast, and Scalable},
  author = {Sandoval, L. F. and Hernandez, A. and Podesta, M. O. and Nieto, N. and Munoz, L. A.},
  year         = {2026},
  institution  = {Minerva Labs},
  address      = {Monterrey, N.L., México}
}
```

**Related Works**

- Jolicoeur-Martineau, A. (2025). *Less is More: Recursive Reasoning with Tiny Networks (TRM).*  
- Liu et al. (2024). *nanoLLaVA-1B: Compact Vision-Language Models for Edge AI.*
- Bjorck et al. (2025). *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots.*

---

## Acknowledgments

Developed as part of the **BabyGR00T** research effort on efficient embodied intelligence.  
Thanks to contributors from open-source projects on recursive reasoning (Samsung SAIT) and compact VLMs (nanoLLaVA team).  
GR00T N1.5 model and datasets are referenced solely for academic distillation and benchmarking purposes.
