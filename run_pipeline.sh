#!/bin/bash
# One-shot clone-and-run pipeline for the ~100M VAE-TRM on BridgeData V2.
#
#   git clone <repo> && cd <repo>
#   pip install -e .                       # deps (torch, transformers, datasets, …)
#   BABYGROOT_DATA=/mnt/nvme/babygroot ./run_pipeline.sh > pipeline.log 2>&1 &
#
# Stages (each is skipped if its output already exists, so this is resumable):
#   1. Download + augment + cache InternVL3 vision features (streams Bridge V2
#      from HuggingFace — no manual data download).            [GPU, hours]
#   2. Train the VQ-VAE action codebook on Bridge's 7-DoF actions.  [GPU, ~1 h]
#   3. Train the 100M VAE-TRM policy on the cached features.    [GPU, hours]
#
# IMPORTANT: training is disk-I/O-bound on the vision cache (~285 MB/step). Put
# BABYGROOT_DATA on a FAST LOCAL DISK (NVMe/ext4) — never a network/FUSE mount.
# If you already have a built cache, copy it to $CACHE_DIR and stage 1 is skipped.
#
# Knobs (env): BABYGROOT_DATA, DATASET_ID, CAMERA, N_EPS, N_VIS_AUG, VAE_STEPS,
#              CACHE_DIR, VAE_CKPT, POL_CKPT.
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${BABYGROOT_DATA:-$PWD/data}"
DATASET_ID="${DATASET_ID:-IPEC-COMMUNITY/bridge_orig_lerobot}"
CAMERA="${CAMERA:-observation.images.image_0}"
N_EPS="${N_EPS:-1800}"
N_VIS_AUG="${N_VIS_AUG:-3}"
VAE_STEPS="${VAE_STEPS:-16000}"
CACHE_DIR="${CACHE_DIR:-$DATA_ROOT/cache/oxe_vision_cache_v2}"
VAE_CKPT="${VAE_CKPT:-$DATA_ROOT/ckpts/oxe_vqvae_1800ep_16k.pt}"
POL_CKPT="${POL_CKPT:-$DATA_ROOT/ckpts/oxe_strm_vae100m.pt}"
mkdir -p "$CACHE_DIR" "$(dirname "$VAE_CKPT")"

echo "=== BabyGR00T VAE-TRM 100M pipeline — START $(date) ==="
echo "  DATA_ROOT=$DATA_ROOT   (override via BABYGROOT_DATA; use a fast local disk)"
echo "  dataset=$DATASET_ID  camera=$CAMERA  n_eps=$N_EPS  n_vis_aug=$N_VIS_AUG"

# ── Stage 1: vision cache (download + augment + cache) ──
if [ -f "$CACHE_DIR/meta.json" ]; then
    echo "=== STAGE 1/3: cache already at $CACHE_DIR — skipping ==="
else
    echo "=== STAGE 1/3: build vision cache ($N_EPS eps, $N_VIS_AUG aug) — $(date) ==="
    python -u -m scripts.cache_vision \
        --dataset oxe --oxe-dataset-id "$DATASET_ID" --oxe-camera "$CAMERA" \
        --cache-dir "$CACHE_DIR" --n-eps-cap "$N_EPS" --n-vis-aug "$N_VIS_AUG"
fi

# ── Stage 2: VQ-VAE action codebook (7-DoF) ──
if [ -f "$VAE_CKPT" ]; then
    echo "=== STAGE 2/3: VQ-VAE already at $VAE_CKPT — skipping ==="
else
    echo "=== STAGE 2/3: train VQ-VAE (action-dim 7, $VAE_STEPS steps) — $(date) ==="
    python -u -m scripts.train_vqvae \
        --dataset oxe --oxe-dataset-id "$DATASET_ID" --action-dim 7 \
        --steps "$VAE_STEPS" --batch-size "${VAE_BATCH:-32}" --n-eps-cap "$N_EPS" \
        --ckpt-path "$VAE_CKPT"
fi

# ── Stage 3: 100M VAE-TRM policy (auto-resumes from POL_CKPT) ──
echo "=== STAGE 3/3: VAE-TRM 100M policy — $(date) ==="
BABYGROOT_DATA="$DATA_ROOT" VAE_CKPT="$VAE_CKPT" CACHE_DIR="$CACHE_DIR" POL_CKPT="$POL_CKPT" \
    bash launch_strm_vae_bridge.sh

echo "=== PIPELINE DONE $(date) ==="
echo "  cache:  $CACHE_DIR"
echo "  vae:    $VAE_CKPT"
echo "  policy: $POL_CKPT"
