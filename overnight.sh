#!/bin/bash
# BridgeData V2 (OXE) overnight pipeline — cache + CQ-VAE + policy in one go.
#
# Disk budget: ~20 GB. At ~24 MB/chunk and ~2 chunks/episode at Bridge's
# 5 fps, that's ~350 episodes (~700 chunks ~ 17 GB cache + ~1 GB ckpts).
#
# Stages:
#   1. Cache InternVL3 features for 350 Bridge episodes  (~1 h)
#   2. Train CQ-VAE on Bridge's 7-DoF action distribution (~45 min)
#   3. Train v5 policy on the cached features            (~4-5 h)
#
# set -e stops the chain on any failure so we don't burn the rest of the
# night on a doomed run.
set -e
cd /home/alex/Desktop/BabyGR00T

DATASET_ID=IPEC-COMMUNITY/bridge_orig_lerobot
CAMERA=observation.images.image_0
N_EPS_CAP=350
CACHE_DIR=oxe_vision_cache
VAE_CKPT=oxe_vae_revin.pt
POL_CKPT=oxe_strm_v5.pt

echo "=== START: $(date) ==="
echo "  dataset:   $DATASET_ID"
echo "  camera:    $CAMERA"
echo "  n_eps_cap: $N_EPS_CAP"
echo "  cache:     $CACHE_DIR"
echo

echo "=== STAGE 1/3: vision cache (Bridge V2, no aug) ==="
python -m scripts.cache_vision \
    --dataset oxe \
    --oxe-dataset-id "$DATASET_ID" \
    --oxe-camera     "$CAMERA" \
    --cache-dir      "$CACHE_DIR" \
    --n-eps-cap      "$N_EPS_CAP" \
    --n-vis-aug 0
echo "=== STAGE 1 DONE: $(date) ==="
echo "  disk usage of cache:"
du -sh "$CACHE_DIR" 2>&1 | head -1
df -h . 2>&1 | tail -2
echo

echo "=== STAGE 2/3: CQ-VAE training (Bridge 7-DoF actions) ==="
python -m scripts.train_cqvae \
    --dataset oxe \
    --oxe-dataset-id "$DATASET_ID" \
    --action-dim 7 \
    --steps 8000 --batch-size 32 \
    --n-eps-cap "$N_EPS_CAP" \
    --ckpt-path "$VAE_CKPT"
echo "=== STAGE 2 DONE: $(date) ==="
echo

echo "=== STAGE 3/3: policy training (v5 recipe, 20k steps) ==="
python -m scripts.train_policy \
    --dataset oxe \
    --oxe-dataset-id "$DATASET_ID" \
    --oxe-camera     "$CAMERA" \
    --steps 20000 \
    --batch-size 2 --grad-accum 4 --num-workers 2 \
    --vae-ckpt   "$VAE_CKPT" \
    --cache-dir  "$CACHE_DIR" \
    --ckpt-path  "$POL_CKPT" \
    --log-every 200 --probe-every 1000 --ckpt-every 1000
echo "=== STAGE 3 DONE: $(date) ==="
echo

echo "=== ALL DONE: $(date) ==="
echo "  final disk: $(df -h . | tail -1 | awk '{print $5, "used,", $4, "free"}')"
echo "  cache:   $CACHE_DIR"
echo "  vae:     $VAE_CKPT"
echo "  policy:  $POL_CKPT"
