#!/bin/bash
# BridgeData V2 (OXE) overnight pipeline — VQ-VAE variant.
#
# Companion to overnight.sh (which uses the 3-level CQ-VAE). This script
# trains the single-level VQ-VAE on Bridge then a policy on top of it,
# so the two checkpoints can be compared head-to-head:
#
#   oxe_strm_v5.pt        (policy trained with CQ-VAE codebook)
#   oxe_strm_v5_vqvae.pt  (policy trained with VQ-VAE codebook)
#
# Reuses the existing oxe_vision_cache/ (vision features don't depend on
# the action codebook). If you haven't cached vision yet, run overnight.sh
# first or do Stage 1 standalone.
set -e
cd "$(dirname "$0")"

DATASET_ID=IPEC-COMMUNITY/bridge_orig_lerobot
CAMERA=observation.images.image_0
N_EPS_CAP=350
CACHE_DIR=oxe_vision_cache
VAE_CKPT=oxe_vqvae.pt
POL_CKPT=oxe_strm_v5_vqvae.pt

echo "=== START: $(date) ==="
echo "  dataset:   $DATASET_ID"
echo "  vae kind:  vqvae (single-level conv, no skip)"
echo "  n_eps_cap: $N_EPS_CAP"
echo

if [ ! -d "$CACHE_DIR" ] || [ ! -f "$CACHE_DIR/meta.json" ]; then
    echo "ERROR: $CACHE_DIR/meta.json missing — run overnight.sh first or"
    echo "       cache vision features with scripts.cache_vision."
    exit 1
fi
echo "=== reusing existing vision cache ($CACHE_DIR) ==="
du -sh "$CACHE_DIR" 2>&1 | head -1
echo

echo "=== STAGE 1/2: VQ-VAE training (single-level, Bridge 7-DoF) ==="
python -m scripts.train_vqvae \
    --dataset oxe \
    --oxe-dataset-id "$DATASET_ID" \
    --action-dim 7 \
    --steps 8000 --batch-size 32 \
    --n-eps-cap "$N_EPS_CAP" \
    --ckpt-path "$VAE_CKPT"
echo "=== STAGE 1 DONE: $(date) ==="
echo

echo "=== STAGE 2/2: policy training (v5 recipe, 20k steps, VQ-VAE codebook) ==="
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
echo "=== STAGE 2 DONE: $(date) ==="
echo

echo "=== ALL DONE: $(date) ==="
echo "  cache:        $CACHE_DIR"
echo "  vqvae:        $VAE_CKPT"
echo "  policy:       $POL_CKPT"
echo "  compare against: oxe_vae_revin.pt + oxe_strm_v5.pt"
