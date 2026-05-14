#!/bin/bash
set -e
cd /home/alex/Desktop/BabyGR00T
echo "=== START: $(date) ==="
echo

echo "=== STAGE 1/3: vision cache (OXE, no aug) ==="
python -m scripts.cache_vision \
    --dataset oxe \
    --oxe-dataset-id lerobot/svla_so101_pickplace \
    --oxe-camera observation.images.up \
    --cache-dir oxe_vision_cache \
    --n-vis-aug 0
echo "=== STAGE 1 DONE: $(date) ==="
echo

echo "=== STAGE 2/3: CQ-VAE training (OXE) ==="
python -m scripts.train_cqvae \
    --dataset oxe \
    --oxe-dataset-id lerobot/svla_so101_pickplace \
    --steps 8000 --batch-size 32 \
    --ckpt-path oxe_vae_revin.pt
echo "=== STAGE 2 DONE: $(date) ==="
echo

echo "=== STAGE 3/3: policy training (v5, OXE, 15k steps) ==="
python -m scripts.train_policy \
    --dataset oxe \
    --oxe-dataset-id lerobot/svla_so101_pickplace \
    --steps 15000 \
    --batch-size 2 --grad-accum 4 --num-workers 2 \
    --vae-ckpt oxe_vae_revin.pt \
    --cache-dir oxe_vision_cache \
    --ckpt-path oxe_strm_v5.pt \
    --log-every 200 --probe-every 1000 --ckpt-every 1000
echo "=== STAGE 3 DONE: $(date) ==="
echo

echo "=== ALL DONE: $(date) ==="
