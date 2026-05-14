#!/bin/bash
set -e
cd /home/alex/Desktop/BabyGR00T
echo "=== STAGE 3/3 RESUME: policy training (v5 recipe, 20k steps) ==="
echo "=== START: $(date) ==="
python -m scripts.train_policy \
    --dataset oxe \
    --oxe-dataset-id IPEC-COMMUNITY/bridge_orig_lerobot \
    --oxe-camera     observation.images.image_0 \
    --steps 20000 \
    --batch-size 2 --grad-accum 4 --num-workers 2 \
    --vae-ckpt   oxe_vae_revin.pt \
    --cache-dir  oxe_vision_cache \
    --ckpt-path  oxe_strm_v5.pt \
    --log-every 200 --probe-every 1000 --ckpt-every 1000
echo "=== ALL DONE: $(date) ==="
