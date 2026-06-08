#!/bin/bash
# Stage 3 of the pipeline: train the ~100M VAE-TRM policy (STRMPolicyVAE) on the
# cached BridgeData V2 features. Scaled to ~100M params in the TRM ONLY via width
# (dim=1728, depth=2 — the tiny 2-sub-block net, just wide). 'The rest'
# (aggregator + resampler) stays at the baseline vis_dim=768 and a thin
# Linear(768->1728) projects the vision KV up, so the resampler does NOT balloon
# with the TRM width (it would otherwise grow as dim² to ~290M).
#
# fp16 AMP (the z_H/z_L accumulators stay fp32 — seeded from fp32 embeddings — so
# the unbounded additive logits never overflow). Optimized flash/mem-efficient
# SDPA kept. Throughput: num_workers=0 + lru_size=16 (cached hidden tensors are
# ~95 MB; workers would serialize them over IPC every step). Atomic checkpoints
# every 250 steps; auto-resumes from $POL_CKPT if present.
#
# Paths are env-configurable. Default data root is repo-local ./data; point
# BABYGROOT_DATA at a FAST LOCAL DISK (NVMe/ext4) — training is disk-I/O-bound on
# the cache, so never put it on a network/FUSE mount. Examples:
#   BABYGROOT_DATA=/mnt/nvme/babygroot bash launch_strm_vae_bridge.sh
#   BABYGROOT_DATA=/media/alex/D:/babygroot bash launch_strm_vae_bridge.sh  # this box
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="${BABYGROOT_DATA:-$PWD/data}"
VAE_CKPT="${VAE_CKPT:-$DATA_ROOT/ckpts/oxe_vqvae_1800ep_16k.pt}"
CACHE_DIR="${CACHE_DIR:-$DATA_ROOT/cache/oxe_vision_cache_v2}"
POL_CKPT="${POL_CKPT:-$DATA_ROOT/ckpts/oxe_strm_vae100m.pt}"
mkdir -p "$(dirname "$POL_CKPT")"

RESUME_ARG=""
if [ -f "$POL_CKPT" ]; then
    RESUME_ARG="--resume $POL_CKPT"
    echo "  (found existing ckpt — will resume: $POL_CKPT)"
fi

REVIVE_ARG=""
if [ -n "${POL_REVIVE:-}" ]; then
    REVIVE_ARG="--revive --revive-thresh ${REVIVE_THRESH:-0.02} \
--revive-patience ${REVIVE_PATIENCE:-200} --revive-to ${REVIVE_TO:-0.1} \
--revive-cooldown ${REVIVE_COOLDOWN:-400} --revive-decay ${REVIVE_DECAY:-1.0}"
    echo "  (ρ revival ON: thresh=${REVIVE_THRESH:-0.02} patience=${REVIVE_PATIENCE:-200} to=${REVIVE_TO:-0.1} cooldown=${REVIVE_COOLDOWN:-400} decay=${REVIVE_DECAY:-1.0})"
fi

echo "=== VAE-TRM 100M on Bridge: dim=1728 depth=2 vis_dim=768, vae-latent, fp16 ==="
echo "=== START: $(date) ==="
echo "  VAE:   $VAE_CKPT"
echo "  cache: $CACHE_DIR"
echo "  ckpt:  $POL_CKPT"

python -u -m scripts.train_policy \
    --dataset oxe \
    --oxe-dataset-id IPEC-COMMUNITY/bridge_orig_lerobot \
    --oxe-camera     observation.images.image_0 \
    --vae-latent --beta 1e-3 --free-bits 0.1 \
    --amp-dtype fp16 \
    --dim 1728 --vis-dim 768 --depth 2 \
    --L-inner 5 --H-outer 4 --h-max 12 \
    --steps "${POL_STEPS:-20000}" \
    --batch-size "${POL_BATCH:-3}" --grad-accum "${POL_ACCUM:-3}" \
    --num-workers "${POL_WORKERS:-0}" --lru-size "${POL_LRU:-16}" \
    --lr 9.5e-4 \
    --n-probe 24 \
    $RESUME_ARG \
    $REVIVE_ARG \
    --vae-ckpt   "$VAE_CKPT" \
    --cache-dir  "$CACHE_DIR" \
    --ckpt-path  "$POL_CKPT" \
    --log-every 200 --probe-every 1000 --ckpt-every 250

echo "=== DONE: $(date) ==="
