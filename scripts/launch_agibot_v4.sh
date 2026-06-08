#!/usr/bin/env bash
# agibot_fp_v4: BIG run — 7 tasks (3283 eps, 6.4x baseline), non-VAE deterministic STRMPolicy.
# Improvements vs v3nv: batch 512 + lr 4e-3 (linear scaling, smoother grads) + 56 workers
# (fill the 40% util dips) + val-n 1152 (de-noise the probes that kept whipsawing us).
# Regularization/aug already on: visual aug (strong_aug default), dropout 0.1, modality-drop 0.3,
# state-noise 0.005, h_max 6. APR texture-aug OFF by default (set APR=0.5 to enable for sim2real).
# ScaleNorm: SCALENORM=1 adds --output-scalenorm (well-conditioned fixed point; decide from v3sn A/B).
set -euo pipefail
cd /home/research/Projects/BBGr/BabyGR00T
PY=/home/research/Projects/BBGr/.venv/bin/python
SCALENORM="${SCALENORM:-1}"          # 1 = include output ScaleNorm (pending A/B verdict)
APR="${APR:-0.0}"                    # >0 = APR texture randomization (sim2real); 0 = off
EXTRA=""
[ "$SCALENORM" = "1" ] && EXTRA="$EXTRA --output-scalenorm"
TAG="v4"; [ "$SCALENORM" = "1" ] && TAG="${TAG}sn"
nohup $PY -m scripts.train_oxe \
  --vae-dir data/ckpts --t5-cache data/cache/t5_agibot.pt --image-var data/cache/image_var_global.pt \
  --no-shared-vae --only-robots agibot --vision cnn --img-size 224 --no-vae $EXTRA \
  --update-mode damped --layerscale-init 0.1 --one-step-grad \
  --dim 288 --depth 3 --heads 8 --kv-heads 2 --ff-hidden 1152 --H-outer 2 --h-max 6 --L-inner 5 \
  --policy-dropout 0.1 --cnn-dropout 0.1 --state-noise 0.005 --action-noise 0.0 --apr-prob "$APR" \
  --chunk-stride 2 --batch-size 512 --num-workers 56 --lr 4e-3 --lr-warmup-frac 0.10 \
  --steps 50000 --grad-clip-max 100 \
  --val-probe-every 2000 --fp-probe-every 4000 --val-n 1152 --log-every 500 \
  --val-early-stop-patience 12 --val-min-delta 0.003 \
  --train-eps-file data/splits/agibot_train_eps.json --val-eps-file data/splits/agibot_val_eps.json \
  --ckpt-path data/ckpts/agibot_fp_${TAG}.pt --val-best-path data/ckpts/agibot_fp_${TAG}_bestval.pt \
  > data/logs/agibot_fp_${TAG}.log 2>&1 &
echo "v4 (${TAG}, SCALENORM=$SCALENORM APR=$APR) launched PID $! -> data/logs/agibot_fp_${TAG}.log"
