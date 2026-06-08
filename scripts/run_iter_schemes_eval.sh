#!/usr/bin/env bash
# Drive v4's trained g through 3 fixed-point schemes (accumulator / damped / nesterov) on one batch.
# Answers: would undamped-with-decay (accumulator) converge to a TRUE fixed point, or only shrink its
# step? does damping make it possible? does nesterov accelerate? Plots data/logs/iter_schemes.png.
# Run AFTER v4 frees the GPU (it loads v4sn_bestval). Arch flags MUST match v4 or the ckpt won't load.
set -euo pipefail
cd /home/research/Projects/BBGr/BabyGR00T
CKPT="${1:-data/ckpts/agibot_fp_v4sn_bestval.pt}"
/home/research/Projects/BBGr/.venv/bin/python -m scripts.train_oxe \
  --vae-dir data/ckpts --t5-cache data/cache/t5_agibot.pt --image-var data/cache/image_var_global.pt \
  --no-shared-vae --only-robots agibot --vision cnn --img-size 224 --no-vae --output-scalenorm \
  --update-mode damped --layerscale-init 0.1 --one-step-grad \
  --dim 288 --depth 3 --heads 8 --kv-heads 2 --ff-hidden 1152 --H-outer 2 --h-max 6 --L-inner 5 \
  --chunk-stride 2 --num-workers 8 --val-probe-every 1 --val-n 96 \
  --resume --eval-iter-schemes --iter-n 40 --iter-beta 0.7 \
  --ckpt-path "$CKPT" \
  --train-eps-file data/splits/agibot_train_eps.json --val-eps-file data/splits/agibot_val_eps.json
