#!/usr/bin/env bash
# agibot_fp_v3nv: NON-VAE (deterministic STRMPolicy) ~4x scale + de-regularized, multi-task, h_max=6.
# Diagnosis behind it: task354-alone was data-limited (val saturates ~32%/2.96, then a train-val
# gap opens = memorization). Fix = MORE TASKS (primary) + moderate capacity + cut the regularizers
# that fight fitting (dropout kept 0.1, state-noise low), KEEP visual augmentation.
# --no-vae: deterministic fixed-point TRM (no belief latent, no KL, no ‖z*‖ drift) — the intended arch.
# Run AFTER: fetch queue done (354,374,356,327,367 converted) + make_agibot_splits.py.
set -euo pipefail
cd /home/research/Projects/BBGr/BabyGR00T
PY=/home/research/Projects/BBGr/.venv/bin/python
nohup $PY -m scripts.train_oxe \
  --vae-dir data/ckpts --t5-cache data/cache/t5_agibot.pt --image-var data/cache/image_var_global.pt \
  --no-shared-vae --only-robots agibot --vision cnn --img-size 224 --no-vae \
  --update-mode damped --layerscale-init 0.1 --one-step-grad \
  --dim 288 --depth 3 --heads 8 --kv-heads 2 --ff-hidden 1152 \
  --H-outer 2 --h-max 6 --L-inner 5 \
  --policy-dropout 0.1 --cnn-dropout 0.1 --state-noise 0.005 --action-noise 0.0 \
  --chunk-stride 2 --batch-size 384 --num-workers 48 --lr 3e-3 --lr-warmup-frac 0.10 \
  --steps 100000 --grad-clip-max 100 \
  --val-probe-every 2000 --fp-probe-every 4000 --val-n 384 --log-every 500 \
  --val-early-stop-patience 12 --val-min-delta 0.003 \
  --train-eps-file data/splits/agibot_train_eps.json --val-eps-file data/splits/agibot_val_eps.json \
  --ckpt-path data/ckpts/agibot_fp_v3nv.pt --val-best-path data/ckpts/agibot_fp_v3nv_bestval.pt \
  > data/logs/agibot_fp_v3nv.log 2>&1 &
echo "v3nv (non-VAE) launched PID $!  -> data/logs/agibot_fp_v3nv.log"
