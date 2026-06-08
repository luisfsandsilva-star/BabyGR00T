#!/usr/bin/env bash
# agibot_fp_v5: v4 (7 tasks, ScaleNorm, APR — all validated, broke the 2.30 floor → 2.139) PLUS the
# two one-flag drift fixes targeting σ_g at the root:
#   --weight-decay 1e-2 (10x, bounds weight norms ⇒ bounds σ_g)
#   --g-input-noise 0.02 (Bishop-1995 input noise ≡ explicit ‖J_g‖/Lipschitz penalty)
# Goal: keep σ_g low ⇒ learned α stays large (≳0.4) ⇒ fixed point stays reachable (resid small)
# ⇒ possibly push val below 2.139. (Nesterov accel = separate v6 change to _inner/_outer, not here.)
set -euo pipefail
cd /home/research/Projects/BBGr/BabyGR00T
PY=/home/research/Projects/BBGr/.venv/bin/python
nohup $PY -m scripts.train_oxe \
  --vae-dir data/ckpts --t5-cache data/cache/t5_agibot.pt --image-var data/cache/image_var_global.pt \
  --no-shared-vae --only-robots agibot --vision cnn --img-size 224 --no-vae --output-scalenorm \
  --update-mode damped --layerscale-init 0.1 --one-step-grad \
  --dim 288 --depth 3 --heads 8 --kv-heads 2 --ff-hidden 1152 --H-outer 2 --h-max 6 --L-inner 5 \
  --policy-dropout 0.1 --cnn-dropout 0.1 --state-noise 0.005 --action-noise 0.0 --apr-prob 0.5 \
  --weight-decay 1e-2 --g-input-noise 0.02 --no-revive \
  --chunk-stride 2 --batch-size 512 --num-workers 56 --lr 4e-3 --lr-warmup-frac 0.10 \
  --steps 50000 --grad-clip-max 100 \
  --val-probe-every 2000 --fp-probe-every 4000 --val-n 1152 --log-every 500 \
  --val-early-stop-patience 12 --val-min-delta 0.003 \
  --train-eps-file data/splits/agibot_train_eps.json --val-eps-file data/splits/agibot_val_eps.json \
  --ckpt-path data/ckpts/agibot_fp_v5.pt --val-best-path data/ckpts/agibot_fp_v5_bestval.pt \
  > data/logs/agibot_fp_v5.log 2>&1 &
echo "v5 (wd 1e-2 + g-input-noise 0.02) launched PID $! -> data/logs/agibot_fp_v5.log"
