#!/bin/bash
# v14_widowx_v3 RESUME from step 6000 best ckpt (val_acc 34.41%).
#
# Changes vs v3 first launch:
#   --num-workers 12              (was 24) — RAM OOM root cause: 24 workers × 13GB T5
#                                              cache copies via Python refcount CoW = 150GB.
#                                              12 workers keeps total RAM ~80GB.
#   --H-outer 3 --h-max 3                  — kept at 3 (matches v3 first launch which was
#                                              healthy at H=3 thanks to clamp+no-Tikhonov).
#   --resume                                — pick up from step 6000 (val_acc 34.41%).
#   No --reset-opt                          — keep momentum buffers; training was healthy.
#
# Everything else unchanged: clamp 2.0, σ_g=0, no-revive, BS=512, LR cosine continues.
cd /home/research/Projects/BBGr/BabyGR00T
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup /home/research/Projects/BBGr/.venv/bin/python -u -m scripts.train_oxe \
    --t5-cache data/cache/t5_text_cache_paraphrased.pt \
    --image-var data/cache/image_var_global.pt \
    --no-vae --no-shared-vae \
    --only-robots widowx \
    --update-mode damped --alpha-parametrization sigmoid --alpha-per-dim \
    --no-revive \
    --n-emb-prefix 16 --label-smoothing 0.10 \
    --mask-sampler cosine \
    --chunk-stride 4 --action-noise 0.10 --g-input-noise 0.0 \
    --use-paraphrase-sampling \
    --dim 512 --depth 2 --heads 8 --kv-heads 2 --ff-hidden 2048 \
    --L-inner 5 --H-outer 3 --h-max 3 \
    --steps 30000 --batch-size 512 --lr 8.97e-3 \
    --lr-schedule cosine --lr-warmup-frac 0.10 \
    --cnn-pe --weight-decay 2e-3 --dropout-prob 0.2 --cnn-dropout 0.2 \
    --state-noise 0.05 --strong-aug --ema-decay 0.999 \
    --grad-clip-max 100.0 --num-workers 12 --prefetch-factor 4 \
    --log-every 200 --ckpt-every 1000 \
    --resume \
    --ckpt-path data/ckpts/oxe_policy_v14_widowx_v3.pt \
    > /tmp/v14_widowx_v3_resumed.log 2>&1 &
echo "v14_widowx_v3 RESUMED launched, PID $! (NW=12, H=3, resume from step 6000)"
disown
