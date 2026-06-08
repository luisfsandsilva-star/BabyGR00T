#!/bin/bash
# v3 CNN-only texture-invariance recovery finetune.
#   Resume from v3 best (92.97% val EMA), FREEZE everything except the EfficientCNN,
#   train under APR (FFT amplitude randomizer) + MixStyle (channel-stat randomizer)
#   so the CNN stops relying on real-photo texture → sim no longer OOD.
#   Objective-aligned early-stop on sim→real CNN-feature kNN ratio (OOD recovery).
#   Architecture flags identical to v3 (state_dicts must load). Fresh step + momentum.
#
# Finetunes a COPY (oxe_policy_widowx_v3_cnnft.pt); v3 best is left untouched.
cd /home/research/Projects/BBGr/BabyGR00T
SRC=data/ckpts/oxe_policy_v14_widowx_v3_resumed_best.pt
DST=data/ckpts/oxe_policy_widowx_v3_cnnft.pt
cp -n "$SRC" "$DST"
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
    --steps 10000 --batch-size 512 --lr 2.0e-3 \
    --lr-schedule cosine --lr-warmup-frac 0.10 \
    --cnn-pe --weight-decay 2e-3 --dropout-prob 0.2 --cnn-dropout 0.2 \
    --state-noise 0.05 --strong-aug --ema-decay 0.999 \
    --grad-clip-max 100.0 --num-workers 12 --prefetch-factor 4 \
    --log-every 100 --ckpt-every 1000 \
    --resume --reset-step --reset-opt \
    --freeze-except-cnn \
    --apr-prob 0.5 --apr-eta-max 1.0 --mixstyle-p 0.5 \
    --ood-probe-every 250 --ood-early-stop-patience 8 --ood-min-delta 0.01 \
    --ood-sim /tmp/sim_frames.npy --ood-real /tmp/real_frames.npy \
    --ood-best-path data/ckpts/oxe_policy_widowx_v3_cnnft_bestood.pt \
    --ckpt-path "$DST" \
    > /tmp/widowx_v3_cnnft.log 2>&1 &
echo "v3 CNN-only finetune launched, PID $! (freeze-except-cnn, APR+MixStyle, 10k, OOD early-stop)"
disown
