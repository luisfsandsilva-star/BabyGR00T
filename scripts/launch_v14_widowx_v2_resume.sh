#!/bin/bash
# v14_widowx_v2 RESUME from step 5000 best ckpt (val_acc 18.30%):
#   - --resume                       reload weights from --ckpt-path
#   - --reset-opt                    fresh momentum buffers (old opt was in bad state
#                                    by step 7000: huge grads, ρ_H collapsed)
#   - --no-revive                    let ρ collapse naturally; revive's bounce-up
#                                    perturbs training (caused loss spikes when ρ_H
#                                    was effectively dead but the model had adapted)
#   - --ckpt-path points to a SEPARATE _resumed.pt file so _best.pt stays preserved.
# Same architecture/hparams as the first launch (BS, LR, σ, dropout, etc.).
# The cosine LR schedule continues from saved step (~5000) → step 30000 = 8.97e-4 floor.
cd /home/research/Projects/BBGr/BabyGR00T
: "${BS:?set BS}"
: "${LR:?set LR}"
: "${SIGMA:?set SIGMA}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup /home/research/Projects/BBGr/.venv/bin/python -u -m scripts.train_oxe \
    --t5-cache data/cache/t5_text_cache_paraphrased.pt \
    --image-var data/cache/image_var_global.pt \
    --no-vae --no-shared-vae \
    --only-robots widowx \
    --update-mode damped --alpha-parametrization sigmoid --alpha-per-dim \
    --no-revive \
    --n-emb-prefix 16 --label-smoothing 0.10 \
    --mask-sampler cosine \
    --chunk-stride 4 --action-noise 0.10 --g-input-noise ${SIGMA} \
    --use-paraphrase-sampling \
    --dim 512 --depth 2 --heads 8 --kv-heads 2 --ff-hidden 2048 \
    --L-inner 5 --H-outer 3 --h-max 3 \
    --steps 30000 --batch-size ${BS} --lr ${LR} \
    --lr-schedule cosine --lr-warmup-frac 0.10 \
    --cnn-pe --weight-decay 2e-3 --dropout-prob 0.2 --cnn-dropout 0.2 \
    --state-noise 0.05 --strong-aug --ema-decay 0.999 \
    --grad-clip-max 100.0 --num-workers 24 --prefetch-factor 4 \
    --log-every 200 --ckpt-every 1000 \
    --resume --reset-opt \
    --ckpt-path data/ckpts/oxe_policy_v14_widowx_v2_resumed.pt \
    > /tmp/v14_widowx_v2_resumed.log 2>&1 &
echo "v14_widowx_v2 RESUMED launched, PID $! (BS=${BS}, LR=${LR}, σ=${SIGMA}, no-revive, reset-opt)"
disown
