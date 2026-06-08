#!/bin/bash
# v14_widowx_v3: clamp + no-Tikhonov retest (per Tikhonov placement empirical test)
#
# Changes vs v2:
#   - --g-input-noise 0          (was 0.03) — empirical test showed σ>0 grows Lipschitz
#                                              at every placement we tested; σ=0 was best.
#                                              Model self-regulates Jacobian without it.
#   - LARS trust_max=2.0 (default)        — clamp in optimizer prevents the exponential
#                                              weight runaway that drove Lip→1.28M before.
#   - --no-revive                          — ρ_H collapse OK, revive's bounce destabilizes.
#
# Everything else unchanged: BS=512, LR=8.97e-3 (from finder), dim=512, depth=2,
# H_outer=3 (kept; user wants to verify if clamp+no-Tikhonov alone fixes things).
# Fresh init (not resume) — previous resume ckpts had ||W|| in the unstable regime.
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
    --grad-clip-max 100.0 --num-workers 24 --prefetch-factor 4 \
    --log-every 200 --ckpt-every 1000 \
    --ckpt-path data/ckpts/oxe_policy_v14_widowx_v3.pt \
    > /tmp/v14_widowx_v3.log 2>&1 &
echo "v14_widowx_v3 launched, PID $! (clamp=2.0 in optimizer, σ_g=0, no-revive)"
disown
