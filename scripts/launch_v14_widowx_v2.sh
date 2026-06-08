#!/bin/bash
# v14_widowx_v2: single-emb widowx test with HARDER regularization.
# Changes vs v14_widowx_v1:
#   - drop --cnn-film-by-emb        (dead weight: only 1 emb → constant γ/β)
#   - drop --per-emb-head           (single emb → 1 head, not 11)
#   - --dropout-prob 0.2            (was 0.1)
#   - --cnn-dropout 0.2             (was 0.1)
#   - --action-noise 0.10           (was 0.02) — denoising-AE data aug
#   - --g-input-noise ${SIGMA:-0.0} — Bishop-1995 Tikhonov on g (sweep first!)
#   - --state-noise 0.05            (was 0.02)
#   - --label-smoothing 0.10        (was 0.05)
#   - --weight-decay 2e-3           (was 1e-3) — applied to 2D params only (LARS fix)
#   - --steps 30000                 (was 100000 → 530 epochs; now ≈160 epochs)
#   - --batch-size ${BS:-?}         (set after BS scan)
#   - --lr ${LR:-?}                 (set after LR range test)
# AMP bf16 on by default. Same VAE (oxe_vqvae_widowx.pt, val_mse=0.0055).
# Goal: stay in the regularization regime; let sentinel kill on val-ACC plateau.
cd /home/research/Projects/BBGr/BabyGR00T
: "${BS:?set BS (from find_bs_lr.py Phase 1)}"
: "${LR:?set LR (from find_bs_lr.py Phase 2)}"
: "${SIGMA:?set SIGMA (from find_bs_lr.py Phase 3)}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup /home/research/Projects/BBGr/.venv/bin/python -u -m scripts.train_oxe \
    --t5-cache data/cache/t5_text_cache_paraphrased.pt \
    --image-var data/cache/image_var_global.pt \
    --no-vae --no-shared-vae \
    --only-robots widowx \
    --update-mode damped --alpha-parametrization sigmoid --alpha-per-dim \
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
    --ckpt-path data/ckpts/oxe_policy_v14_widowx_v2.pt \
    > /tmp/v14_widowx_v2.log 2>&1 &
echo "v14_widowx_v2 launched, PID $! (BS=${BS}, LR=${LR}, σ=${SIGMA})"
disown
