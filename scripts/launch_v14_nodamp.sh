#!/bin/bash
# v14b: α=1 (pure Picard / no damping) ablation. Same as v14 except --rho-fixed 10
# (sigmoid(10)≈1.0). Both ρ_L and ρ_H are frozen, no revive. Tests whether the
# direct recurrence (z ← g(z+context) repeatedly) behaves better than the gated/damped
# update at L_inner=5, H_outer=3.
cd /home/research/Projects/BBGr/BabyGR00T
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup /home/research/Projects/BBGr/.venv/bin/python -u -m scripts.train_oxe \
    --t5-cache data/cache/t5_text_cache_paraphrased.pt \
    --image-var data/cache/image_var_global.pt \
    --no-vae --no-shared-vae \
    --update-mode damped --alpha-parametrization sigmoid --alpha-per-dim \
    --rho-fixed 10.0 \
    --n-emb-prefix 16 --per-emb-head --label-smoothing 0.05 \
    --mask-sampler cosine \
    --chunk-stride 4 --action-noise 0.02 \
    --use-paraphrase-sampling \
    --cnn-film-by-emb \
    --dim 1280 --depth 3 --heads 20 --kv-heads 5 --ff-hidden 10240 \
    --L-inner 5 --H-outer 3 --h-max 3 \
    --steps 200000 --batch-size 256 --lr 2e-3 \
    --lr-schedule cosine --lr-warmup-frac 0.10 \
    --cnn-pe --weight-decay 1e-3 --dropout-prob 0.1 --cnn-dropout 0.1 \
    --state-noise 0.02 --strong-aug --ema-decay 0.999 \
    --grad-clip-max 100.0 --num-workers 8 \
    --log-every 200 --ckpt-every 1000 \
    --ckpt-path data/ckpts/oxe_policy_v14b_nodamp.pt \
    --exclude-datasets \
        iamlab_cmu_pickup_insert_lerobot \
        toto_lerobot \
        cmu_play_fusion_lerobot \
        nyu_franka_play_dataset_lerobot \
        stanford_hydra_dataset_lerobot \
        ucsd_kitchen_dataset_lerobot \
    > /tmp/v14b_nodamp.log 2>&1 &
echo "v14b (α=1 pure Picard) launched, PID $!"
disown
