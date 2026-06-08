#!/bin/bash
# Scaling sweep on Sudoku-Extreme (100k) — params/width, compute/L, outer-H axes.
# Sequential (1 GPU). Each run self-terminates (--early-stop), then fixed-test Extreme eval.
# All runs: EMA 0.999, bf16, compile, inner early-stop, stochastic depth.
cd /home/research/Projects/BBGr/BabyGR00T
PY=/home/research/Projects/BBGr/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA=data/cache/sudoku_extreme_100k.pt
RES=data/logs/sweep_results.csv
[ -f "$RES" ] || echo "axis,tag,dim,L,H,batch,params_M,ckpt_step,ss_cell,ss_solve,mg_solve" > "$RES"

run () {  # axis tag dim L H batch
  axis=$1; tag=$2; dim=$3; L=$4; H=$5; bs=$6
  ck=data/ckpt/sweep_${tag}.pt; log=data/logs/sweep_${tag}.log; ev=data/logs/sweep_${tag}_eval.log
  echo "=== [$tag] axis=$axis dim$dim L$L H$H bs$bs  $(date +%H:%M) ==="
  $PY -m scripts.train_sudoku_v2 --steps 50000 --early-stop 12 --batch $bs --dim $dim --depth 2 \
      --heads 8 --kv-heads 2 --ff $((dim*4)) --H $H --L $L --lr 1e-3 --cur-extra 20 \
      --data "$DATA" --inner-tol 0.01 --compile --ema-decay 0.999 --ckpt-path "$ck" --tag "$tag" --no-final-diag > "$log" 2>&1
  $PY -m scripts.eval_sudoku_extreme --ckpt "$ck" --n 5000 > "$ev" 2>&1
  pm=$(grep -oP 'params: \K[0-9.]+' "$log" | head -1)
  st=$(grep -oP 'ckpt step \K[0-9]+' "$ev" | head -1)
  ssc=$(grep -oP 'single-shot: cell=\K[0-9.]+' "$ev" | head -1)
  sss=$(grep -oP 'single-shot:.*solve=\K[0-9.]+' "$ev" | head -1)
  mgs=$(grep -oP 'n_steps=16: cell=[0-9.]+%  solve=\K[0-9.]+' "$ev" | head -1)
  echo "$axis,$tag,$dim,$L,$H,$bs,$pm,$st,$ssc,$sss,$mgs" >> "$RES"
  echo "    -> ss_solve=$sss  mg_solve=$mgs  (params ${pm}M, step $st)"
}

# ── params / width axis (L16 H3) ──
run params d128 128 16 3 384
run params d192 192 16 3 384
run params d256 256 16 3 320     # baseline (also the EMA reference vs the no-EMA 100k run)
run params d384 384 16 3 224
run params d512 512 16 3 160
# ── compute / inner-L axis (dim256 H3) ──
run compute L8  256 8  3 320
run compute L32 256 32 3 320
run compute L64 256 64 3 320
# ── outer-H / single-loop axis (dim256 L16) ──
run outerH H1 256 16 1 320
run outerH H2 256 16 2 320
run outerH H4 256 16 4 288
echo "=== SWEEP DONE $(date +%H:%M) ==="; cat "$RES"
