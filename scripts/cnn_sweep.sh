#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# UNATTENDED CNN-architecture sweep for the CNN + frozen-T5 vision conditioning.
#
# For each CNN variant it runs the 43-chunk OVERFIT sanity check and logs whether
# the model can now LEARN the vision→action mapping (train loss should crater
# well below the ~4.0 marginal plateau and accuracy climb — the test the old
# InternVL/resampler model failed). Sequential, logs a results table, no
# babysitting:  nohup bash scripts/cnn_sweep.sh > cnn_sweep.log 2>&1 &
#
# PREREQUISITE (gating): the integrated CNN+T5 trainer must exist —
#   scripts/train_policy with:  --vision cnn  --text t5cache
#   + a raw-frame cache (--frame-cache) and the T5 cache (--t5-cache),
#   + CNN-config flags: --cnn-dims --cnn-depths --cnn-expand --cnn-norm
#                       --cnn-out-dim --cnn-pe
# This guard checks for that support and exits cleanly if the integration
# isn't wired yet, so the queued sweep never runs against a stale pipeline.
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PY=/home/research/Projects/BBGr/.venv/bin/python
RESULTS=cnn_sweep_results.txt
STEPS=${STEPS:-1200}        # ~220 epochs over the 43-chunk subset
CAP=${CAP:-24}             # episodes (~43 chunks)
FRAME_CACHE=${FRAME_CACHE:-data/cache/oxe_frames_224}
T5_CACHE=${T5_CACHE:-data/cache/t5_text_cache.pt}

# ── prerequisite guard ──
if ! "$PY" -m scripts.train_cnn_policy --help 2>/dev/null | grep -q -- '--cnn-dims'; then
  echo "GATED: scripts.train_cnn_policy not available — CNN+T5 integration missing."
  exit 3
fi
if [ ! -e "$T5_CACHE" ]; then
  echo "GATED: missing $T5_CACHE (T5 cache — run scripts.cache_t5)."
  exit 3
fi

# variant name : extra CNN flags  (scale-up + PE + norm + expand)
ORDER="v0_baseline v1_expand4 v2_layernorm v3_pe v4_scaled v5_kitchen_sink"
flags_for() { case "$1" in
  v0_baseline)    echo "--cnn-dims 64,128,256,384  --cnn-depths 2,2,4,2 --cnn-out-dim 384 --cnn-expand 3 --cnn-norm scalenorm" ;;
  v1_expand4)     echo "--cnn-dims 64,128,256,384  --cnn-depths 2,2,4,2 --cnn-out-dim 384 --cnn-expand 4 --cnn-norm scalenorm" ;;
  v2_layernorm)   echo "--cnn-dims 64,128,256,384  --cnn-depths 2,2,4,2 --cnn-out-dim 384 --cnn-expand 3 --cnn-norm layernorm" ;;
  v3_pe)          echo "--cnn-dims 64,128,256,384  --cnn-depths 2,2,4,2 --cnn-out-dim 384 --cnn-expand 3 --cnn-norm scalenorm --cnn-pe" ;;
  v4_scaled)      echo "--cnn-dims 96,192,384,512  --cnn-depths 3,3,9,3 --cnn-out-dim 512 --cnn-expand 3 --cnn-norm scalenorm" ;;
  v5_kitchen_sink)echo "--cnn-dims 96,192,384,512  --cnn-depths 3,3,9,3 --cnn-out-dim 512 --cnn-expand 4 --cnn-norm layernorm --cnn-pe" ;;
esac; }

printf "%-16s | %-10s | %-8s | %-9s | %s\n" variant final_loss train_acc steps verdict > "$RESULTS"
echo "=== CNN SWEEP START $(date)  (steps=$STEPS cap=$CAP) ==="

for name in $ORDER; do
  flags=$(flags_for "$name"); ckpt="data/ckpts/sweep_${name}.pt"; log="sweep_${name}.log"
  rm -f "$ckpt"
  echo "=== [$(date)] $name : $flags ==="
  CUDA_VISIBLE_DEVICES=0 "$PY" -u -m scripts.train_cnn_policy \
    --dim 1728 --depth 2 --L-inner 5 --H-outer 2 --h-max 4 \
    --steps "$STEPS" --batch-size 8 --lr 9.5e-4 --n-eps-cap "$CAP" $flags \
    --vae-ckpt data/ckpts/oxe_vqvae_1800ep_16k.pt --t5-cache "$T5_CACHE" \
    --ckpt-path "$ckpt" --log-every 50 > "$log" 2>&1 || echo "  (run errored — see $log)"
  last=$(grep -E "step +[0-9]+/" "$log" | tail -1)
  loss=$(echo "$last" | grep -oE "loss=[0-9.]+" | cut -d= -f2)
  acc=$(echo "$last"  | grep -oE "acc=[0-9.]+%" | head -1)
  st=$(echo "$last" | grep -oE "step +[0-9]+" | grep -oE "[0-9]+")
  # verdict: memorization = loss well below the ~4.0 marginal plateau
  verdict="no-learn"
  [ -n "$loss" ] && awk "BEGIN{exit !($loss < 2.5)}" 2>/dev/null && verdict="LEARNS✓"
  [ -n "$loss" ] && awk "BEGIN{exit !($loss < 3.5 && $loss >= 2.5)}" 2>/dev/null && verdict="partial"
  printf "%-16s | %-10s | %-8s | %-9s | %s\n" "$name" "${loss:-NA}" "${acc:-NA}" "${st:-NA}" "$verdict" >> "$RESULTS"
  echo "  -> $name: loss=${loss:-NA} acc=${acc:-NA} verdict=$verdict"
done

echo "=== CNN SWEEP DONE $(date) ==="
echo "results -> $RESULTS"; cat "$RESULTS"
