#!/bin/bash
# Stage-3 VRAM probe: drive the REAL train_policy (via launch_strm_vae_bridge.sh)
# for a few steps at increasing batch sizes, sampling peak GPU memory and
# catching OOM. Reports the largest batch that fits with headroom. Uses a
# throwaway ckpt so it never touches the real run. Run between stage 2 and 3,
# while the GPU is otherwise idle.
#
#   ./probe_batch.sh "6 12 16 24 32"
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDS="${1:-6 12 16 24 32}"
TMP_CKPT="/tmp/pol_probe.pt"
TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

echo "GPU total ${TOTAL} MiB | probing batches: $CANDS"
BEST=""
for b in $CANDS; do
  rm -f "$TMP_CKPT"
  # few steps, grad-accum 1 (pure batch footprint), throwaway ckpt, no probes/logs
  POL_BATCH="$b" POL_ACCUM=1 POL_STEPS=3 POL_CKPT="$TMP_CKPT" \
    timeout 420 bash launch_strm_vae_bridge.sh > "/tmp/pol_probe_${b}.log" 2>&1 &
  PID=$!
  peak=0; oom=0
  while kill -0 "$PID" 2>/dev/null; do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$u" -gt "$peak" ] && peak=$u
    grep -qiE "OutOfMemory|CUDA out of memory" "/tmp/pol_probe_${b}.log" && oom=1 && kill "$PID" 2>/dev/null
    sleep 2
  done
  wait "$PID" 2>/dev/null; rc=$?
  grep -qiE "OutOfMemory|CUDA out of memory" "/tmp/pol_probe_${b}.log" && oom=1
  free=$(( TOTAL - peak ))
  if [ "$oom" -eq 1 ]; then
    echo "  batch ${b}: OOM (peak ~${peak} MiB)"; break
  elif [ "$rc" -ne 0 ]; then
    echo "  batch ${b}: run failed rc=$rc (see /tmp/pol_probe_${b}.log) — stopping"; break
  else
    echo "  batch ${b}: peak ${peak} MiB | free ${free} MiB OK"
    [ "$free" -ge 1024 ] && BEST="$b"
  fi
done
rm -f "$TMP_CKPT"
echo ""
echo "RECOMMENDED POL_BATCH=${BEST:-<none fit>}"
