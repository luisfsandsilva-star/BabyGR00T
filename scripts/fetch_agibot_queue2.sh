#!/usr/bin/env bash
# Fetch tasks 410 + 359, all-intra re-encode each, drop raw. (max-diversity set C, ~6.4x data)
set -uo pipefail
cd /home/research/Projects/BBGr/BabyGR00T
PY=/home/research/Projects/BBGr/.venv/bin/python
for T in 410 359; do
  echo "===== QUEUE2: task $T  $(date -u +%H:%M:%S) ====="
  bash scripts/fetch_agibot_task.sh "$T" || { echo "[$T] FETCH FAILED rc=$?"; continue; }
  echo "[$T] === all-intra re-encode ==="
  $PY -m scripts.reencode_agibot_allintra \
      --glob "data/oxe/agibot_task$T/videos/**/episode_*.mp4" --workers 56 --crf 20 --size 224 \
      || echo "[$T] REENCODE FAILED rc=$?"
  rm -rf "data/agibot_raw/obs$T" ~/.cache/huggingface/hub/datasets--agibot-world--AgiBotWorld-Alpha/blobs/* data/agibot_raw/_tars/* 2>/dev/null || true
  echo "[$T] DONE  kept: $(du -sh data/oxe/agibot_task$T 2>/dev/null | cut -f1)  disk: $(df -h . | tail -1 | awk '{print $4}') free"
done
echo "===== QUEUE2 DONE  $(date -u +%H:%M:%S) ====="
# regenerate global multi-task splits across ALL agibot tasks
$PY -m scripts.make_agibot_splits --val-frac 0.15 --seed 0
