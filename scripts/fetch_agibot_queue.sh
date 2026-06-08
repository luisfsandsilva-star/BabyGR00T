#!/usr/bin/env bash
set -uo pipefail
cd /home/research/Projects/BBGr/BabyGR00T
for T in 356 327 367; do
  echo "===== QUEUE: task $T  $(date -u +%H:%M:%S) ====="
  bash scripts/fetch_agibot_task.sh "$T" || echo "[$T] FAILED rc=$?"
  rm -rf ~/.cache/huggingface/hub/datasets--agibot-world--AgiBotWorld-Alpha/blobs/* data/agibot_raw/_tars/* 2>/dev/null || true
  echo "[$T] disk: $(df -h . | tail -1 | awk '{print $4" free"}')"
done
echo "===== QUEUE DONE  $(date -u +%H:%M:%S) ====="
ls -d data/oxe/agibot_task* 2>/dev/null
