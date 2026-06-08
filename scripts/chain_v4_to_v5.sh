#!/usr/bin/env bash
# Wait for v4's step-38000 val-probe. If it sets a NEW best (val<2.139) → plateau NOT confirmed,
# leave v4 running. If it's a noimp (no new best) → plateau confirmed → kill v4, launch v5.
set -uo pipefail
cd /home/research/Projects/BBGr/BabyGR00T
LOG=data/logs/agibot_fp_v4sn.log
CH=data/logs/chain_v5.log
echo "=== chain_v4_to_v5 start $(date -u +%H:%M:%S) — waiting for v4 step-38000 probe ===" >> "$CH"
while ! grep -q "val-probe\] step 38000:" "$LOG" 2>/dev/null; do
  pgrep -f "[p]ython -m scripts.train_oxe" >/dev/null || { echo "v4 exited before 38k $(date -u +%H:%M:%S)" >> "$CH"; break; }
  sleep 30
done
PROBE=$(grep "val-probe\] step 38000:" "$LOG" | tail -1)
echo "38k probe: $PROBE" >> "$CH"
if echo "$PROBE" | grep -q "best, saved"; then
  echo "=== NEW BEST at 38k → NOT a plateau; leaving v4 running, NOT launching v5 $(date -u +%H:%M:%S) ===" >> "$CH"
else
  echo "=== noimp at 38k → PLATEAU CONFIRMED; killing v4, launching v5 $(date -u +%H:%M:%S) ===" >> "$CH"
  pkill -f "[p]ython -m scripts.train_oxe"; sleep 10
  pgrep -f "[p]ython -m scripts.train_oxe" >/dev/null && { pkill -9 -f "[p]ython -m scripts.train_oxe"; sleep 5; }
  bash scripts/launch_agibot_v5.sh >> "$CH" 2>&1
  echo "=== v5 launched $(date -u +%H:%M:%S) ===" >> "$CH"
fi
