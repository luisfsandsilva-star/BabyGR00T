#!/usr/bin/env bash
# Autonomous chain: let the v3sn A/B reach its verdict (step >=16000, past v3nv's 14k best),
# snapshot the verdict, then kill it and launch the big 7-task v4 run (SCALENORM=1). Robust to
# v3sn crashing early (loop also exits if the process is gone).
set -uo pipefail
cd /home/research/Projects/BBGr/BabyGR00T
CHAIN=data/logs/chain.log
V3SN=data/logs/agibot_fp_v3sn.log
echo "=== CHAIN start $(date -u +%H:%M:%S) — waiting for v3sn step>=16000 ===" >> "$CHAIN"

while pgrep -f "[p]ython -m scripts.train_oxe" >/dev/null; do
  step=$(grep -oE "^  step +[0-9]+" "$V3SN" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | tail -1)
  [ "${step:-0}" -ge 16000 ] && { echo "=== v3sn reached step ${step} $(date -u +%H:%M:%S) ===" >> "$CHAIN"; break; }
  sleep 60
done

{
  echo "=== v3sn A/B verdict snapshot (vs v3nv baseline) $(date -u +%H:%M:%S) ==="
  echo "-- v3sn last val probes --"; grep "val-probe\] step" "$V3SN" | tail -5
  echo "-- v3sn last fp probes --";  grep -oE "fp-probe\] step [0-9]+: σ_g=[0-9.]+.*α=[0-9.]+.*‖z\*‖[^ ]*=[0-9./]+" "$V3SN" | tail -3
} >> "$CHAIN"

echo "=== killing v3sn, launching v4 (SCALENORM=1 APR=0.5) $(date -u +%H:%M:%S) ===" >> "$CHAIN"
pkill -f "[p]ython -m scripts.train_oxe"; sleep 10
pgrep -f "[p]ython -m scripts.train_oxe" >/dev/null && { pkill -9 -f "[p]ython -m scripts.train_oxe"; sleep 5; }

SCALENORM=1 APR=0.5 bash scripts/launch_agibot_v4.sh >> "$CHAIN" 2>&1
echo "=== CHAIN done — v4 launched $(date -u +%H:%M:%S) ===" >> "$CHAIN"
