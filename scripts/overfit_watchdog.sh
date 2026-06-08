#!/bin/bash
# Auto-kill training when held-out generalization degrades (overfitting) or plateaus.
# The trainer saves the best-cosine checkpoint, so killing loses nothing.
#   bash scripts/overfit_watchdog.sh <log> <train_pid>
# Triggers (on the held-out test-cosine generalization metric):
#   OVERFIT  — cosine falls >DROP% below its best while train_acc>95%, for BAD_N consecutive evals
#   PLATEAU  — no new best cosine (gain >MINDELTA%) for PATIENCE evals (generous backstop)
LOG="${1:-data/logs/sudoku_extreme_train.log}"
PID="${2:?need train PID}"
DROP=1.0; MINDELTA=0.3; BAD_N=2; PATIENCE=12
best=0; best_step=0; bad=0; noimp=0
tail -n +1 -F "$LOG" 2>/dev/null | while IFS= read -r line; do
  [ "${line#step}" = "$line" ] && continue                       # only step lines
  cos=$(printf '%s' "$line" | grep -oP 'test cosine=\K[0-9.]+'); [ -z "$cos" ] && continue
  tr=$(printf '%s'  "$line" | grep -oP 'train_acc=\K[0-9.]+')
  step=$(printf '%s' "$line"| grep -oP '^step[[:space:]]+\K[0-9]+')
  imp=$(awk -v c="$cos" -v b="$best" -v d="$MINDELTA" 'BEGIN{print (c>b+d)?1:0}')
  if [ "$imp" = 1 ]; then best=$cos; best_step=$step; noimp=0; bad=0; continue; fi
  noimp=$((noimp+1))
  deg=$(awk -v c="$cos" -v b="$best" -v dr="$DROP" -v t="${tr:-0}" 'BEGIN{print (c<b-dr && t>95)?1:0}')
  if [ "$deg" = 1 ]; then bad=$((bad+1)); else bad=0; fi
  if [ "$bad" -ge "$BAD_N" ]; then
    echo "[watchdog] OVERFIT @ step $step: test cosine ${cos}% fell >${DROP}% below best ${best}% (step $best_step), train_acc ${tr}% — killing PID $PID"
    CKPT=/home/research/Projects/BBGr/BabyGR00T/data/ckpt
    cp -f "$CKPT/sudoku_v2_best.pt" "$CKPT/sudoku_v2_best_final.pt" 2>/dev/null  # snapshot before kill
    kill -9 "$PID" 2>/dev/null; break
  fi
  if [ "$noimp" -ge "$PATIENCE" ]; then
    echo "[watchdog] PLATEAU @ step $step: no cosine gain >${MINDELTA}% for $PATIENCE evals (best ${best}% @ step $best_step) — killing PID $PID"
    CKPT=/home/research/Projects/BBGr/BabyGR00T/data/ckpt
    cp -f "$CKPT/sudoku_v2_best.pt" "$CKPT/sudoku_v2_best_final.pt" 2>/dev/null  # snapshot before kill
    kill -9 "$PID" 2>/dev/null; break
  fi
done
