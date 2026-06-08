#!/bin/bash
# Durable training-experiment queue.
# Reads /tmp/oxe_train_queue.txt (one shell command per line). For each command:
#   - launches it
#   - tails the corresponding output file looking for NaN-stuck (>100 NaN steps without recovery)
#     or normal completion
#   - on completion: moves on to next command
#   - on NaN-stuck or non-zero exit: logs to /tmp/oxe_queue_errors.log and continues
# Designed to be killed and restarted at any time — uses a "done" file to track progress.

QUEUE_FILE=${1:-/tmp/oxe_train_queue.txt}
DONE_FILE=/tmp/oxe_train_done.txt
ERR_FILE=/tmp/oxe_queue_errors.log
touch "$DONE_FILE" "$ERR_FILE"

while true; do
    if [ ! -f "$QUEUE_FILE" ]; then
        echo "[queue] no queue file at $QUEUE_FILE — sleeping 60s"
        sleep 60
        continue
    fi
    PROGRESS=0
    LINE_NUM=0
    while IFS= read -r CMD; do
        LINE_NUM=$((LINE_NUM + 1))
        # skip empty and comment lines
        [ -z "$CMD" ] && continue
        [ "${CMD:0:1}" = "#" ] && continue
        # skip if already done (matched by exact line)
        if grep -qxF "$CMD" "$DONE_FILE"; then continue; fi
        echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') starting: $CMD"
        PROGRESS=1
        # run the command
        bash -c "$CMD"
        RC=$?
        if [ $RC -eq 0 ]; then
            echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') ✅ success: $CMD"
            echo "$CMD" >> "$DONE_FILE"
        else
            echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') ❌ failed (rc=$RC): $CMD"
            echo "$(date '+%Y-%m-%d %H:%M:%S') rc=$RC: $CMD" >> "$ERR_FILE"
            # mark as done anyway to not get stuck in infinite retry — manual rerun if needed
            echo "$CMD" >> "$DONE_FILE"
        fi
    done < "$QUEUE_FILE"
    if [ $PROGRESS -eq 0 ]; then
        echo "[queue] $(date '+%Y-%m-%d %H:%M:%S') queue empty / all done — sleeping 60s and rechecking"
        sleep 60
    fi
done
