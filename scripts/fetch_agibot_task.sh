#!/usr/bin/env bash
# Fetch + convert ONE AgiBot task end-to-end, keeping only what we need.
#   download observations/<task>/*.tar  ->  extract only */videos/head_color.mp4  ->  delete tar
#   extract <task>/* proprio from the local proprio tar (no download)
#   run convert_agibot_task.py  ->  data/oxe/agibot_task<id>
# Disk-frugal: tars are deleted right after video extraction; only ~8-15G/ task is kept.
set -euo pipefail
TASK="${1:?usage: fetch_agibot_task.sh <task_id>}"
ROOT=/home/research/Projects/BBGr/BabyGR00T
PY=/home/research/Projects/BBGr/.venv/bin/python
REPO=agibot-world/AgiBotWorld-Alpha
OBS_DIR=$ROOT/data/agibot_raw/obs$TASK
PROP_DIR=$ROOT/data/agibot_proprio
PROP_TAR=$ROOT/data/agibot_raw/proprio_stats/648533-923022.tar
OUT=$ROOT/data/oxe/agibot_task$TASK
cd "$ROOT"
source ~/.hf_env 2>/dev/null || true
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1

if [ -d "$OUT/meta" ]; then echo "[$TASK] already converted -> $OUT  SKIP"; exit 0; fi
mkdir -p "$OBS_DIR" "$PROP_DIR"

echo "[$TASK] === proprio (from local tar, no download) ==="
if ! ls "$PROP_DIR/$TASK"/*/proprio_stats.h5 >/dev/null 2>&1; then
  tar xf "$PROP_TAR" -C "$PROP_DIR" "$TASK" 2>/dev/null || \
  tar xf "$PROP_TAR" -C "$PROP_DIR" --wildcards "$TASK/*"
fi
echo "[$TASK] proprio eps: $(ls -d "$PROP_DIR/$TASK"/*/ 2>/dev/null | wc -l)"

echo "[$TASK] === list + download obs tars ==="
TARS=$($PY - "$REPO" "$TASK" <<'PY'
import sys
from huggingface_hub import HfApi
repo, task = sys.argv[1], sys.argv[2]
for f in HfApi().list_repo_files(repo_id=repo, repo_type="dataset"):
    if f.startswith(f"observations/{task}/") and f.endswith(".tar"):
        print(f)
PY
)
for f in $TARS; do
  echo "[$TASK]   downloading $f"
  LP=$($PY - "$REPO" "$f" "$ROOT/data/agibot_raw/_tars" <<'PY'
import sys
from huggingface_hub import hf_hub_download
# local_dir => file lands as a real copy (no cache blob left behind to fill disk)
print(hf_hub_download(repo_id=sys.argv[1], repo_type="dataset", filename=sys.argv[2],
                      local_dir=sys.argv[3]))
PY
)
  echo "[$TASK]   extracting head_color from $(basename "$f")"
  tar xf "$LP" -C "$OBS_DIR" --wildcards '*/videos/head_color.mp4' 2>/dev/null || true
  rm -f "$LP"                              # drop the big tar (mostly depth pngs we don't need)
done
echo "[$TASK] obs eps extracted: $(ls -d "$OBS_DIR"/*/ 2>/dev/null | wc -l)  size: $(du -sh "$OBS_DIR" 2>/dev/null | cut -f1)"

echo "[$TASK] === convert ==="
$PY -m scripts.convert_agibot_task \
  --obs-dir "$OBS_DIR" --proprio-dir "$PROP_DIR/$TASK" \
  --task-json data/agibot_raw/task_info/task_$TASK.json --out-dir "$OUT"
echo "[$TASK] DONE -> $OUT  ($(du -sh "$OUT" 2>/dev/null | cut -f1))"
