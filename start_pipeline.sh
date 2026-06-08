#!/bin/bash
# Post-reboot one-shot launcher for the BabyGR00T VAE-TRM pipeline.
#   - verifies the GPU is actually usable (driver/lib match + torch sees CUDA)
#   - launches run_pipeline.sh FULLY DETACHED (setsid) so it survives the
#     SSH/Sunshine session ending. Logs to pipeline.log.
#   - 3x cache augmentation (~300GB) via the stock N_VIS_AUG=3 / N_EPS=1800.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY=/home/research/Projects/BBGr/.venv/bin/python
export BABYGROOT_DATA="${BABYGROOT_DATA:-$REPO/data}"
# stock defaults already target ~300GB: N_EPS=1800, N_VIS_AUG=3 (4 variants/chunk)
export N_VIS_AUG="${N_VIS_AUG:-3}"

echo "=== GPU preflight ==="
if ! nvidia-smi >/dev/null 2>&1; then
    echo "FAIL: nvidia-smi still errors — driver/library mismatch not resolved."
    nvidia-smi 2>&1 | head -3
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "FAIL: torch cannot see CUDA."
    "$PY" -c "import torch; print('cuda avail:', torch.cuda.is_available())" 2>&1 | tail -3
    exit 1
fi
echo "torch CUDA OK"

echo "=== free disk (need ~300GB for the cache) ==="
df -h "$BABYGROOT_DATA" 2>/dev/null || df -h "$REPO"

echo "=== launching pipeline detached -> $REPO/pipeline.log ==="
# venv-on-PATH so 'python' inside run_pipeline.sh resolves to the venv interpreter
export PATH="/home/research/Projects/BBGr/.venv/bin:$PATH"
setsid bash -c "BABYGROOT_DATA='$BABYGROOT_DATA' N_VIS_AUG='$N_VIS_AUG' ./run_pipeline.sh > pipeline.log 2>&1" </dev/null >/dev/null 2>&1 &
sleep 1
echo "started. PID group launched. tail with:  tail -f $REPO/pipeline.log"
